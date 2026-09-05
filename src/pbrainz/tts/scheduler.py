"""Synchronized synthesis queue and dialogue playback scheduler."""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Awaitable, Callable
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from pbrainz.config import Settings

from ..conversation_runtime import ConversationRuntime, Utterance, UtteranceState
from .audio import AudioOutput, PiperProvider
from .models import (
    FailureCallback,
    SpeechCallback,
    SynthesizedAudio,
    SynthesizedAudioChunk,
    TTSException,
)

LOGGER = logging.getLogger(__name__)


class SynthesisQueue:
    """Fixed worker count and bounded pending queue for Piper inference."""

    def __init__(
        self,
        provider: PiperProvider,
        workers: int,
        capacity: int,
        on_result: Callable[
            [Utterance, SynthesizedAudio | None, Exception | None], Awaitable[None]
        ],
    ) -> None:
        self.provider = provider
        self.worker_count = max(1, min(int(workers), 4))
        self.queue: asyncio.Queue[Utterance] = asyncio.Queue(maxsize=max(1, int(capacity)))
        self.on_result = on_result
        self._tasks: list[asyncio.Task[None]] = []
        self._executor: ThreadPoolExecutor | None = None
        self.active_workers = 0

    async def start(self) -> None:
        if self._tasks:
            return
        self._executor = ThreadPoolExecutor(
            max_workers=self.worker_count,
            thread_name_prefix="pbrainz-piper",
        )
        self._tasks = [
            asyncio.create_task(self._worker(), name=f"piper-worker-{i}")
            for i in range(self.worker_count)
        ]

    async def stop(self) -> None:
        tasks = self._tasks
        self._tasks = []
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        while not self.queue.empty():
            self.queue.get_nowait()
            self.queue.task_done()
        if self._executor is not None:
            self._executor.shutdown(wait=False, cancel_futures=True)
            self._executor = None

    def submit(self, utterance: Utterance) -> bool:
        if not self._tasks:
            return False
        try:
            self.queue.put_nowait(utterance)
            return True
        except asyncio.QueueFull:
            return False

    async def _worker(self) -> None:
        loop = asyncio.get_running_loop()
        while True:
            utterance = await self.queue.get()
            self.active_workers += 1
            try:
                if self._executor is None:
                    raise TTSException("synthesis executor is not running")
                audio = await loop.run_in_executor(
                    self._executor,
                    self.provider.synthesize,
                    utterance.text,
                    utterance.voice_binding,
                )
                await self.on_result(utterance, audio, None)
            except asyncio.CancelledError:
                raise
            except Exception as error:
                await self.on_result(utterance, None, error)
            finally:
                self.active_workers -= 1
                self.queue.task_done()


class SpeechScheduler:
    """Coordinates floor ownership, look-ahead, FIFO, and bounded playback."""

    def __init__(self, provider: PiperProvider, output: AudioOutput, settings: Settings) -> None:
        self.provider = provider
        self.output = output
        self.settings = settings
        self.runtimes: dict[str, ConversationRuntime] = {}
        self._callbacks: dict[
            str, tuple[SpeechCallback | None, SpeechCallback | None, FailureCallback | None]
        ] = {}
        self._play_tasks: set[asyncio.Task[None]] = set()
        self._play_task_conversations: dict[asyncio.Task[None], str] = {}
        self._wake = asyncio.Event()
        self._stopped = False
        self._loop_task: asyncio.Task[None] | None = None
        self._audio: dict[str, SynthesizedAudio] = {}
        self._streaming_ids: set[str] = set()
        self._playback_semaphore = asyncio.Semaphore(
            max(1, min(settings.tts_max_simultaneous_playback, 8))
        )
        self._active_processes: dict[str, asyncio.subprocess.Process] = {}
        self.synthesis = SynthesisQueue(
            provider,
            settings.tts_synthesis_workers,
            max(2, settings.tts_max_generated_ahead),
            self._on_synthesis_result,
        )

    async def start(self) -> None:
        if self._loop_task:
            return
        self._stopped = False
        await self.synthesis.start()
        self._loop_task = asyncio.create_task(self._run(), name="pbrainz-speech-scheduler")

    async def stop(self) -> None:
        self._stopped = True
        self._wake.set()
        if self._loop_task:
            self._loop_task.cancel()
            await asyncio.gather(self._loop_task, return_exceptions=True)
            self._loop_task = None
        for task in tuple(self._play_tasks):
            task.cancel()
        if self._play_tasks:
            await asyncio.gather(*self._play_tasks, return_exceptions=True)
        self._play_task_conversations.clear()
        for process in tuple(self._active_processes.values()):
            if process.returncode is None:
                process.terminate()
        await self.synthesis.stop()
        for audio in self._audio.values():
            audio.path.unlink(missing_ok=True)
        self._audio.clear()
        self._streaming_ids.clear()
        self.runtimes.clear()
        self._callbacks.clear()

    async def cancel_conversation(self, conversation_id: str) -> int:
        """Cancel queued/local audio for a stale or rejected conversation."""

        runtime = self.runtimes.get(conversation_id)
        if runtime is None:
            return 0
        utterance_ids = {utterance.utterance_id for utterance in runtime.utterances}
        for utterance_id in utterance_ids:
            audio = self._audio.pop(utterance_id, None)
            if audio is not None:
                audio.path.unlink(missing_ok=True)
            self._streaming_ids.discard(utterance_id)
            self._callbacks.pop(utterance_id, None)
        # Close before canceling tasks so their cancellation cleanup cannot
        # start another queued line while the stale conversation is leaving.
        runtime.close()
        tasks = [
            task
            for task, task_conversation_id in self._play_task_conversations.items()
            if task_conversation_id == conversation_id
        ]
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self.runtimes.pop(conversation_id, None)
        self._wake.set()
        LOGGER.info(
            "TTS conversation canceled conversation=%s utterances=%s active_tasks=%s",
            conversation_id,
            len(utterance_ids),
            len(tasks),
        )
        return len(utterance_ids)

    def register_runtime(self, runtime: ConversationRuntime) -> None:
        self.runtimes[runtime.conversation_id] = runtime

    async def enqueue(
        self,
        utterance: Utterance,
        *,
        on_started: SpeechCallback | None = None,
        on_finished: SpeechCallback | None = None,
        on_failed: FailureCallback | None = None,
        streaming: bool | None = None,
        wait_for_capacity: bool = False,
    ) -> bool:
        if not self._loop_task or not self.provider.can_synthesize(utterance.voice_binding):
            return False
        runtime = self.runtimes.get(utterance.conversation_id)
        if runtime is None:
            runtime = ConversationRuntime(
                utterance.conversation_id,
                participants=(utterance.speaker_key,),
                max_generated_ahead=self.settings.tts_max_generated_ahead,
                max_tts_ready_ahead=self.settings.tts_max_tts_ready_ahead,
            )
            self.runtimes[utterance.conversation_id] = runtime
        else:
            runtime.add_participant(utterance.speaker_id, utterance.speaker_kind)
        if not await self._wait_for_capacity(runtime, wait_for_capacity):
            return False
        runtime.enqueue_generated(utterance)
        self._callbacks[utterance.utterance_id] = (on_started, on_finished, on_failed)
        use_streaming = (
            self._streaming_available(utterance)
            if streaming is None
            else streaming is True and self._streaming_available(utterance)
        )
        if use_streaming:
            self._streaming_ids.add(utterance.utterance_id)
        else:
            self._pump(runtime)
        self._wake.set()
        return True

    async def _wait_for_capacity(
        self, runtime: ConversationRuntime, wait_for_capacity: bool
    ) -> bool:
        """Apply backpressure to streamed text instead of dropping later lines."""

        if runtime.can_generate():
            return True
        if not wait_for_capacity:
            return False
        timeout = max(1.0, min(float(self.settings.tts_synthesis_timeout), 60.0))
        deadline = asyncio.get_running_loop().time() + timeout
        while not runtime.can_generate():
            remaining = deadline - asyncio.get_running_loop().time()
            if remaining <= 0:
                return False
            self._wake.clear()
            if runtime.can_generate():
                continue
            try:
                await asyncio.wait_for(self._wake.wait(), timeout=remaining)
            except TimeoutError:
                return runtime.can_generate()
        return True

    def _streaming_available(self, utterance: Utterance) -> bool:
        provider_check = getattr(self.provider, "can_stream", None)
        return bool(
            callable(provider_check)
            and provider_check(utterance.voice_binding)
            and getattr(self.output, "streaming_available", False)
        )

    async def _on_synthesis_result(
        self,
        utterance: Utterance,
        audio: SynthesizedAudio | None,
        error: Exception | None,
    ) -> None:
        runtime = self.runtimes.get(utterance.conversation_id)
        if not runtime:
            if audio:
                audio.path.unlink(missing_ok=True)
            return
        if error or not audio:
            runtime.mark_complete(utterance.utterance_id, failed=True)
            callbacks = self._callbacks.pop(utterance.utterance_id, (None, None, None))
            if callbacks[2]:
                await _safe_callback(callbacks[2], utterance, error or TTSException("no audio"))
            self._pump(runtime)
            self._wake.set()
            return
        utterance.estimated_or_actual_duration_ms = audio.duration_ms
        self._audio[utterance.utterance_id] = audio
        runtime.mark_tts_ready(utterance.utterance_id, audio.duration_ms)
        self._pump(runtime)
        self._wake.set()

    def _pump(self, runtime: ConversationRuntime) -> None:
        while (
            runtime.generated_queue and runtime.tts_ready_ahead_count < runtime.max_tts_ready_ahead
        ):
            utterance_id = runtime.generated_queue[0]
            # Streaming utterances are consumed by _run one at a time. If
            # _pump submits them to the WAV queue here, later lines switch
            # pipelines and can be delayed, reordered, or effectively lost.
            if utterance_id in self._streaming_ids:
                break
            utterance = runtime.mark_tts_pending(utterance_id)
            if not self.synthesis.submit(utterance):
                # The bounded queue is backpressure, not a reason to spawn a
                # new worker. Leave the item generated so a later result can retry.
                runtime.generated_queue.appendleft(utterance_id)
                utterance.tts_state = UtteranceState.GENERATED
                utterance.playback_state = UtteranceState.GENERATED
                break

    async def _run(self) -> None:
        while not self._stopped:
            did_work = False
            for runtime in tuple(self.runtimes.values()):
                if runtime.state != "active":
                    continue
                stream_utterance = self._next_streaming(runtime)
                if stream_utterance and runtime.can_start(stream_utterance):
                    runtime.mark_tts_pending(stream_utterance.utterance_id)
                    self._streaming_ids.discard(stream_utterance.utterance_id)
                    runtime.mark_waiting_for_floor(stream_utterance.utterance_id)
                    task = asyncio.create_task(
                        self._play_stream(runtime, stream_utterance),
                        name=f"speech-stream-{stream_utterance.utterance_id}",
                    )
                    self._track_play_task(task, runtime.conversation_id)
                    did_work = True
                    continue
                utterance = runtime.next_ready()
                if utterance and runtime.can_start(utterance):
                    runtime.mark_waiting_for_floor(utterance.utterance_id)
                    task = asyncio.create_task(
                        self._play(runtime, utterance), name=f"speech-{utterance.utterance_id}"
                    )
                    self._track_play_task(task, runtime.conversation_id)
                    did_work = True
            if not did_work:
                self._wake.clear()
                await self._wake.wait()

    def _next_streaming(self, runtime: ConversationRuntime) -> Utterance | None:
        """Return the first queued utterance selected for streaming playback."""

        queued = set(runtime.generated_queue)
        if not queued:
            return None
        for utterance in runtime.utterances:
            if utterance.utterance_id in queued and utterance.utterance_id in self._streaming_ids:
                return utterance
        return None

    def _track_play_task(self, task: asyncio.Task[None], conversation_id: str) -> None:
        self._play_tasks.add(task)
        self._play_task_conversations[task] = conversation_id

        def cleanup(done: asyncio.Task[None]) -> None:
            self._play_tasks.discard(done)
            self._play_task_conversations.pop(done, None)

        task.add_done_callback(cleanup)

    async def _play_stream(self, runtime: ConversationRuntime, utterance: Utterance) -> None:
        """Synthesize sentence chunks and feed one persistent local player."""

        callbacks = self._callbacks.get(utterance.utterance_id, (None, None, None))
        chunks: asyncio.Queue[tuple[SynthesizedAudioChunk | None, Exception | None]] = (
            asyncio.Queue(maxsize=2)
        )
        producer: asyncio.Task[None] | None = None
        process: asyncio.subprocess.Process | None = None
        stream_started_at = time.perf_counter()
        try:
            stream_factory = getattr(self.provider, "stream_synthesize", None)
            if not callable(stream_factory) or utterance.voice_binding is None:
                raise TTSException("streaming TTS provider is unavailable")
            iterator = stream_factory(utterance.text, utterance.voice_binding)
            producer = asyncio.create_task(
                self._produce_stream(iterator, chunks),
                name=f"piper-stream-producer-{utterance.utterance_id}",
            )

            buffered: list[SynthesizedAudioChunk] = []
            buffered_ms = 0
            target_buffer_ms = max(1, int(self.settings.tts_audio_buffer_ms))
            while not buffered or buffered_ms < target_buffer_ms:
                chunk, error = await chunks.get()
                if error is not None:
                    raise error
                if chunk is None:
                    break
                buffered.append(chunk)
                buffered_ms += chunk.duration_ms
            if not buffered:
                raise TTSException("streaming TTS produced no audio")

            first = buffered[0]
            async with self._playback_semaphore:
                process = await self.output.start_stream(
                    first.sample_rate,
                    first.sample_width,
                    first.sample_channels,
                )
                self._active_processes[utterance.utterance_id] = process
                if not utterance.is_overlap:
                    await asyncio.sleep(max(0, self.settings.tts_natural_gap_ms) / 1000)
                await self._write_stream_chunks(process, buffered, first)
                runtime.mark_playing(utterance.utterance_id)
                self._pump(runtime)
                if callbacks[0]:
                    await _safe_callback(callbacks[0], utterance)
                LOGGER.info(
                    "TTS streaming playback started utterance=%s speaker=%s first_audio_ms=%s "
                    "buffered_ms=%s backend=%s",
                    utterance.utterance_id,
                    utterance.speaker_id,
                    round((time.perf_counter() - stream_started_at) * 1000),
                    buffered_ms,
                    getattr(self.output, "command_name", type(self.output).__name__),
                )

                while True:
                    chunk, error = await chunks.get()
                    if error is not None:
                        raise error
                    if chunk is None:
                        break
                    self._validate_stream_format(chunk, first)
                    await self._write_stream_chunks(process, [chunk], first)
                await self._close_stream_input(process)
                returncode = await process.wait()
                player_detail = await self._read_player_stderr(process)
                if returncode not in (None, 0):
                    detail = f": {player_detail}" if player_detail else ""
                    raise TTSException(
                        f"Linux audio player exited with status {returncode}{detail}"
                    )
                if callbacks[1]:
                    await _safe_callback(callbacks[1], utterance)
                runtime.mark_complete(utterance.utterance_id)
                LOGGER.info(
                    "TTS streaming playback finished utterance=%s speaker=%s elapsed_ms=%s "
                    "backend=%s",
                    utterance.utterance_id,
                    utterance.speaker_id,
                    round((time.perf_counter() - stream_started_at) * 1000),
                    getattr(self.output, "command_name", type(self.output).__name__),
                )
        except asyncio.CancelledError:
            if process is not None and process.returncode is None:
                process.terminate()
                try:
                    await process.wait()
                except (OSError, asyncio.CancelledError):
                    pass
            raise
        except Exception as error:
            if process is not None and process.returncode is None:
                process.terminate()
                try:
                    await process.wait()
                except OSError:
                    pass
            runtime.mark_complete(utterance.utterance_id, failed=True)
            LOGGER.warning(
                "TTS streaming playback failed utterance=%s speaker=%s backend=%s: %s",
                utterance.utterance_id,
                utterance.speaker_id,
                getattr(self.output, "command_name", type(self.output).__name__),
                error,
            )
            if callbacks[2]:
                await _safe_callback(callbacks[2], utterance, error)
        finally:
            if producer is not None:
                producer.cancel()
                await asyncio.gather(producer, return_exceptions=True)
            self._active_processes.pop(utterance.utterance_id, None)
            self._callbacks.pop(utterance.utterance_id, None)
            self._pump(runtime)
            self._wake.set()

    async def _produce_stream(
        self,
        iterator: Any,
        output: asyncio.Queue[tuple[SynthesizedAudioChunk | None, Exception | None]],
    ) -> None:
        """Move blocking Piper iteration off the event loop with backpressure."""

        try:
            while True:
                chunk = await asyncio.to_thread(_next_stream_chunk, iterator)
                if chunk is None:
                    await output.put((None, None))
                    return
                await output.put((chunk, None))
        except asyncio.CancelledError:
            raise
        except Exception as error:
            await output.put((None, error))

    @staticmethod
    async def _read_player_stderr(process: asyncio.subprocess.Process) -> str:
        stderr = getattr(process, "stderr", None)
        if stderr is None or not hasattr(stderr, "read"):
            return ""
        try:
            value = await stderr.read()
        except OSError:
            return ""
        if isinstance(value, bytes):
            return value.decode(errors="replace").strip()[:500]
        return str(value or "").strip()[:500]

    @staticmethod
    def _validate_stream_format(
        chunk: SynthesizedAudioChunk, first: SynthesizedAudioChunk
    ) -> None:
        if (
            chunk.sample_rate != first.sample_rate
            or chunk.sample_width != first.sample_width
            or chunk.sample_channels != first.sample_channels
        ):
            raise TTSException("streaming TTS changed audio format mid-utterance")

    @staticmethod
    async def _write_stream_chunks(
        process: asyncio.subprocess.Process,
        chunks: list[SynthesizedAudioChunk],
        first: SynthesizedAudioChunk,
    ) -> None:
        stdin = process.stdin
        if stdin is None:
            raise TTSException("streaming audio player has no input pipe")
        try:
            for chunk in chunks:
                SpeechScheduler._validate_stream_format(chunk, first)
                stdin.write(chunk.pcm)
                await stdin.drain()
        except (BrokenPipeError, ConnectionResetError, OSError) as error:
            raise TTSException(f"streaming audio player closed its input: {error}") from error

    @staticmethod
    async def _close_stream_input(process: asyncio.subprocess.Process) -> None:
        stdin = process.stdin
        if stdin is None:
            return
        stdin.close()
        try:
            await stdin.wait_closed()
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass

    async def _play(self, runtime: ConversationRuntime, utterance: Utterance) -> None:
        callbacks = self._callbacks.get(utterance.utterance_id, (None, None, None))
        audio = self._audio.pop(utterance.utterance_id, None)
        if audio is None:
            # Lookups remain intentionally private to the scheduler; no audio
            # payload ever crosses the bridge.
            runtime.mark_complete(utterance.utterance_id, failed=True)
            if callbacks[2]:
                await _safe_callback(callbacks[2], utterance, TTSException("audio result was lost"))
            self._callbacks.pop(utterance.utterance_id, None)
            self._pump(runtime)
            self._wake.set()
            return
        process: asyncio.subprocess.Process | None = None
        try:
            if not utterance.is_overlap:
                await asyncio.sleep(max(0, self.settings.tts_natural_gap_ms) / 1000)
            runtime.mark_playing(utterance.utterance_id)
            self._pump(runtime)
            async with self._playback_semaphore:
                process = await self.output.start(audio.path)
                self._active_processes[utterance.utterance_id] = process
                if callbacks[0]:
                    await _safe_callback(callbacks[0], utterance)
                await process.wait()
                if callbacks[1]:
                    await _safe_callback(callbacks[1], utterance)
            runtime.mark_complete(utterance.utterance_id)
        except asyncio.CancelledError:
            if process is not None and process.returncode is None:
                process.terminate()
                try:
                    await process.wait()
                except (OSError, asyncio.CancelledError):
                    pass
            if audio:
                audio.path.unlink(missing_ok=True)
            raise
        except Exception as error:
            runtime.mark_complete(utterance.utterance_id, failed=True)
            if callbacks[2]:
                await _safe_callback(callbacks[2], utterance, error)
        finally:
            self._active_processes.pop(utterance.utterance_id, None)
            audio.path.unlink(missing_ok=True)
            self._callbacks.pop(utterance.utterance_id, None)
            self._pump(runtime)
            self._wake.set()


async def _safe_callback(callback: Callable[..., Awaitable[None]], *args: Any) -> None:
    try:
        await callback(*args)
    except Exception as error:
        LOGGER.warning("TTS lifecycle callback failed: %s", error)


def _next_stream_chunk(iterator: Any) -> SynthesizedAudioChunk | None:
    """Return one blocking-generator item without leaking StopIteration."""

    try:
        return next(iterator)
    except StopIteration:
        return None

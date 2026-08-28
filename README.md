# HoomansLLM

HoomansLLM is a small local LLM gateway for Project Hoomans. It exposes an
OpenAI-compatible Chat Completions API and keeps provider-specific code behind
one async adapter interface.

The first release supports:

- OpenAI Cloud through the official `openai` Python SDK.
- Ollama and LM Studio through separate OpenAI-compatible profiles using the
  same official `openai` Python SDK adapter.
- A separate Custom OpenAI-compatible profile for user-defined LLM routers.
- Google Gemini through the official `google-genai` Python SDK.
- Non-streaming and Server-Sent Events (SSE) streaming responses.

## Requirements

- Python 3.11 or newer
- An API key for a hosted provider, or a local Ollama/LM Studio server

## Run on Linux or macOS

From this directory:

```bash
./scripts/run.sh
```

The first run creates `.venv` and installs the runtime dependencies
automatically. Later runs reuse that private environment and start normally.
No activation or separate install command is required.

## Run on Windows

Run PowerShell from this directory, or double-click `scripts\run.bat`:

```powershell
.\scripts\run.ps1
```

The first run creates `.venv` and installs the runtime dependencies
automatically. Later runs reuse that private environment and start normally.
The launcher finds the project root relative to itself, so it can be started
from any working directory. It invokes `pip` only through the private venv
interpreter and does not install packages into system Python.

The explicit `install.sh` and `install.ps1` scripts remain available for
developers who want the test and lint dependencies installed as well.

The server listens on `http://127.0.0.1:8000` by default. For development, you
can still activate the environment (`source .venv/bin/activate` on Linux/macOS
or `.\.venv\Scripts\Activate.ps1` on Windows) and run `python -m hoomans_llm`
directly.

When started by either launcher, a lightweight native Python `tkinter` control
panel opens automatically. It shows whether the Project Hoomans bridge is
detected and ready, whether the local HoomansLLM bridge worker is running, and
whether each provider is configured. It also allows the default provider,
model, timeout, polling interval, and Project Hoomans bridge setting to be
changed. These settings and API keys are saved in the local SQLite database;
the native panel displays provider keys in plain text, but keys are never
written to the activity log. Set `OPEN_GUI=false` to run the API without the
panel.

The panel's Chat test tab and Test API button send direct provider requests
through `/api/chat`, so providers can be checked before Project Hoomans is
running. The game-facing `/v1/chat/completions` endpoint remains bridge-gated.

The optional TTS tab configures local Piper synthesis and OS audio playback.
TTS is disabled by default. The TTS tab loads Piper's official remote voice
catalog, clearly marks each voice as `Installed` or `Not installed`, and can
download the model and adjacent `.onnx.json` config directly with checksum
verification. Installed files are placed under the configured model root.
The `piper-tts` runtime is included in source installs and release bundles, so
an installed voice can be synthesized locally when an OS audio player such as
`ffplay` is available.
Press `Refresh catalog` to update the remote list. Select a voice and use
`Test selected voice` to play Piper's pre-generated sample without installing
the model; `Install selected voice` downloads it for local synthesis with a
live progress bar and prevents duplicate installations. The
selected catalog language is saved automatically. Piper model metadata is read
from each adjacent `.onnx.json` file. The catalog's gender filter uses
best-effort hints for named voices and labels multi-speaker voices as `mixed`;
Piper does not publish gender metadata. Add a local `voice_metadata.json`
overlay when you want to override a label, such as:

```json
{
  "voices": {
    "en_US-lessac-medium": {"gender": "female", "display_name": "Lessac"}
  }
}
```

Click any catalog column heading to toggle ascending and descending order;
the selected voice remains selected while filters, refreshes, and sorting rerender the list.

Voice presets map the canonical Project Hoomans slots (`VoiceFemale:0` through
`VoiceMale:3`) to local Piper model IDs. Model paths and IDs remain inside
HoomansLLM and never enter Project Hoomans NPC data or the bridge audio path.
Preset selections are saved automatically when a combobox changes. Female and
male preset groups default to their matching gender, with a checkbox on each
group to show all installed genders. Each option includes its gender label,
such as `female`, `male`, `mixed`, or `unknown`.
TTS performance tuning is available in the main `Settings` tab; the TTS tab is
kept focused on voice installation, presets, and playback controls.
When enabled, HoomansLLM plays the local WAV output and sends only compact
speech lifecycle events so the game can synchronize its existing subtitles.
Missing Piper, models, or audio output automatically falls back to text-only
conversation.

`tkinter` is included with standard Windows Python installations. On Linux,
install the distribution's Tk package if it is missing (for example,
`sudo apt install python3-tk` on Debian/Ubuntu).

The first launch creates a portable `data/` directory beside the running
program. It contains the SQLite file for settings, provider credentials, model
catalogs, recent activity, TTS models, and memory. The source launchers and
the AppImage use the directory in which they are installed; this keeps each
copy self-contained and movable. Set `HOOMANSLLM_DB` to override the database
location. Existing legacy databases from the working directory or Linux
`~/.config/HoomansLLM/` are imported once when the portable store has not been
configured. Existing `.env` or `.env.local` values are also imported once
when the database is first created. The database is local-only and should be
kept private.

Provider model catalogs are cached per provider in SQLite and loaded during
startup without network requests. The panel's Refresh models button updates
only the selected provider, so an unavailable local provider cannot slow down
startup or replace another provider's model list.

The source install is intentionally non-editable so the environment contains
the installed application package, which is a cleaner base for a future
executable build. Python virtual environments are isolated but are not
guaranteed to be relocatable across machines; if the project moves to a
different machine or Python installation, rerun the platform installer to
recreate `.venv`.

## Build release artifacts

Release builds must run on their target operating system:

```bash
# Windows PowerShell
python -m pip install ".[build]"
python scripts/build_release.py --target exe

# Linux
python3 -m pip install ".[build]"
python3 scripts/build_release.py --target appimage
```

The Windows build produces a single `.exe`. The Linux build produces an
AppImage and downloads the official `appimagetool` automatically when needed.
Pushing a `v*` tag runs both builds through
`.github/workflows/release.yml` and attaches the artifacts to a GitHub Release.
The Windows `.exe` is not produced on Linux; run the Windows command on a
Windows machine or dispatch the GitHub Actions workflow. Local Linux builds
are written to `dist/release/HoomansLLM-<version>-x86_64.AppImage`, while the
Windows runner writes `dist/release/HoomansLLM-<version>-x86_64.exe`.

## Configuration

Use the native control panel to change settings and provider credentials. The
important controls are:

- default provider and model;
- separate OpenAI Cloud endpoint/API key;
- separate Ollama endpoint/API key (defaults to `http://127.0.0.1:11434/v1`);
- separate LM Studio endpoint/API key (defaults to `http://127.0.0.1:1234/v1`);
- separate Custom endpoint/API key (the endpoint is required; the key is
  optional);
- Gemini API key;
- request timeout, bridge polling, and Project Hoomans bridge state.
- light or dark control-panel theme.

The optional `HOOMANSLLM_DB` process environment variable changes the SQLite
database path. `OPEN_GUI=false` runs the API without opening the native panel.

For bridge debugging, print the persisted recent activity without starting the
server:

```bash
./scripts/run.sh --activity
./scripts/run.sh --activity 100
./scripts/run.sh --activity-json
```

`--activity` shows the newest 50 entries by default and accepts up to 500.
The same flags work with the frozen Windows executable or AppImage. Release
builds keep a terminal attached so bridge state, NPC task messages, provider
responses, and delivery errors are also visible while the GUI/server runs.
Message previews are bounded and API keys are never logged.

The OpenAI, Ollama, LM Studio, and Custom profiles use the OpenAI Chat
Completions protocol. Ollama and LM Studio normally need no API key. The Custom
profile accepts any user-defined OpenAI-compatible endpoint and does not
require a key. Each profile has its own endpoint, credentials, model catalog,
and selected model, so changing one does not change the others.
`OPENAI_BASE_URL` remains available for OpenAI Cloud during environment-based
setup.

For a request using model `default` or `auto`, HoomansLLM prefers
`DEFAULT_PROVIDER` when it is configured and falls back to the first enabled
provider with credentials or a configured local endpoint. This lets a
Gemini-only or local-only setup work without changing the in-game Project
Hoomans integration.

By default, chat requests are accepted only while the PsychopatzCore bridge is
`READY`. HoomansLLM reads the bridge's validated runtime marker from the shared
`PsychopatzBridge/state` directory. When the bridge is ready, the server polls
Project Hoomans' narrow NPC-chat capability, calls the configured provider, and
delivers the reply back through the game tunnel. Set `BRIDGE_REQUIRED=false`
only for standalone provider testing.

The control-panel bridge switch updates the same
`~/Zomboid/Lua/PsychopatzCore_Bridge.txt` setting used by the game and controls
HoomansLLM's polling worker together. Project Hoomans applies that setting
while the game is running, so the profiler is not required for this workflow.

Keep the default localhost bind unless you have separately secured the
service. This initial server does not provide authentication.

## API

Health check:

```bash
curl http://127.0.0.1:8000/health
```

OpenAI-compatible request:

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-4o-mini",
    "messages": [
      {"role": "system", "content": "You are a helpful NPC."},
      {"role": "user", "content": "Say hello in one sentence."}
    ]
  }'
```

Gemini request:

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "provider": "gemini",
    "model": "gemini-2.5-flash",
    "messages": [{"role": "user", "content": "Say hello in one sentence."}]
  }'
```

The `provider` field is a HoomansLLM extension. For convenience, a model can
also be prefixed with `openai/`, `openai:`, `ollama/`, `ollama:`,
`lmstudio/`, `lmstudio:`, `custom/`, `custom:`, `gemini/`, or
`gemini:`; the prefix selects the provider and is removed before the upstream
request.

Set `"stream": true` to receive OpenAI-compatible SSE chunks followed by
`data: [DONE]`.

## In-game NPC chat

Start this server before opening a Project Hoomans conversation. Enable the
Project Hoomans local bridge in the game's bridge configuration, then open an
NPC conversation. A `TYPE TO TALK` input appears beneath the response choices.
Submit a message there; the game sends the current NPC conversation history to
the bridge, HoomansLLM calls the selected provider, and the NPC reply is added
to the conversation log.

The game-side capability is intentionally limited to `pollChat`, `deliverChat`,
and compact `speechStarted`/`speechFinished`/`speechFallback` events in the
`projecthoomans.llm` namespace. Requests are tied to the current runtime ID
and the active NPC conversation. Provider keys remain in
HoomansLLM's local SQLite database and never enter the game tunnel.

The structured game request also carries a compact canonical character card,
relationship snapshot, notable current state, recent dialogue, and the
semantic tools exposed for that NPC. HoomansLLM owns prompt assembly and
conversation memory; Project Hoomans remains authoritative for gameplay. Any
returned order intent is sent back as an untrusted semantic tool call and is
validated by the game's existing command registry before it can be submitted.

## NPC memory and context

NPC memory is separate from the settings database. HoomansLLM creates one
SQLite database per save/world under the configured `memory_root` (by default,
the `memory/` directory beside the settings database). The filename contains a
short hash of the stable Project Zomboid save identifier, while the full
identifier is stored in the database metadata. Memory rows are scoped by the
exact `(world_uuid, player_uuid, npc_uuid)` tuple, so NPCs, players, and saves
cannot bleed into one another.

The store has a single shared `memories` table rather than NPC-specific tables,
conversation sessions/turns, commitments, provenance, and indexes. It uses
SQLite FTS5 when available and falls back to bounded token matching when it is
not. Retrieval is deliberately small and deterministic: recent turns are
bounded, active commitments are always considered, and relevant memories are
ranked before prompt assembly. Embeddings and vector extensions are not
required by this foundation.

Memory writes are failure-contained: a database problem is logged and the
provider request continues without memory. Consolidation runs at the configured
turn threshold or when a session ends. The default consolidator is a small
deterministic extractor for identity, preferences, commitments, and a bounded
conversation summary; it is an explicit replaceable hook for a future extractor
provider.

Detailed retrieval and context diagnostics are disabled by default. Set
`LLM_DIAGNOSTICS=true` only while troubleshooting; API keys are never included.

See [`docs/architecture.md`](docs/architecture.md) for the request flow,
schema, multiplayer boundary, failure model, and extension points.

## Adding a provider

Implement `LLMProvider` in `src/hoomans_llm/providers/`, normalize the provider
response into `CompletionResult` and `StreamEvent`, then register the factory
in `ProviderRegistry`. The HTTP routes and API models do not need to change.

## Development

```bash
source .venv/bin/activate       # Linux/macOS
# .\.venv\Scripts\Activate.ps1 # Windows PowerShell
pytest
ruff check .
```

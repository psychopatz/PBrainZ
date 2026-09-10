# PBrainZ

PBrainZ is a modular local AI gateway and control panel for Project Zomboid
mods. It exposes an OpenAI-compatible Chat Completions API, local voice
playback, memory, and game bridges while keeping provider-specific code behind
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
or `.\.venv\Scripts\Activate.ps1` on Windows) and run `p-brainz`
directly.

When started by either launcher, a lightweight native Python `tkinter` control
panel opens automatically. It shows whether the Project Hoomans bridge is
detected and ready, whether the local PBrainZ bridge worker is running, and
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
Linux PipeWire's `pw-play` or FFmpeg's `ffplay` is available.
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
PBrainZ and never enter Project Hoomans NPC data or the bridge audio path.
Preset selections are saved automatically when a combobox changes. Female and
male preset groups default to their matching gender, with a checkbox on each
group to show all installed genders. Each option includes its gender label,
such as `female`, `male`, `mixed`, or `unknown`.
TTS performance tuning is available in the main `Settings` tab; the TTS tab is
kept focused on voice installation, presets, and playback controls.
When enabled, PBrainZ normally streams ordered sentence-sized PCM chunks from
the Python Piper runtime into a local PipeWire (`pw-play`) or `ffplay` process,
allowing longer responses to begin playback before the entire response has
been synthesized. Ambient response chunks apply backpressure and stale
responses are canceled, so accepted text is not silently discarded. If Python
Piper or a streaming player is unavailable, it falls back to the bounded
full-WAV path;
missing Piper, models, or audio output still automatically falls back to
text-only conversation. Only compact speech lifecycle events cross the bridge;
audio remains local to the client.

`tkinter` is included with standard Windows Python installations. On Linux,
install the distribution's Tk package if it is missing (for example,
`sudo apt install python3-tk` on Debian/Ubuntu).

The first launch creates a portable `data/` directory beside the running
program. It contains the SQLite file for settings, provider credentials, model
catalogs, recent activity, TTS models, and memory. The source launchers and
the AppImage use the directory in which they are installed; this keeps each
copy self-contained and movable. Set `PBRAINZ_DB` to override the database
location. The database is created fresh for each portable copy, is local-only,
and should be kept private. Never include `data/` in a release archive or
upload it with the executable; it is runtime state and may contain provider
credentials.

Provider model catalogs are cached per provider in SQLite and loaded during
startup. By default startup does not make network requests; the panel's
Refresh models button updates only the selected provider. When
`AUTO_REFRESH_MODELS=true` is enabled, startup refreshes only the active
provider before bridge requests begin, preventing a removed AI Horde model
from being selected from an old cache while keeping unavailable local
providers from blocking startup by default.

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

PBrainZ is intentionally pre-1.0 while core memory behavior is still under
development. Local release builds automatically increment the patch version,
so a build changes `0.1.0` to `0.1.1`. Use `--bump minor` or `--bump major` for
milestones, `--bump none` to rebuild without changing the version, or
`--version 0.2.0` to package an explicit version. A failed build restores the
previous version. CI tag builds pass their tag as the explicit version.

The Windows build produces a single `.exe`. The Linux build produces an
AppImage and downloads the official `appimagetool` automatically when needed.
Pushing a `v*` tag runs both builds through
`.github/workflows/release.yml` and attaches the artifacts to a GitHub Release.
The frozen GUI artifacts are windowed applications: Windows does not open a
companion command prompt, and the Windows executable uses the checked-in
PBrainZ icon resource. The release builder refuses to use an output directory
that already contains private runtime data.
The Windows `.exe` is not produced on Linux; run the Windows command on a
Windows machine or dispatch the GitHub Actions workflow. Local Linux builds
are written to `dist/release/PBrainZ-<version>-x86_64.AppImage`, while the
Windows runner writes `dist/release/PBrainZ-<version>-x86_64.exe`.

## Configuration

Use the native control panel to change settings and provider credentials. The
dedicated `Templates` tab manages the prompt profiles used by Project Hoomans
NPC conversations. It includes built-in native-chat and instruct-text profiles
plus reusable custom profiles. The editor exposes the system addendum, context
template, example dialogue, turn prefixes, stop sequences, a resolved preview,
and profile actions (new, duplicate, activate, reset, and delete). Provider
credentials, model catalogs, and generation settings remain in the `Control
panel` tab.

The template context editor supports `{{system}}`, `{{history}}`, `{{user}}`,
`{{assistant}}`, `{{examples}}`, `{{character}}`, `{{char}}`, `{{user_name}}`,
`{{user_prefix}}`, and `{{assistant_prefix}}`. Native-chat profiles preserve
provider-native message roles; instruct profiles render one bounded prompt for
text-template endpoints such as Horde. With the default Native chat profile,
Horde automatically uses the built-in Instruct text profile per request;
Gemini and other providers keep native chat. Activating a custom profile (or
Instruct text explicitly) overrides that automatic Horde selection.

- default provider and model;
- separate OpenAI Cloud endpoint/API key;
- separate Ollama endpoint/API key (defaults to `http://127.0.0.1:11434/v1`);
- separate LM Studio endpoint/API key (defaults to `http://127.0.0.1:1234/v1`);
- separate Custom endpoint/API key (the endpoint is required; the key is
  optional);
- AI Horde endpoint/API key (defaults to `https://oai.aihorde.net/v1`; the key
  is optional and anonymous access is used when it is blank);
- Gemini API key;
- request timeout, bridge polling, and Project Hoomans bridge state.
- light or dark control-panel theme.

The optional `PBRAINZ_DB` process environment variable changes the SQLite
database path. `OPEN_GUI=false` runs the API without opening the native panel.

For bridge debugging, print the persisted recent activity without starting the
server:

```bash
./scripts/run.sh --activity
./scripts/run.sh --activity 100
./scripts/run.sh --activity-json
```

`--activity` shows the newest 50 entries by default and accepts up to 500.
The same flags are accepted by the frozen Windows executable or AppImage, but
those GUI artifacts are intentionally windowed and do not open a terminal for
their output. Use the native activity panel or the source launcher when you
need terminal output. Message previews are bounded and API keys are never
logged.

The OpenAI, Ollama, LM Studio, Custom, and AI Horde profiles use the OpenAI Chat
Completions protocol. Ollama and LM Studio normally need no API key. The Horde
profile uses AI Horde's official OpenAI-compatible gateway and automatically
uses its anonymous key when the optional key field is empty, so it can use
community-hosted free models without local inference hardware or a paid API.
Registering an AI Horde account and supplying its key can improve queue
priority. The Custom profile accepts any user-defined OpenAI-compatible endpoint
and does not require a key. Each profile has its own endpoint, credentials,
model catalog, and selected model, so changing one does not change the others.
The Horde OpenAI gateway is text-completion focused and does not provide native
function/tool calling. PBrainZ keeps that limitation inside the provider
adapter: the shared conversation pipeline still normalizes native calls and
provider text action envelopes into the same untrusted semantic tool calls.
`OPENAI_BASE_URL` remains available for OpenAI Cloud during environment-based
setup.

For a request using model `default` or `auto`, PBrainZ prefers
`DEFAULT_PROVIDER` when it is configured and falls back to the first enabled
provider with credentials or a configured local endpoint. This lets a
Gemini-only or local-only setup work without changing the in-game Project
Hoomans integration.

By default, chat requests are accepted only while the PsychopatzCore bridge is
`READY`. PBrainZ reads the bridge's validated runtime marker from the shared
`PsychopatzBridge/state` directory. When the bridge is ready, the server polls
Project Hoomans' narrow NPC-chat capability, calls the configured provider, and
delivers the reply back through the game tunnel. Set `BRIDGE_REQUIRED=false`
only for standalone provider testing.

The control-panel bridge switch updates the same
`~/Zomboid/Lua/PsychopatzCore_Bridge.txt` setting used by the game and controls
PBrainZ's polling worker together. Project Hoomans applies that setting
while the game is running, so the profiler is not required for this workflow.

The Settings tab exposes the Project Zomboid data directory used by the bridge.
It defaults to the current user's conventional `Zomboid` directory and can be
changed with the folder picker for redirected, portable, or non-standard
installations. PBrainZ derives `Lua/PsychopatzBridge` and the Core bridge
toggle-file path from that directory; explicit `ZOMBOID_BRIDGE_ROOT` and
`ZOMBOID_BRIDGE_CONFIG` environment overrides remain supported.

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

The `provider` field is a PBrainZ extension. For convenience, a model can
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
the bridge, PBrainZ calls the selected provider, and the NPC reply is added
to the conversation log.

The game-side capability is intentionally limited to `pollChat`, `deliverChat`,
`pollConversationSync`, `ackConversationSync`, and compact
`speechStarted`/`speechFinished`/`speechFallback` events in the
`projecthoomans.llm` namespace. Requests are tied to the current runtime ID.
Canonical conversation messages use a bounded, retryable sync outbox, so
closing the conversation UI does not discard a provider response. Provider
keys remain in PBrainZ's local SQLite database and never enter the game tunnel.

The structured game request also carries a compact canonical character card,
relationship snapshot, notable current state, recent dialogue, and the
semantic tools exposed for that NPC. PBrainZ owns prompt assembly and
conversation memory; Project Hoomans remains authoritative for gameplay. Any
returned order or social intent—whether native or text-encoded—is sent back as
an untrusted semantic tool call and is validated by the game's existing command
or relationship authority before it can be applied. Provider errors and
fallback dialogue are marked ineligible for future NPC context and RAG.

## NPC memory and context

NPC memory is separate from the settings database. PBrainZ creates one
SQLite database per save/world under the configured `memory_root` (by default,
the `memory/` directory beside the settings database). The filename contains a
short hash of the stable Project Zomboid save identifier, while the full
identifier is stored in the database metadata. Memory rows are scoped by the
exact `(world_uuid, player_uuid, npc_uuid)` tuple, so NPCs, players, and saves
cannot bleed into one another.

The store has a single shared `memories` table rather than NPC-specific tables,
conversation sessions/turns, commitments, provenance, and indexes. Canonical
turns retain a stable message ID plus Project Zomboid game-day/world-age
fields; duplicate bridge deliveries are ignored. It uses SQLite FTS5 when
available and falls back to bounded token matching when it is not. Retrieval is
deliberately small and deterministic: recent turns are bounded, active
commitments are always considered, and relevant memories plus a small recall
window from older transcript turns are selected before prompt assembly.
Embeddings and vector extensions are not required by this foundation.

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

Implement `LLMProvider` in `src/pbrainz/providers/`, normalize the provider
response into `CompletionResult` and `StreamEvent`, then register the factory
in `ProviderRegistry`. The HTTP routes and API models do not need to change.

## Development

```bash
source .venv/bin/activate       # Linux/macOS
# .\.venv\Scripts\Activate.ps1 # Windows PowerShell
pytest
ruff check .
```

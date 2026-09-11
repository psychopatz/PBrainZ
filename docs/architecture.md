# PBrainZ architecture

This document describes the production-oriented foundation shared by Project
Hoomans and PBrainZ. The profiler is not part of this boundary.

## Responsibilities

```text
Project Hoomans client
  -> canonical Message event + compact conversation context
PsychopatzCore file bridge
  -> pollChat / deliverChat / pollConversationSync / ackConversationSync
PBrainZ bridge pump
  -> canonical-message ingestion + typed memory-primitives ingestion
     -> ConversationService
     -> actor-visible MemoryStore + bounded MemoryRetriever (SQLite)
     -> ContextBuilder (bounded provider messages)
        -> ToolRouter (eligibility, relevance, top-K canonical schemas)
     -> ProviderRegistry (Gemini or OpenAI-compatible adapters)
  <- response text + optional semantic intents
Project Hoomans Commands/Queries
  -> validation and authoritative gameplay mutation
```

Project Hoomans owns gameplay state, conversation presentation, identity
resolution, and all gameplay mutations. PBrainZ owns provider
communication, prompt/context assembly, transcript persistence, retrieval, and
memory consolidation. Python never writes NPC state, relationship values,
inventory, health, tasks, factions, or combat state.

The public transcript belongs to the conversation scene. Memory, relationships,
opinions, and private cognition belong to individual actors. RAG selects
knowledge and capabilities; canonical Project Hoomans systems supply truth,
permissions, and action schemas.

## Structured request

The `conversation_context` object is a bounded adapter payload, not a second
game state database. It contains:

- `world_uuid`: the required request world label; storage is selected from the
  explicit fields below;
- `world_mode`: `singleplayer` or `multiplayer`;
- for single-player, `save_relative_path`: the exact path relative to
  `<Zomboid>/Saves`, such as `Apocalypse/2026-09-06_10-21-49`;
- for client-side multiplayer, `server_instance_id` and the server's stable
  `server_world_generation` (a new generation must be emitted after a wipe or
  reset);
- `player_uuid`: Project Hoomans' stable `characterUUID`;
- `npc_uuid`: the canonical Project Hoomans NPC ID;
- `message_id`: the canonical ID used to make retries idempotent;
- a compact character card, relationship snapshot, notable current state,
  preferences, scene participants, game day, recent dialogue, and current
  player message;
- only the semantic tools exposed for the current NPC/context.

The Python boundary accepts both camelCase and snake_case IDs, but requires all
three scope IDs and the current message for structured NPC dialogue. Direct
`messages` bridge requests do not use NPC memory. Project Hoomans should send
the save/server identity on the first request after the world opens and on any
world transition; PBrainZ deliberately does not guess the active save by
scanning for the newest directory. The bridge adapter can derive the
single-player locator from Project Zomboid's current-save and save-directory
Lua APIs and include it on each structured request.

## Memory schema and isolation

Each logical world has one SQLite file. The filename is derived from a SHA-256
prefix of the canonical identity; the complete identity is recorded in
`world_metadata` and checked when the store opens. The schema intentionally
uses shared tables:

- In single-player, the canonical identity is `sp-v1|<save-relative-path>`
  and the database is under the exact save folder:
  `<Zomboid>/Saves/<mode>/<save>/PBrainZ/memory/`.
- In client-side multiplayer, the canonical identity is
  `mp-v1|<server-instance-id>|<server-world-generation>` and the database
  remains in PBrainZ's external memory root. A server reset must create a new
  generation; a server name alone is not a safe identity.
- An explicit `memory_root` setting remains an intentional override for
  portable/test/custom deployments.
The save-local directory is separate from Project Zomboid's own binary files,
so SQLite WAL/shm files cannot corrupt `map.bin`, `players.db`, or other save
artifacts. Removing the save folder removes its PBrainZ memory as ordinary
contents of that folder. PBrainZ never deletes a save folder itself.

| Table | Purpose |
| --- | --- |
| `world_metadata` | Save/world identity and schema metadata |
| `conversation_sessions` | Session lifecycle and summary pointer |
| `conversation_turns` | Full pbrainz-owned transcript rows with canonical IDs and game dates |
| `memories` | FACT, SOCIAL_EVENT, COMMITMENT, PERSONAL_EVENT, OPINION, DISCOVERY, and CONVERSATION_SUMMARY rows |
| `commitments` | Active/fulfilled commitment state |
| `conversation_episodes` | Bounded coherent scene/event summaries with participants, witnesses, topics, and visibility |
| `day_synopses` | Compact per-NPC, per-game-day developments and structured lists |
| `structured_facts` | Commitments, plans, claims, and other typed statements with truth status |
| `memory_fts` | Optional FTS5 index over active memories |
| `memory_entities` | Per-world NPC/player display-name projection for UI browsing |

Every session, turn, memory, episode, fact, and commitment carries `world_uuid`,
`player_uuid`, and `npc_uuid`. Provenance records source, session, and turn
information. Turns additionally retain `message_id`, `game_day`,
`world_age_hours`, and speaker identity. Memories and episodes carry explicit
`PUBLIC`, `OBSERVED`, `TOLD`, `PRIVATE`, or `SYSTEM` visibility. Actor access is
filtered before relevance ranking; a public episode is shared only when its
participant/witness list names the actor. Private cognition never crosses that
boundary.

FTS5 is preferred when the SQLite build supplies it. The fallback is bounded
token matching. No embedding model, `sqlite-vec`, or vector service is needed
for this initial implementation, leaving that as a future `MemoryRetriever`
implementation.

## Context and memory lifecycle

`ConversationService` performs this sequence for each structured turn:

1. Open or reuse the lazy store for the world.
2. Read bounded working memory, the actor-visible day synopsis/facts, and a
   scene/focal-actor context.
3. Gate historical retrieval with a cheap deterministic cue check. For
   historical questions, filter by actor visibility, participants, entities,
   and requested kind before bounded relevance ranking.
4. Store the player turn.
5. Build bounded provider messages with these optional sections: relationship,
   scene, day synopsis, structured facts, relevant preferences, relevant
   memories, notable state, and available tools.
   The core NPC rules, canonical character card, and current player message are
   retained under the hard character budget.
6. Call the selected provider and store the assistant turn.
7. At a bounded consolidation boundary, update the episode, day synopsis,
   structured facts, and compact searchable summary. Raw turns remain outside
   the prompt and are never deleted merely because a new game day begins.

Canonical messages from authored conversations and background dialogue enter
the same pbrainz store through the bridge sync outbox. The LLM path may write
the player and provider rows immediately so the next request has low latency;
the later canonical outbox delivery is collapsed by `message_id`, so it does
not create a second copy.

Typed gameplay memory primitives use the same bridge poll/ack channel but a
separate bounded client outbox. Project Hoomans supplies only a primitive type,
stable player/NPC IDs, authoritative source metadata, and a structured event
time. PBrainZ validates the identity and date, renders deterministic memory
content, and assigns an idempotent key. The current primitives are
`first_meeting` and `pre_outbreak_relationship`; new primitive renderers can be
registered without changing the bridge protocol. The game calendar is emitted
as numeric year/month/day/hour/minute fields plus `game_day` and
`world_age_hours`, with an ISO value and human label for inspection.

The default consolidator creates a compact summary and extracts only clear
identity facts, preferences, and commitments. `MemoryStore`,
`MemoryRetriever`, `ContextBuilder`, and `ConversationConsolidator` are
replaceable boundaries; a future provider-backed extractor can be added
without changing the game bridge contract.

## Semantic gameplay boundary

Project Hoomans advertises canonical function schemas such as `social_react`
and `order_follow`. `ToolRouter` applies deterministic eligibility first,
rejecting malformed, client-only, internal, or game-ineligible cards; it then
selects a small relevant top-K subset while preserving the exact supplied
schema. Relevance is not authorization. The bridge pump forwards a provider
tool call only if its name was exposed in that turn. On the game side, order
calls are checked against the current tool list, the command registry, and
`PNC.Client.ExecuteCompanionCommand`; server/client authority then performs the
existing validation. The NPC ID always comes from the active conversation, not
from model arguments.

`social_react` is currently recorded as an observable intent because the mod has
no generic social-event mutation endpoint. It is deliberately not translated
into a relationship write. A future authoritative SocialEvent command can be
registered behind the same boundary.

## Single-player and multiplayer

In single-player the local client publishes the pending conversation and typed
memory primitives through the local PsychopatzCore bridge and PBrainZ returns
the response. In multiplayer, the client still supplies only its scoped
context and semantic request; server-authoritative lifelong relationships are
projected to the client as primitives, never written by PBrainZ to server
state. PBrainZ has no authority to mutate server state; any order must travel
through Project Hoomans' normal server-validated command path. Stable player
character IDs keep separate players' memory scopes isolated.

## Failure containment and performance

- The bridge is runtime-ID checked and remains bounded by the existing slot
  protocol.
- The game keeps bounded save-scoped message and primitive outboxes rather than
  per-NPC transcript tables; each holds at most 256 pending records, sends at
  most four messages/eight primitives per acknowledgement window, and removes
  entries only after pbrainz acknowledges their canonical IDs.
- Memory stores are opened lazily per world, use one connection per operation,
  WAL mode, and do not perform network work during application startup.
- Model catalogs remain independent per provider and are refreshed only on the
  existing explicit refresh path/background refresh setting.
- SQLite read/write failures disable memory for that turn but do not discard a
  provider response.
- Provider failures return the existing in-game fallback response and do not
  mutate gameplay state.
- Context length, recent turns, retrieved memories, tool count, message size,
  and consolidation cadence are all bounded by settings.
- Diagnostics are disabled by default and include only non-secret counts,
  scopes, paths, and retrieval reasons when explicitly enabled.

## Testing and future hooks

Python tests cover save/pair isolation, shared schema, FTS fallback behavior,
context budgeting, service consolidation, bridge request handling, active-save
selection, typed primitive persistence, and semantic tool allow-listing.
Project Hoomans has Lua smoke tests for stable save/player/NPC identity,
compact context, typed memory primitive enqueue/ack behavior, and tool
construction.

Future work can add a provider-backed memory extractor, embeddings or a vector
retriever, dynamic Tool RAG, richer authoritative SocialEvent commands,
streamed tool-call handling, and explicit player confirmation for higher-impact
orders without changing the current ownership boundary.

## Date and retention policy

Project Hoomans keeps only the current game day's compact UI history per
player/NPC thread. That cache is allowed to rotate at day rollover because it
is presentation state, not the authoritative transcript. The sync outboxes are
not cleared at rollover: unacknowledged messages and typed primitives remain
until pbrainz stores them. PBrainZ retains the full dated turn rows, one compact episode per
consolidation boundary, per-NPC day synopses, and typed conversational facts.
It can recall relevant older turns/episodes across sessions, but does not
inject an entire previous day's transcript into a normal prompt. The bounded
client primitive outbox is also serialized in the same client-side save-scoped
ModData pattern; it contains IDs, event types, provenance, and calendar
fields—not provider credentials or generated prose.

## Native memory inspection and mock chat

The control panel's `Memories` tab asks the running game for its fresh active
world identity, labels that world as `CURRENT`, and selects it by default. A
manual combobox selection and an `Auto-select current` action remain available.
It enumerates existing world databases and shows a bounded, searchable union
of active summaries, episodes, structured facts, and day synopses. Known NPC
names are projected from per-world entity metadata; UUIDs remain visible in
the detail view for diagnostics. The tab deliberately does not dump raw
conversation turns.
Deletion requires an explicit record selection and confirmation, then removes
only that record inside the selected world database; commitment indexes are
removed with their parent memory.

The `Chat test` tab can opt into a structured NPC/RAG mock. Its world, player,
NPC, game-day, and session identifiers are isolated from gameplay, while the
request still uses `ConversationService`, actor-scoped retrieval, context
budgeting, consolidation, and the configured provider. The sample-memory
button creates an idempotent fixture across all four displayed memory layers,
and the chat output reports which records were retrieved.

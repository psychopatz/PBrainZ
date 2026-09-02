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
  -> canonical-message ingestion + ConversationService
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

- `world_uuid`: `pz-save:<getCurrentSaveName()>`, stable for the save;
- `player_uuid`: Project Hoomans' stable `characterUUID`;
- `npc_uuid`: the canonical Project Hoomans NPC ID;
- `message_id`: the canonical ID used to make retries idempotent;
- a compact character card, relationship snapshot, notable current state,
  preferences, scene participants, game day, recent dialogue, and current
  player message;
- only the semantic tools exposed for the current NPC/context.

The Python boundary accepts both camelCase and snake_case IDs, but requires all
three scope IDs and the current message for structured NPC dialogue. Direct
`messages` bridge requests do not use NPC memory.

## Memory schema and isolation

Each world has one SQLite file. The filename is derived from a SHA-256 prefix of
the world UUID; the complete UUID is recorded in `world_metadata` and checked
when the store opens. The schema intentionally uses shared tables:

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

In single-player the local client publishes the pending conversation through
the local PsychopatzCore bridge and PBrainZ returns the response. In
multiplayer, the client still supplies only its scoped context and semantic
request. PBrainZ has no authority to mutate server state; any order must
travel through Project Hoomans' normal server-validated command path. Stable
player character IDs keep separate players' memory scopes isolated.

## Failure containment and performance

- The bridge is runtime-ID checked and remains bounded by the existing slot
  protocol.
- The game keeps one save-scoped outbox rather than per-NPC transcript tables;
  it holds at most 256 pending messages, sends at most four per poll, and
  removes entries only after pbrainz acknowledges them.
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
context budgeting, service consolidation, bridge request handling, and
semantic tool allow-listing. Project Hoomans has a Lua smoke test for stable
save/player/NPC identity, compact context, and tool construction.

Future work can add a provider-backed memory extractor, embeddings or a vector
retriever, dynamic Tool RAG, richer authoritative SocialEvent commands,
streamed tool-call handling, and explicit player confirmation for higher-impact
orders without changing the current ownership boundary.

## Date and retention policy

Project Hoomans keeps only the current game day's compact UI history per
player/NPC thread. That cache is allowed to rotate at day rollover because it
is presentation state, not the authoritative transcript. The sync outbox is
not cleared at rollover: unacknowledged messages remain until pbrainz stores
them. PBrainZ retains the full dated turn rows, one compact episode per
consolidation boundary, per-NPC day synopses, and typed conversational facts.
It can recall relevant older turns/episodes across sessions, but does not
inject an entire previous day's transcript into a normal prompt. No new data
is serialized into NPC ModData beyond the existing bounded canonical-message
outbox.

## Native memory inspection and mock chat

The control panel's `Memories` tab enumerates existing world databases and
shows a bounded, searchable union of active summaries, episodes, structured
facts, and day synopses. It deliberately does not dump raw conversation turns.
Deletion requires an explicit record selection and confirmation, then removes
only that record inside the selected world database; commitment indexes are
removed with their parent memory.

The `Chat test` tab can opt into a structured NPC/RAG mock. Its world, player,
NPC, game-day, and session identifiers are isolated from gameplay, while the
request still uses `ConversationService`, actor-scoped retrieval, context
budgeting, consolidation, and the configured provider. The sample-memory
button creates an idempotent fixture across all four displayed memory layers,
and the chat output reports which records were retrieved.

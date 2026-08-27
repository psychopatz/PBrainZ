# Project Hoomans LLM foundation

This document describes the production-oriented foundation shared by Project
Hoomans and HoomansLLM. The profiler is not part of this boundary.

## Responsibilities

```text
Project Hoomans client
  -> compact conversation context + stable identities
PsychopatzCore file bridge
  -> pollChat / deliverChat
HoomansLLM bridge pump
  -> ConversationService
     -> MemoryStore + MemoryRetriever (SQLite)
     -> ContextBuilder (bounded provider messages)
     -> ProviderRegistry (Gemini or OpenAI-compatible adapters)
  <- response text + optional semantic intents
Project Hoomans Commands/Queries
  -> validation and authoritative gameplay mutation
```

Project Hoomans owns gameplay state, conversation presentation, identity
resolution, and all gameplay mutations. HoomansLLM owns provider
communication, prompt/context assembly, transcript persistence, retrieval, and
memory consolidation. Python never writes NPC state, relationship values,
inventory, health, tasks, factions, or combat state.

## Structured request

The `conversation_context` object is a bounded adapter payload, not a second
game state database. It contains:

- `world_uuid`: `pz-save:<getCurrentSaveName()>`, stable for the save;
- `player_uuid`: Project Hoomans' stable `characterUUID`;
- `npc_uuid`: the canonical Project Hoomans NPC ID;
- a compact character card, relationship snapshot, notable current state,
  preferences, recent dialogue, and current player message;
- only the semantic tools exposed for the current NPC/context.

The Python boundary accepts both camelCase and snake_case IDs for compatibility,
but requires all three scope IDs and the current message for structured NPC
dialogue. The older direct `messages` bridge shape remains supported for
backward compatibility and does not use NPC memory.

## Memory schema and isolation

Each world has one SQLite file. The filename is derived from a SHA-256 prefix of
the world UUID; the complete UUID is recorded in `world_metadata` and checked
when the store opens. The schema intentionally uses shared tables:

| Table | Purpose |
| --- | --- |
| `world_metadata` | Save/world identity and schema metadata |
| `conversation_sessions` | Session lifecycle and summary pointer |
| `conversation_turns` | Bounded transcript history for a scoped session |
| `memories` | FACT, SOCIAL_EVENT, COMMITMENT, PERSONAL_EVENT, OPINION, DISCOVERY, and CONVERSATION_SUMMARY rows |
| `commitments` | Active/fulfilled commitment state |
| `memory_fts` | Optional FTS5 index over active memories |

Every session, turn, memory, and commitment carries `world_uuid`,
`player_uuid`, and `npc_uuid`. Provenance records source, session, and turn
information. Retrieval filters by the complete tuple before ranking; no query
can retrieve an unrelated NPC's memory.

FTS5 is preferred when the SQLite build supplies it. The fallback is bounded
token matching. No embedding model, `sqlite-vec`, or vector service is needed
for this initial implementation, leaving that as a future `MemoryRetriever`
implementation.

## Context and memory lifecycle

`ConversationService` performs this sequence for each structured turn:

1. Open or reuse the lazy store for the world.
2. Read a bounded recent-turn window and retrieve a small set of scoped
   memories, including active commitments.
3. Store the player turn.
4. Build bounded provider messages with these optional sections: relationship,
   relevant preferences, relevant memories, notable state, and available tools.
   The core NPC rules, canonical character card, and current player message are
   retained under the hard character budget.
5. Call the selected provider and store the assistant turn.
6. Consolidate at the configured turn threshold or on session end.

The default consolidator creates a compact summary and extracts only clear
identity facts, preferences, and commitments. `MemoryStore`,
`MemoryRetriever`, `ContextBuilder`, and `ConversationConsolidator` are
replaceable boundaries; a future provider-backed extractor can be added
without changing the game bridge contract.

## Semantic gameplay boundary

Project Hoomans may advertise valid function tools such as `social_react` and
`order_follow`. Names are normalized to the OpenAI function-name grammar. The
bridge pump forwards a provider tool call only if its name was exposed in that
turn. On the game side, order calls are checked against the current tool list,
the command registry, and `PNC.Client.ExecuteCompanionCommand`; server/client
authority then performs the existing validation. The NPC ID always comes from
the active conversation, not from model arguments.

`social_react` is currently recorded as an observable intent because the mod has
no generic social-event mutation endpoint. It is deliberately not translated
into a relationship write. A future authoritative SocialEvent command can be
registered behind the same boundary.

## Single-player and multiplayer

In single-player the local client publishes the pending conversation through
the local PsychopatzCore bridge and HoomansLLM returns the response. In
multiplayer, the client still supplies only its scoped context and semantic
request. HoomansLLM has no authority to mutate server state; any order must
travel through Project Hoomans' normal server-validated command path. Stable
player character IDs keep separate players' memory scopes isolated.

## Failure containment and performance

- The bridge is runtime-ID checked and remains bounded by the existing slot
  protocol.
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
context budgeting, service consolidation, legacy bridge compatibility, and
semantic tool allow-listing. Project Hoomans has a Lua smoke test for stable
save/player/NPC identity, compact context, and tool construction.

Future work can add a provider-backed memory extractor, embeddings or a vector
retriever, richer authoritative SocialEvent commands, streamed tool-call
handling, and explicit player confirmation for higher-impact orders without
changing the current ownership boundary.

// Tables for the simulation core. Registered in convex/schema.ts by
// patches/orion-sim-core-hooks.patch.
//
// simState: one versioned document per world holding the whole core State
// (places, stock, needs, trust, facts, quests, the PRNG). One document means
// every event's effects commit atomically and a replay starts from one value.
// simEvents: the append-only log of every applied (or refused) event.
import { defineTable } from 'convex/server';
import { v } from 'convex/values';

export const simValue = v.union(v.number(), v.string(), v.boolean());
export const simEffect = v.object({ op: v.string(), target: v.string(), value: simValue });

export const simTables = {
  simState: defineTable({
    worldId: v.id('worlds'),
    version: v.number(),
    minute: v.number(),
    seed: v.number(),
    state: v.any(),
  }).index('worldId', ['worldId']),
  simEvents: defineTable({
    worldId: v.id('worlds'),
    eventId: v.string(),
    kind: v.string(),
    source: v.union(v.literal('sim'), v.literal('juniper')),
    title: v.string(),
    place: v.union(v.string(), v.null()),
    effects: v.array(simEffect),
    // PRNG state when the event applied (replay starts from simState + this).
    seed: v.number(),
    // applied | rumor | quest_opened | rejected
    status: v.string(),
    reason: v.union(v.string(), v.null()),
    minute: v.number(),
  })
    .index('worldMinute', ['worldId', 'minute'])
    .index('worldSource', ['worldId', 'source', 'minute']),
};

// Inspect and (re)seed the simulation.
//   npx convex run sim/state:snapshot            -> state + last 20 events
//   npx convex run sim/state:bootstrap '{"seed": 7, "replace": true}'
import { internalMutation, internalQuery } from '../_generated/server';
import { v } from 'convex/values';
import { type State, createState } from './core';
import { SIM_CAST, SIM_SEED } from './seed';
import { defaultWorldId, ensureState, loadState } from './store';

export const snapshot = internalQuery({
  args: { events: v.optional(v.number()) },
  handler: async (ctx, args) => {
    const worldId = await defaultWorldId(ctx);
    if (!worldId) return null;
    const doc = await loadState(ctx, worldId);
    const events = await ctx.db
      .query('simEvents')
      .withIndex('worldMinute', (q) => q.eq('worldId', worldId))
      .order('desc')
      .take(Math.max(0, Math.min(args.events ?? 20, 200)));
    return { worldId, state: doc?.state ?? null, events };
  },
});

// Without replace: creates the state only if missing. With replace: resets the
// world's sim state to the card seed. The event log is kept, so the game clock
// and event numbering carry on from the old state instead of restarting (old
// and new rows never share a minute or an event id).
export const bootstrap = internalMutation({
  args: { seed: v.optional(v.number()), replace: v.optional(v.boolean()) },
  handler: async (ctx, args) => {
    const worldId = await defaultWorldId(ctx);
    if (!worldId) throw new Error('no default world (run init first)');
    const existing = await loadState(ctx, worldId);
    if (existing && args.replace) {
      const state = createState(args.seed ?? SIM_SEED, SIM_CAST);
      const old = existing.state as State;
      state.minute = old.minute;
      state.next_event = old.next_event;
      await ctx.db.replace(existing._id, {
        worldId, version: state.version, minute: state.minute, seed: state.seed, state,
      });
      return { created: false, replaced: true, seed: state.seed };
    }
    const doc = await ensureState(ctx, worldId, args.seed);
    return { created: !existing, replaced: false, seed: doc.seed };
  },
});

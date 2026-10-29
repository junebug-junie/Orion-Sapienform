// Cron entry (patches/orion-sim-core-hooks.patch): advance the default
// world's simulation by SIM_TICK_MINUTES game-minutes every 5 wall-clock
// minutes. The tick runs only while the world is running (embodiment's
// heartbeat keeps it running while Orion's town body is up). Game time pauses
// with the world and does not catch up afterwards, so after any pause game
// hours drift from wall-clock hours: game 08:00 (breakfast) is not 08:00 on
// the wall. `minutes` is capped at a game day (core.MAX_TICK_MINUTES).
import { internalMutation } from '../_generated/server';
import { v } from 'convex/values';
import { type State, tick } from './core';
import { defaultWorldId, ensureState, logEvent, saveState } from './store';

export const SIM_TICK_MINUTES = 5;

export const run = internalMutation({
  args: { minutes: v.optional(v.number()), force: v.optional(v.boolean()) },
  handler: async (ctx, args) => {
    const worldId = await defaultWorldId(ctx);
    if (!worldId) return { ticked: false, reason: 'no default world' };
    const status = await ctx.db
      .query('worldStatus')
      .withIndex('worldId', (q) => q.eq('worldId', worldId))
      .unique();
    if (status?.status !== 'running' && !args.force) {
      return { ticked: false, reason: `world is ${status?.status ?? 'missing'}` };
    }
    const doc = await ensureState(ctx, worldId);
    const out = tick(doc.state as State, args.minutes ?? SIM_TICK_MINUTES);
    await saveState(ctx, doc._id, out.state);
    for (const e of out.events) await logEvent(ctx, worldId, e);
    return { ticked: true, minute: out.state.minute, version: out.state.version, events: out.events.length };
  },
});

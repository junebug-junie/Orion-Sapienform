// Cron entry (patches/orion-sim-core-hooks.patch): every CLOCK_TICK_SECONDS,
// advance the default world's game clock by core.advanceClock -- FAST_CLOCK_X
// game minutes while the town needs no GPU, real time while a chat or an NPC
// model call is open. The tick runs only while the world is running (Orion's
// sessions, or Juniper's), so game time pauses with the world and does not
// catch up afterwards: game hours do not follow wall-clock hours.
// `minutes` (manual runs) skips the clock and is capped at a game day.
import { internalMutation } from '../_generated/server';
import { v } from 'convex/values';
import { type State, advanceClock, tick, townNeedsGpu } from './core';
import { defaultWorldId, ensureState, logEvent, saveState } from './store';

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
    let minutes = args.minutes;
    let busy: boolean | null = null;
    if (minutes === undefined) {
      busy = townNeedsGpu(await ctx.db.get(worldId));
      const next = advanceClock(doc.clock, busy);
      await ctx.db.patch(doc._id, { clock: next.clock });
      minutes = next.minutes;
      if (minutes === 0) return { ticked: false, reason: 'clock carry', busy };
    }
    const out = tick(doc.state as State, minutes);
    await saveState(ctx, doc._id, out.state);
    for (const e of out.events) await logEvent(ctx, worldId, e);
    return { ticked: true, busy, minute: out.state.minute, version: out.state.version, events: out.events.length };
  },
});

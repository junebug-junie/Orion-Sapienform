// Juniper's way in (PR 2: CLI only; the Hub form and bus path arrive in PR 4).
//   npx convex run sim/events:inject '{"event": {...}}'
// Every event goes through core.applyEvent, the one apply path. A refused
// event is logged with status "rejected" and the reason; an unknown source is
// refused outright (no row: the log only holds sources it knows).
import { internalMutation } from '../_generated/server';
import { v } from 'convex/values';
import { type EventInput, SOURCES, SimError, type State, applyEvent } from './core';
import { defaultWorldId, ensureState, logEvent, saveState } from './store';

export const inject = internalMutation({
  args: { event: v.any() },
  handler: async (ctx, args) => {
    const input = (args.event ?? {}) as EventInput;
    if (!SOURCES.includes(input.source)) {
      throw new Error(`unknown source ${String(input.source)} (allowed: ${SOURCES.join(', ')})`);
    }
    const worldId = await defaultWorldId(ctx);
    if (!worldId) throw new Error('no default world (run init first)');
    const doc = await ensureState(ctx, worldId);
    const state = doc.state as State;
    try {
      const out = applyEvent(state, input);
      await saveState(ctx, doc._id, out.state);
      await logEvent(ctx, worldId, out.event);
      return { accepted: true, eventId: out.event.id, status: out.event.status };
    } catch (err) {
      if (!(err instanceof SimError)) throw err;
      await ctx.db.insert('simEvents', {
        worldId, eventId: '', kind: typeof input.kind === 'string' ? input.kind.slice(0, 32) : '',
        source: input.source as 'sim' | 'juniper',
        title: typeof input.title === 'string' ? input.title.slice(0, 200) : '',
        place: null, effects: [], seed: state.rng, status: 'rejected', reason: err.message,
        minute: state.minute,
      });
      return { accepted: false, reason: err.message };
    }
  },
});

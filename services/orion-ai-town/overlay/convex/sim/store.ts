// Shared helpers for the sim Convex functions (not a Convex function module
// by itself: it exports no queries or mutations).
import type { Id } from '../_generated/dataModel';
import type { MutationCtx, QueryCtx } from '../_generated/server';
import { type AppliedEvent, type State, createState } from './core';
import { SIM_CAST, SIM_SEED } from './seed';

export async function defaultWorldId(ctx: QueryCtx): Promise<Id<'worlds'> | null> {
  const status = await ctx.db
    .query('worldStatus')
    .filter((q) => q.eq(q.field('isDefault'), true))
    .unique();
  return status ? status.worldId : null;
}

export async function loadState(ctx: QueryCtx, worldId: Id<'worlds'>) {
  return await ctx.db
    .query('simState')
    .withIndex('worldId', (q) => q.eq('worldId', worldId))
    .unique();
}

// Create the world's sim state from the card seed when it does not exist yet.
export async function ensureState(ctx: MutationCtx, worldId: Id<'worlds'>, seed?: number) {
  const existing = await loadState(ctx, worldId);
  if (existing) return existing;
  const state = createState(seed ?? SIM_SEED, SIM_CAST);
  const id = await ctx.db.insert('simState', {
    worldId, version: state.version, minute: state.minute, seed: state.seed, state,
  });
  return (await ctx.db.get(id))!;
}

export async function saveState(ctx: MutationCtx, docId: Id<'simState'>, state: State) {
  await ctx.db.patch(docId, { version: state.version, minute: state.minute, state });
}

export async function logEvent(ctx: MutationCtx, worldId: Id<'worlds'>, e: AppliedEvent) {
  await ctx.db.insert('simEvents', {
    worldId, eventId: e.id, kind: e.kind, source: e.source, title: e.title, place: e.place,
    effects: e.effects, seed: e.seed, status: e.status, reason: null, minute: e.minute,
  });
}

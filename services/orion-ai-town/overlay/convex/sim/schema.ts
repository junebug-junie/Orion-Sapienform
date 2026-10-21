import {defineTable} from 'convex/server';
import {v} from 'convex/values';
// One bounded world projection permits atomic multi-entity effects. The event,
// scene and prediction histories stay separate and are never copied into it.
export const simTables = {
  simWorlds: defineTable({worldId:v.id('worlds'), state:v.any(), enabled:v.boolean(), lastForecastMinute:v.number(), lastProposalDay:v.optional(v.number())}).index('worldId',['worldId']),
  simEvents: defineTable({worldId:v.id('worlds'), eventId:v.string(), source:v.string(), day:v.number(), payload:v.any()}).index('eventId',['worldId','eventId']).index('sourceDay',['worldId','source','day']),
  simScenes: defineTable({worldId:v.id('worlds'), conversationId:v.string(), status:v.string(), snapshot:v.any(), participants:v.array(v.string()), purpose:v.string(), legal:v.any(), scene:v.any(), posted:v.number(), forecast:v.boolean(), deadline:v.number(), samples:v.any(), error:v.string()}).index('conversation',['worldId','conversationId']),
  simPredictions: defineTable({worldId:v.id('worlds'), targetId:v.string(), status:v.string(), request:v.any(), probabilities:v.any(), score:v.any()}).index('target',['worldId','targetId']).index('status',['worldId','status']),
  simOutbox: defineTable({worldId:v.id('worlds'), eventId:v.string(), kind:v.string(), payload:v.any(), acknowledged:v.boolean()}).index('pending',['worldId','acknowledged']).index('eventId',['worldId','eventId']),
  simMirrorSnapshots: defineTable({worldId:v.id('worlds'), snapshotId:v.string(), payload:v.any()}).index('snapshot',['worldId','snapshotId']),
};

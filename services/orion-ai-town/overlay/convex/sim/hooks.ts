import {anyApi} from 'convex/server';
const refs:any=anyApi.sim.runtime;
export async function townMessage(ctx:any,args:any):Promise<{text:string;leave:boolean;index:number}|null>{
  const current=await ctx.runQuery(refs.snapshot,{worldId:args.worldId});if(!current?.enabled)return null;
  const prompt=await ctx.runQuery((anyApi as any).agent.conversation.queryPromptData,{worldId:args.worldId,playerId:args.playerId,otherPlayerId:args.otherPlayerId,conversationId:args.conversationId});
  if(!prompt.otherAgent||prompt.player.human||prompt.otherPlayer.human)return null;
  let row=await ctx.runQuery(refs.getScene,{worldId:args.worldId,conversationId:args.conversationId});
  if(!row)row=await ctx.runAction(refs.generateScene,{worldId:args.worldId,conversationId:args.conversationId,participants:[args.playerId,args.otherPlayerId]});
  if(!row||row.status!=='ready')return {text:'',leave:true,index:-1};
  const line=row.scene.lines[row.posted];if(!line||line.speaker!==current.state.agents[args.playerId]?.name)return {text:'',leave:true,index:-1};
  return {text:line.text,leave:row.posted===row.scene.lines.length-1,index:row.posted};
}

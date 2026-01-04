from .llm_rpc import CouncilRPC, AgentChainRPC # Ensure AgentChainRPC is imported
from .chat_front import run_chat_general, run_chat_agentic

async def handle_chat_request(
    bus,
    payload: dict,
    session_id: str,
) -> Dict[str, Any]:
    user_messages = payload.get("messages", [])
    temperature = payload.get("temperature", 0.7)
    mode = payload.get("mode", "brain")
    use_recall = bool(payload.get("use_recall", False))
    packs = payload.get("packs")

    if not isinstance(user_messages, list) or len(user_messages) == 0:
        return {"error": "Invalid payload: missing messages[]"}

    user_prompt = user_messages[-1].get("content", "") or ""

    if mode == "council":
        rpc = CouncilRPC(bus)
        reply = await rpc.call_llm(
            prompt=user_prompt,
            history=user_messages[-5:],
            temperature=temperature,
        )

        # ─────────────────────────────────────────────────────────────
        # 🛡️ DELEGATION PROTOCOL (Phase 2.4)
        # ─────────────────────────────────────────────────────────────
        meta = reply.get("meta", {})
        if meta.get("decision") == "delegate":
            logger.info("[%s] Hub: Council requested DELEGATION. Re-routing to Planner.", session_id)
            
            # Switch to 'agentic' mode to trigger the ReAct Planner
            convo = await run_chat_agentic(
                bus,
                session_id=session_id,
                user_id=None,
                messages=user_messages,
                chat_mode="agentic",
                temperature=temperature,
                use_recall=use_recall,
                packs=packs,
            )
            return {
                "session_id": session_id,
                "mode": "delegated_planner",
                "use_recall": use_recall,
                "text": convo.get("text") or "",
                "tokens": convo.get("tokens") or 0,
                "raw": convo.get("raw_agent_chain"),
                "recall_debug": convo.get("recall_debug") or {},
                "spark_meta": convo.get("spark_meta"),
                "meta": {"delegated_from": "council"}
            }

        # Standard Council response path
        text = reply.get("final_text") or reply.get("text") or ""
        tokens = len(text.split()) if text else 0

        return {
            "session_id": session_id,
            "mode": mode,
            "use_recall": use_recall,
            "text": text,
            "tokens": tokens,
            "raw": reply,
            "recall_debug": {},
            "spark_meta": None,
        }

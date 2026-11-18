"""Exact context gate for the deep thinker, before the engine can clamp output."""
from fastapi.responses import JSONResponse

PROFILE = "deep.thinker.v1"
CONTEXT = 262144
OUTPUT = 32768
MIN_RESERVE = 16384


def budget_error(body, prompt_tokens, context):
    if body.get("n", 1) != 1 or body.get("best_of", 1) != 1:
        return "deep thinker permits one completion per request"
    if body.get("truncate_prompt_tokens") is not None:
        return "deep thinker does not permit prompt truncation"
    a, b = body.get("max_completion_tokens"), body.get("max_tokens")
    if a is not None and b is not None and a != b:
        return "conflicting output budgets"
    output = a if a is not None else b if b is not None else OUTPUT
    if isinstance(output, bool) or not isinstance(output, int) or not 1 <= output <= OUTPUT:
        return f"output budget must be an integer from 1 to {OUTPUT}"
    if prompt_tokens + max(MIN_RESERVE, output) > context:
        return "prompt plus reserved reasoning/output exceeds live context; shorten explicitly"
    return None


async def check_request(request, cfg):
    """Called only on the pinned 1Cat deep worker's OpenAI chat endpoint."""
    from vllm.entrypoints.serve.tokenize.protocol import TokenizeChatRequest
    from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
    try:
        if request.url.path not in {'/v1/chat/completions', '/v1/messages'}:
            raise ValueError("deep thinker accepts chat/messages; other generation protocols are unsupported")
        body = await request.json()
        if not isinstance(body, dict):
            raise ValueError("request must be an object")
        error = budget_error(body, 0, cfg.model_config.max_model_len)
        if error:
            raise ValueError(error)
        if request.url.path == '/v1/messages':
            from vllm.entrypoints.anthropic.protocol import AnthropicMessagesRequest
            from vllm.entrypoints.anthropic.serving import AnthropicServingMessages
            chat_request = AnthropicServingMessages._convert_anthropic_to_openai_request(AnthropicMessagesRequest(**body))
        else:
            chat_request = ChatCompletionRequest(**body)
        # The SAME chat parameters as generation, including documents, tools,
        # reasoning_effort and thinking defaults; count before any context clamp.
        class ExactTokenizeRequest(TokenizeChatRequest):
            def build_chat_params(self, default_template, default_template_content_format):
                return chat_request.build_chat_params(default_template, default_template_content_format)
        fields = TokenizeChatRequest.model_fields
        token_request = ExactTokenizeRequest(**{k: v for k, v in chat_request.model_dump().items() if k in fields})
        result = await request.app.state.openai_serving_tokenization.create_tokenize(token_request, request)
        if not hasattr(result, "count"):
            return JSONResponse(result.model_dump(), status_code=result.error.code)
        error = budget_error(body, result.count, cfg.model_config.max_model_len)
        if error:
            raise ValueError(error)
    except ValueError as exc:
        return JSONResponse({"error": {"type": "invalid_context_budget", "message": str(exc)}}, status_code=400)
    return None

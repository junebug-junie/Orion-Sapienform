"""Read-only pool discovery middleware for the pinned vLLM OpenAI server.

Loaded by --middleware; reads initialized engine state, never desired Orion settings.
Does not enable VLLM_SERVER_DEV_MODE or expose environment/system diagnostics.
"""
from fastapi.responses import JSONResponse


async def pool_server_info(request, call_next):
    cfg = getattr(request.app.state, 'vllm_config', None)
    names = getattr(getattr(cfg, 'model_config', None), 'served_model_name', []) or []
    if isinstance(names, str):
        names = [names]
    if 'deep.thinker.v1' in names and request.method == 'POST' and request.url.path not in {'/tokenize', '/detokenize', '/v1/messages/count_tokens'}:
        from .deep_budget import check_request
        rejected = await check_request(request, cfg)
        if rejected is not None:
            return rejected
    if request.url.path != '/orion/server-info':
        return await call_next(request)
    if request.method != 'GET':
        return JSONResponse({'error': 'method_not_allowed'}, status_code=405)
    if cfg is None:
        return JSONResponse({'error': 'engine_not_initialized'}, status_code=503)
    return JSONResponse({'vllm_config': {
        'model_config': {
            'model': cfg.model_config.model,
            'max_model_len': cfg.model_config.max_model_len,
        },
        'scheduler_config': {'max_num_seqs': cfg.scheduler_config.max_num_seqs},
        'parallel_config': {
            'tensor_parallel_size': cfg.parallel_config.tensor_parallel_size,
            'pipeline_parallel_size': cfg.parallel_config.pipeline_parallel_size,
            'data_parallel_size': cfg.parallel_config.data_parallel_size,
        },
    }})

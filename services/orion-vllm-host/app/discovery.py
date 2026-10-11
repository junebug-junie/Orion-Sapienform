"""Read-only pool discovery middleware for the pinned vLLM OpenAI server.

Loaded by --middleware; reads initialized engine state, never desired Orion settings.
Does not enable VLLM_SERVER_DEV_MODE or expose environment/system diagnostics.
"""
from fastapi.responses import JSONResponse


async def pool_server_info(request, call_next):
    if request.url.path != '/orion/server-info':
        return await call_next(request)
    if request.method != 'GET':
        return JSONResponse({'error': 'method_not_allowed'}, status_code=405)
    cfg = getattr(request.app.state, 'vllm_config', None)
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

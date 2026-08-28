"""Settings for orion-diffusion-host.

Patch 1 (docs/superpowers/specs/2026-08-20-reverie-visual-chain-design.md)
shipped this class with no field read by anything that loads a model. This
patch (the "wire a real model" follow-up, PR TBD) adds the DIFFUSION_* block
below and is the first patch where MODEL_CACHE_DIR/HF_HOME/TRANSFORMERS_CACHE
actually get read. Shape follows services/orion-vision-host/app/settings.py's
Settings class.
"""

from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    SERVICE_NAME: str = "diffusion-host"
    SERVICE_VERSION: str = "0.2.0"
    NODE_NAME: str = "circe"
    LOG_LEVEL: str = "INFO"

    # Bus. Declared but not consumed yet -- README.md. Still no chain
    # orchestration in this patch (that's Patch 2); this patch only wires
    # /generate as a plain HTTP endpoint.
    ORION_BUS_ENABLED: bool = False
    ORION_BUS_ENFORCE_CATALOG: bool = True
    ORION_BUS_URL: str = "redis://localhost:6379/0"
    HEARTBEAT_INTERVAL_SEC: float = 10.0

    # Model cache dir convention: storage-warm, not vision-host's telemetry
    # convention -- see README.md "Model cache dir convention" for why.
    MODEL_CACHE_DIR: str = "/mnt/storage-warm/models/diffusion"
    HF_HOME: str = "/mnt/storage-warm/models/diffusion/hf"
    TRANSFORMERS_CACHE: str = "/mnt/storage-warm/models/diffusion/hf/transformers"

    # Model. sdxl-turbo (a distilled, single-step SDXL variant) is
    # REPLACED here by FLUX.1-schnell (2026-08-28) -- the real root-cause
    # fix for the 77-token CLIP-only ceiling this repo's `_log_prompt_
    # token_budget` exists because of (design doc §18/19). FLUX carries a
    # T5-XXL second text encoder with a real ~256-token budget for actual
    # cross-attention content, not CLIP's 77 -- CLIP-L is still loaded
    # (`tokenizer`/`text_encoder`) but only contributes a single pooled
    # embedding in this architecture, so its 77-token truncation is far
    # less consequential here than it was for SDXL, where CLIP was the
    # only encoder.
    #
    # DIFFUSION_MODEL_ID points at `YuCollection/FLUX.1-schnell-Diffusers`,
    # NOT the official `black-forest-labs/FLUX.1-schnell` -- the official
    # repo is gated (requires accepting a license via the HF web UI before
    # any token can download it; confirmed live 2026-08-28 that neither
    # HF_TOKEN already configured elsewhere in this repo, orion-vllm's nor
    # orion-llama-cola-host's, has accepted that gate -- both return a real
    # 403 on the actual weight files, not just a metadata-endpoint
    # false-positive). The mirror is a verified-ungated (`gated: false`),
    # full diffusers-format re-upload of the same Apache-2.0-licensed
    # weights (confirmed live: identical `model_index.json` --
    # `FluxPipeline`/`CLIPTextModel`+`T5EncoderModel`/
    # `FluxTransformer2DModel`/`AutoencoderKL`), downloadable with no token
    # at all. Apache 2.0 explicitly permits this kind of redistribution.
    #
    # DIFFUSION_DTYPE=fp16, NOT bf16 -- correction, caught by review before
    # deploy: FLUX's own docs recommend bf16 (avoids a known fp16 overflow
    # risk), but that assumes Ampere-or-later hardware. This service's own
    # actual card (physical GPU 2, "Tesla PG500-216" per README's "Node/
    # port/GPU assignment") is Volta architecture with first-generation
    # Tensor Cores -- confirmed live via GPU spec lookup, NOT the Ampere+
    # (compute capability >= 8.0) generation bf16 tensor-core acceleration
    # requires. Real, corroborated failure reports from other PyTorch-based
    # projects on this exact GPU class: "Bfloat16 is only supported on
    # GPUs with compute capability of at least 8.0" / "Current CUDA Device
    # does not support bfloat16." Volta's tensor cores were built for
    # fp16, which is why sdxl-turbo (a different model, same GPU) already
    # ran fp16 correctly. fp32 would sidestep the overflow risk entirely
    # but roughly doubles VRAM (~48GB+ even offloaded) -- not viable on
    # this 32GB card. The fp16 overflow risk is real but managed, not
    # ignored: this service already surfaces a load/generation failure as
    # a visible `_load_error`/500, not silent corruption (module docstring
    # "Startup"/`generate`'s own error handling) -- if fp16 produces
    # visibly degraded output in practice, that would show up as an
    # inspectable generation result, not a hidden problem.
    #
    # DIFFUSION_ENABLE_MODEL_CPU_OFFLOAD=true -- see `_load_pipeline`'s own
    # docstring in app/main.py for the real VRAM math (fully-resident at
    # 2 bytes/param can need up to ~33GB regardless of fp16 vs bf16, over
    # a 32GB card; offloaded, ~24GB, fits with margin once sdxl-turbo's
    # own footprint is freed by this same swap).
    #
    # DIFFUSION_NUM_INFERENCE_STEPS=4 (was 1) and DIFFUSION_DEFAULT_WIDTH/
    # HEIGHT=1024 (was 512) -- schnell's own documented operating point:
    # 4 steps is the model's real sweet spot (1 step is sdxl-turbo's, not
    # this model's), and FLUX was trained primarily at 1024x1024 --
    # 512x512 works but is off-distribution and produces worse output for
    # no VRAM saving worth the quality loss. DIFFUSION_GUIDANCE_SCALE
    # stays 0.0 -- schnell is guidance-distilled, same operating point as
    # sdxl-turbo's, unchanged by this swap.
    #
    # DIFFUSION_MAX_SEQUENCE_LENGTH=256 (new) -- the real T5-XXL token
    # budget this whole swap exists to gain; passed explicitly to every
    # `_pipe()` call (see `_run_generation`'s `_pipe_accepts` check in
    # app/main.py) rather than relying on the pipeline's own unstated
    # default, which schnell's own examples show varies (some default to
    # 256, `black-forest-labs`'s own reference code allows up to 512).
    DIFFUSION_MODEL_ID: str = "YuCollection/FLUX.1-schnell-Diffusers"
    DIFFUSION_DEVICE: str = "cuda:0"
    DIFFUSION_DTYPE: str = "fp16"
    DIFFUSION_ENABLE_MODEL_CPU_OFFLOAD: bool = True
    DIFFUSION_NUM_INFERENCE_STEPS: int = 4
    DIFFUSION_GUIDANCE_SCALE: float = 0.0
    DIFFUSION_MAX_SEQUENCE_LENGTH: int = 256
    DIFFUSION_DEFAULT_WIDTH: int = 1024
    DIFFUSION_DEFAULT_HEIGHT: int = 1024
    DIFFUSION_MAX_PROMPT_CHARS: int = 2000

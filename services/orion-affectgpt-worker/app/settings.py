from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    SERVICE_NAME: str = "affectgpt-worker"
    SERVICE_VERSION: str = "0.1.0"
    NODE_NAME: str = "circe"
    LOG_LEVEL: str = "INFO"

    # Bus
    ORION_BUS_ENABLED: bool = True
    ORION_BUS_ENFORCE_CATALOG: bool = False
    ORION_BUS_URL: str = "redis://localhost:6379/0"
    HEARTBEAT_INTERVAL_SEC: float = 10.0

    # Channels
    CHANNEL_AFFECTGPT_INTAKE: str = "orion:exec:request:AffectGptWorkerService"
    CHANNEL_AFFECTGPT_REPLY_PREFIX: str = "orion:affectgpt:reply"

    # AffectGPT source (vendored at build time, see Dockerfile -- not checked
    # into this repo; it's a ~pinned-commit external research codebase, not
    # ours to own/maintain). AFFECTGPT_SRC_ROOT is the directory containing
    # config.py and my_affectgpt/ (the actual code root is one level below
    # the repo's own top-level dir -- AffectGPT/AffectGPT upstream).
    AFFECTGPT_SRC_ROOT: str = "/opt/affectgpt-src/AffectGPT"

    # Model weights + checkpoint. Mounted read-only volumes (see
    # docker-compose.yml) -- never baked into the image (~33GB across
    # CLIP/HuBERT/Qwen2.5-7B/checkpoint). config.py's own PATH_TO_LLM etc. are
    # relative ('models/Qwen2.5-7B-Instruct') to AFFECTGPT_SRC_ROOT, so this
    # must be bind-mounted at <AFFECTGPT_SRC_ROOT>/models -- see compose.
    AFFECTGPT_MODEL_ROOT: str = "/opt/affectgpt-src/AffectGPT/models"
    # Checkpoint directory (the LoRA + Q-Former adapter weights actually
    # trained by AffectGPT, distinct from the frozen base models above).
    # Confirmed live 2026-08-22: only a face-crop-trained checkpoint is
    # reachable (HuggingFace `MERChallenge/AffectGPT`); no frame-mode
    # checkpoint is downloadable from there (only via Baidu Netdisk, an
    # untrusted/inaccessible source for this deployment -- not used).
    AFFECTGPT_CKPT_ROOT: str = (
        "/opt/affectgpt-src/AffectGPT/models/AffectGPT-ckpt/"
        "emercoarse_highlevelfilter4_outputhybird_bestsetup_bestfusion_lz_20250110100"
    )
    AFFECTGPT_CFG_PATH: str = (
        "train_configs/emercoarse_highlevelfilter4_outputhybird_bestsetup_bestfusion_lz.yaml"
    )
    # Fixed epoch (not "last available") so a future checkpoint drop into the
    # same directory can't silently change which weights actually serve.
    AFFECTGPT_CKPT_EPOCH: int = 60

    # The ONLY mode with a real released checkpoint (multiface + audio + face
    # + text). Not configurable at request time -- confirmed live 2026-08-22
    # that a raw-frame checkpoint does not exist for this deployment; running
    # any other face_or_frame value here would silently feed the model input
    # it was never trained to consume.
    AFFECTGPT_FACE_OR_FRAME: str = "multiface_audio_face_text"

    AFFECTGPT_DEFAULT_USER_MESSAGE: str = (
        "Please infer the person's emotional state and provide your reasoning process."
    )

    # GPU. CUDA_VISIBLE_DEVICES is set at the container level (docker-compose
    # `gpus: '"device=N"'`); the visible device is always container-local 0.
    AFFECTGPT_DEVICE: str = "cuda:0"

    # Determinism ceiling (confirmed live 2026-08-22, see README "Determinism"
    # section): do_sample=False + torch.use_deterministic_algorithms reduces
    # but does NOT eliminate run-to-run variance -- the residual source is
    # PyTorch's fused SDPA attention kernel, which is not covered by the
    # determinism guard. Forcing attn_implementation="eager" to close that gap
    # was tried and REJECTED: it corrupted output into incoherent repeating
    # garbage on this checkpoint. sdpa (the default) stays. Not exposed as a
    # request-time option -- this is a model-safety constraint, not a tuning
    # knob.
    AFFECTGPT_MAX_NEW_TOKENS: int = 1200
    AFFECTGPT_MAX_LENGTH: int = 2000

    # Face-crop extraction (Haar cascade substitute for real OpenFace -- see
    # README "Face crops" section for why: the only reachable OpenFace Docker
    # image segfaults on this hardware and upstream has no Dockerfile to
    # build from source). Margin is fraction of detected box added per side.
    AFFECTGPT_FACE_MARGIN: float = 0.25
    AFFECTGPT_FACE_MIN_SIZE_PX: int = 60

    # Single-inflight lock: one 7B model on one GPU cannot usefully serve
    # concurrent requests, and AffectGPT's own Chat object is not documented
    # as thread-safe. A queue is a premature complexity for a single-user
    # (Juniper) signal -- see README non-goals.
    AFFECTGPT_REQUEST_TIMEOUT_S: float = 120.0

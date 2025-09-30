import os

class Settings:
    """
    Lightweight settings loader that:
    - Reads from environment variables
    - Provides type casting (int, float, bool, tuple)
    - Defaults to str if no caster is defined
    """

    # mapping of known env keys -> caster
    _casters = {
        "WIDTH": int,
        "HEIGHT": int,
        "FPS": int,
        "DETECT_EVERY_N_FRAMES": int,
        "MOTION_MIN_AREA": int,
        "FACE_SCALE_FACTOR": float,
        "FACE_MIN_NEIGHBORS": int,
        "FACE_MIN_SIZE": lambda v: tuple(map(int, v.split(","))),
        "ANNOTATE": lambda v: str(v).lower() in ("1", "true", "yes"),

        "ENABLE_UI": lambda v: str(v).lower() in ("1", "true", "yes"),
        "PRESENCE_TIMEOUT": int,
        "PRESENCE_LABEL": str,

        "ENABLE_YOLO": lambda v: str(v).lower() in ("1", "true", "yes"),
        "YOLO_MODEL": str,
        "YOLO_CLASSES": str,
        "YOLO_CONF": float,
        "YOLO_DEVICE": str,

        "LLM_MODEL": str,
        "BRAIN_URL": str,
        "ORION_BUS_ENABLED": lambda v: str(v).lower() in ("1", "true", "yes"),
        "ORION_BUS_URL": str,
    }

    def __getattr__(self, name: str):
        env_key = name.upper()
        val = os.getenv(env_key)
        if val is None:
            raise AttributeError(f"Missing setting: {name}")

        caster = self._casters.get(env_key, str)
        try:
            return caster(val)
        except Exception as e:
            raise ValueError(f"Failed to cast {env_key}={val!r}: {e}")

settings = Settings()

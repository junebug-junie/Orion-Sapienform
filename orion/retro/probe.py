"""Small libretro ABI-v1 probe for operator-supplied, trusted cores and ROMs.

This is an offline benchmark, NOT a sandbox or a deployable game service.
Run native cores in an isolated unprivileged environment without credentials.
"""
from __future__ import annotations

import ctypes as C
import hashlib
import math
from pathlib import Path


class GameInfo(C.Structure):
    _fields_ = [("path", C.c_char_p), ("data", C.c_void_p),
                ("size", C.c_size_t), ("meta", C.c_char_p)]


class SystemInfo(C.Structure):
    _fields_ = [("name", C.c_char_p), ("version", C.c_char_p),
                ("extensions", C.c_char_p), ("need_fullpath", C.c_bool),
                ("block_extract", C.c_bool)]


class Geometry(C.Structure):
    _fields_ = [("base_width", C.c_uint), ("base_height", C.c_uint),
                ("max_width", C.c_uint), ("max_height", C.c_uint),
                ("aspect_ratio", C.c_float)]


class Timing(C.Structure):
    _fields_ = [("fps", C.c_double), ("sample_rate", C.c_double)]


class AVInfo(C.Structure):
    _fields_ = [("geometry", Geometry), ("timing", Timing)]


class Variable(C.Structure):
    _fields_ = [("key", C.c_char_p), ("value", C.c_char_p)]


Environment = C.CFUNCTYPE(C.c_bool, C.c_uint, C.c_void_p)
Video = C.CFUNCTYPE(None, C.c_void_p, C.c_uint, C.c_uint, C.c_size_t)
Audio = C.CFUNCTYPE(None, C.c_int16, C.c_int16)
AudioBatch = C.CFUNCTYPE(C.c_size_t, C.POINTER(C.c_int16), C.c_size_t)
Poll = C.CFUNCTYPE(None)
Input = C.CFUNCTYPE(C.c_int16, C.c_uint, C.c_uint, C.c_uint, C.c_uint)


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


class CoreProbe:
    """Own exactly one native core in a dedicated process; caller owns pacing."""

    def __init__(self, core: Path, rom: Path):
        core, rom = core.resolve(strict=True), rom.resolve(strict=True)
        if rom.suffix.lower() not in {".nes", ".sfc", ".smc"}:
            raise ValueError("only NES/SNES content is supported")
        if not 0 < rom.stat().st_size <= 32 * 1024 * 1024:
            raise ValueError("ROM must be nonempty and at most 32 MiB")
        self.identity = {"core_sha256": digest(core), "rom_sha256": digest(rom)}
        self.lib = C.CDLL(str(core))
        self.initialized = self.loaded = False
        self.buttons = (0, 0)
        self.pixel_format = 0  # ABI default: 0RGB1555
        self.video_frames = self.audio_samples = 0
        self.frame_hash = None
        self.error = None
        self.options: dict[bytes, bytes] = {}
        self.unsupported: set[int] = set()
        self._callbacks = [Environment(self._environment), Video(self._video),
                           Audio(self._audio), AudioBatch(self._audio_batch),
                           Poll(lambda: None), Input(self._input)]
        signatures = {
            "retro_api_version": (C.c_uint, []),
            "retro_get_system_info": (None, [C.POINTER(SystemInfo)]),
            "retro_get_system_av_info": (None, [C.POINTER(AVInfo)]),
            "retro_init": (None, []), "retro_deinit": (None, []),
            "retro_load_game": (C.c_bool, [C.POINTER(GameInfo)]),
            "retro_unload_game": (None, []), "retro_run": (None, []),
            "retro_set_controller_port_device": (None, [C.c_uint, C.c_uint]),
            "retro_serialize_size": (C.c_size_t, []),
            "retro_serialize": (C.c_bool, [C.c_void_p, C.c_size_t]),
            "retro_unserialize": (C.c_bool, [C.c_void_p, C.c_size_t]),
        }
        for name, (result, args) in signatures.items():
            function = getattr(self.lib, name)
            function.restype, function.argtypes = result, args
        if self.lib.retro_api_version() != 1:
            raise ValueError("unsupported libretro ABI")
        for name, callback in zip(("environment", "video_refresh", "audio_sample",
                                   "audio_sample_batch", "input_poll", "input_state"), self._callbacks):
            function = getattr(self.lib, f"retro_set_{name}")
            function.argtypes, function.restype = [type(callback)], None
            function(callback)
        try:
            self.lib.retro_init()
            self.initialized = True
            info = SystemInfo()
            self.lib.retro_get_system_info(C.byref(info))
            extensions = (info.extensions or b"").decode().split("|")
            if rom.suffix[1:].lower() not in extensions:
                raise ValueError("core does not advertise the ROM extension")
            self.identity.update(core_name=(info.name or b"").decode(),
                                 core_version=(info.version or b"").decode())
            self._rom_data = None if info.need_fullpath else C.create_string_buffer(rom.read_bytes())
            self._path = str(rom).encode()
            game = GameInfo(self._path, C.cast(self._rom_data, C.c_void_p), rom.stat().st_size, None)
            if not self.lib.retro_load_game(C.byref(game)):
                raise ValueError("core rejected ROM")
            self.loaded = True
            self.av = AVInfo()
            self.lib.retro_get_system_av_info(C.byref(self.av))
            if not (math.isfinite(self.av.timing.fps) and 1 <= self.av.timing.fps <= 240
                    and math.isfinite(self.av.timing.sample_rate)
                    and 1000 <= self.av.timing.sample_rate <= 192000):
                raise ValueError("invalid core timing")
            for port in (0, 1):
                self.lib.retro_set_controller_port_device(port, 1)
        except BaseException:
            self.close()
            raise

    def _environment(self, command, data):
        # Only callbacks implemented here return true. No VFS, host directories,
        # hardware context, network, logging, or speculative capabilities.
        if command == 3 and data:  # GET_CAN_DUPE
            C.cast(data, C.POINTER(C.c_bool))[0] = True
            return True
        if command == 10 and data:  # SET_PIXEL_FORMAT
            value = C.cast(data, C.POINTER(C.c_int))[0]
            if value in (0, 1, 2):
                self.pixel_format = value
                return True
            return False
        if command == 16 and data:  # SET_VARIABLES, keep first/default value
            values = C.cast(data, C.POINTER(Variable))
            for index in range(1024):
                item = values[index]
                if not item.key:
                    return True
                if item.value and b"; " in item.value:
                    self.options[item.key] = item.value.split(b"; ", 1)[1].split(b"|", 1)[0]
            self.error = "too many core options"
            return False
        if command == 15 and data:  # GET_VARIABLE
            item = C.cast(data, C.POINTER(Variable)).contents
            item.value = self.options.get(item.key)
            return item.value is not None
        if command == 17 and data:  # GET_VARIABLE_UPDATE
            C.cast(data, C.POINTER(C.c_bool))[0] = False
            return True
        if command == 7:  # SHUTDOWN
            self.error = "core requested shutdown"
            return True
        self.unsupported.add(command)
        return False

    def _video(self, data, width, height, pitch):
        if not data:  # duplicate frame; retain previous digest
            return
        row_size = width * (4 if self.pixel_format == 1 else 2)
        if not (0 < width <= 4096 and 0 < height <= 4096 and row_size <= pitch <= 32768):
            self.error = "invalid video geometry"
            return
        # Exclude pitch padding, which may be uninitialized/non-deterministic.
        checksum = hashlib.sha256()
        checksum.update(f"{width}:{height}:{self.pixel_format}:".encode())
        for row in range(height):
            checksum.update(C.string_at(data + row * pitch, row_size))
        self.frame_hash = checksum.hexdigest()
        self.video_frames += 1

    def _audio(self, left, right):
        self.audio_samples += 1

    def _audio_batch(self, data, frames):
        self.audio_samples += frames
        return frames

    def _input(self, port, device, index, button):
        if port < 2 and device == 1 and index == 0 and button < 16:
            return int(bool(self.buttons[port] & (1 << button)))
        return 0

    def step(self, p1=0, p2=0):
        if not self.loaded:
            raise RuntimeError("core is closed")
        if any(type(mask) is not int or not 0 <= mask <= 0xFFFF for mask in (p1, p2)):
            raise ValueError("invalid controller mask")
        self.buttons = (p1, p2)
        self.lib.retro_run()
        if self.error:
            raise RuntimeError(self.error)
        return self.frame_hash

    def snapshot(self):
        if not self.loaded:
            raise RuntimeError("core is closed")
        size = self.lib.retro_serialize_size()
        if not 0 < size <= 64 * 1024 * 1024:
            raise ValueError("unsupported save-state size")
        state = C.create_string_buffer(size)
        if not self.lib.retro_serialize(state, size):
            raise RuntimeError("serialization failed")
        return state.raw

    def restore(self, state):
        if not self.loaded:
            raise RuntimeError("core is closed")
        if len(state) != self.lib.retro_serialize_size():
            raise ValueError("incompatible save-state size")
        buffer = C.create_string_buffer(state)
        if not self.lib.retro_unserialize(buffer, len(state)):
            raise RuntimeError("restore failed")
        self.buttons = (0, 0)
        self.frame_hash = None

    def close(self):
        if self.loaded:
            self.lib.retro_unload_game()
            self.loaded = False
        if self.initialized:
            self.lib.retro_deinit()
            self.initialized = False

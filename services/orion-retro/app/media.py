"""Transient, bounded native callback copies. No media is persisted.

Wire v1: zlib(header length u32 BE + UTF-8 JSON header + packed video + s16le PCM).
Video retains libretro pixel format (0=1555, 1=XRGB8888, 2=565), without pitch.
Audio is stereo, timestamped by cumulative sample count. Dropped packets are gaps.
"""
import ctypes as C
import json
import queue
import struct
import zlib
from orion.retro.probe import CoreProbe


class MediaSlot:
    """One bounded packet, with no pipe or blocking receive after worker death.

    A process killed while owning the lock abandons this session's slot. Both
    sides try the lock without waiting; the API can still detect/stop the child.
    A new session gets a new slot/lock. Never repair or reuse an abandoned lock.
    """
    CAPACITY = 512 * 512 * 4 + 65536 + 4096

    def __init__(self, context):
        self.buffer = context.RawArray('B', self.CAPACITY)
        self.size = context.RawValue('I', 0)
        self.lock = context.Lock()

    def put_nowait(self, packet):
        if len(packet) > self.CAPACITY or not self.lock.acquire(False):
            raise queue.Full
        try:
            if self.size.value:
                raise queue.Full
            memoryview(self.buffer).cast('B')[:len(packet)] = packet
            self.size.value = len(packet)
        finally:
            self.lock.release()

    def get_nowait(self):
        if not self.lock.acquire(False):
            raise queue.Empty
        try:
            size = self.size.value
            if not size:
                raise queue.Empty
            packet = bytes(memoryview(self.buffer).cast('B')[:size])
            self.size.value = 0
            return packet
        finally:
            self.lock.release()


class MediaProbe(CoreProbe):
    def __init__(self, *args, **kwargs):
        self.video = b''; self.pcm = bytearray(); self.geometry = (0, 0)
        super().__init__(*args, **kwargs)

    def _video(self, data, width, height, pitch):
        super()._video(data, width, height, pitch)
        if not data or self.error: return
        # Larger pictures are decimated by a whole factor (N64 Angrylion: 640x480 is the
        # native 320x240 doubled), so the wire keeps its 512x512 bound.
        # Angrylion sends 640x480 and 640x240 (low-res modes are doubled only horizontally).
        # Halve the width, then keep the console's 4:3 picture: both become 320x240.
        # Pictures within the bound (NES/SNES, hi-res 512x448) pass unchanged.
        sx = -(-width // 512)
        sy = 1 if sx == 1 else max(1, min(2, round(height * sx * 4 / (width * 3))))
        if sx > 2 or -(-height // sy) > 512:
            self.error = 'media_geometry_limit'; return
        depth = 4 if self.pixel_format == 1 else 2
        size = width * depth
        if sx == 1:
            self.video = b''.join(C.string_at(data + y*pitch, size) for y in range(0, height, sy))
        else:
            code = 'I' if depth == 4 else 'H'
            self.video = b''.join(memoryview(C.string_at(data + y*pitch, size)).cast(code)[::sx].tobytes()
                                  for y in range(0, height, sy))
        self.geometry = (-(-width // sx), -(-height // sy))

    def _audio(self, left, right):
        super()._audio(left, right)
        if len(self.pcm) < 65536: self.pcm.extend(struct.pack('<hh', left, right))

    def _audio_batch(self, data, frames):
        result = super()._audio_batch(data, frames)
        if frames <= 16384 and len(self.pcm)+frames*4 <= 65536:
            self.pcm.extend(C.string_at(data, frames*4))
        else: self.error = 'media_audio_limit'
        return result

    def publish(self, pipe, frame):
        # A cold core has no picture yet (live 2026-11-21: Angrylion's first frame arrives at
        # ~frame 64). Publishing width=0 packets makes the browser throw on a 0x0 image and
        # close the socket, which pauses the session before the renderer ever warms up.
        # Skipping them without draining would trip media_audio_limit in ~30 frames, killing
        # the core instead: send nothing, but drop the buffered audio.
        if not self.video:
            self.pcm.clear(); return
        if frame % 2: return
        header = dict(version=1, frame=frame, width=self.geometry[0], height=self.geometry[1],
                      format=self.pixel_format, video_bytes=len(self.video),
                      sample_rate=self.av.timing.sample_rate, sample_end=self.audio_samples,
                      audio_frames=len(self.pcm)//4)
        encoded = json.dumps(header, separators=(',', ':')).encode()
        packet = struct.pack('!I', len(encoded))+encoded+self.video+self.pcm
        self.pcm.clear()
        # Full media queues drop packets; they never block controller timing.
        try: pipe.put_nowait(packet)
        except queue.Full: pass


def compress(packet):
    return zlib.compress(packet, 1)

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
    def __init__(self, *args):
        self.video = b''; self.pcm = bytearray(); self.geometry = (0, 0)
        super().__init__(*args)

    def _video(self, data, width, height, pitch):
        super()._video(data, width, height, pitch)
        if not data or self.error: return
        if width > 512 or height > 512:
            self.error = 'media_geometry_limit'; return
        size = width * (4 if self.pixel_format == 1 else 2)
        self.video = b''.join(C.string_at(data + y*pitch, size) for y in range(height))
        self.geometry = (width, height)

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

import ctypes as C
import json
import queue
import struct
import zlib
from types import SimpleNamespace
from app.media import MediaProbe, compress


def test_callback_copy_survives_native_buffer_reuse_and_queue_pressure():
    p=MediaProbe.__new__(MediaProbe)
    p.pixel_format=2;p.error=None;p.video_frames=0;p.audio_samples=0;p.pcm=bytearray()
    p.av=SimpleNamespace(timing=SimpleNamespace(sample_rate=48000))
    pixels=(C.c_uint16*4)(0xf800,0x07e0,0x001f,0xffff)
    p._video(C.addressof(pixels),2,2,4)
    audio=(C.c_int16*4)(123,-123,456,-456)
    p._audio_batch(audio,2)
    pixels[0]=0;audio[0]=0
    q=queue.Queue(1);p.publish(q,2);p.publish(q,4)
    raw=zlib.decompress(compress(q.get_nowait()));n=struct.unpack('!I',raw[:4])[0]
    header=json.loads(raw[4:4+n]);data=raw[4+n:]
    assert header['sample_end']==2 and header['audio_frames']==2
    assert struct.unpack('<H',data[:2])[0]==0xf800
    assert struct.unpack('<h',data[8:10])[0]==123
    assert p.pcm==b'' and q.empty()

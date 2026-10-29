import ctypes as C
import json
import queue
import struct
import zlib
from types import SimpleNamespace
from app.media import MediaProbe, MediaSlot, compress
import multiprocessing as mp
import pytest


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


def _die_while_writing(slot):
    import os
    import signal
    slot.lock.acquire()
    # Simulate a native writer dying with an incomplete packet.
    slot.size.value = 128
    os.kill(os.getpid(), signal.SIGKILL)


def _read_after_writer_death(slot, result):
    try: slot.get_nowait()
    except queue.Empty: result.set()


def test_media_slot_is_bounded_and_copies_packets():
    slot=MediaSlot(mp.get_context('spawn'))
    data=bytearray(b'first')
    slot.put_nowait(data);data[0]=0
    with pytest.raises(queue.Full):slot.put_nowait(b'second')
    assert slot.get_nowait()==b'first'
    with pytest.raises(queue.Empty):slot.get_nowait()
    with pytest.raises(queue.Full):slot.put_nowait(b'x'*(slot.CAPACITY+1))
    slot.put_nowait(b'next')
    assert slot.get_nowait()==b'next'


def test_killed_media_writer_cannot_block_reader():
    ctx=mp.get_context('spawn');slot=MediaSlot(ctx);result=ctx.Event()
    writer=ctx.Process(target=_die_while_writing,args=(slot,))
    reader=ctx.Process(target=_read_after_writer_death,args=(slot,result))
    writer.start()
    try:
        writer.join(3)
        assert writer.exitcode==-9
        reader.start()
        reader.join(3)
        assert not reader.is_alive(),'media receive blocked after writer death'
        assert result.is_set(),'incomplete packet was treated as valid media'
    finally:
        for process in (writer,reader):
            if process.pid:
                if process.is_alive():process.kill()
                process.join(1)

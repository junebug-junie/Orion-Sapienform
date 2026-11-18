"""Prove all twelve SNES buttons reach both hardware ports on real Snes9x.

Only the exact authored diagnostic is accepted. Its WRAM is a test instrument,
not a general game observer. This checks input, callbacks and save/restore.
"""
import argparse
import ctypes as C
import json
from pathlib import Path
import struct

from .probe import CoreProbe
from .snes_pad_lab import build_rom


class SnesPadProbe(CoreProbe):
    def pads(self):
        self.lib.retro_get_memory_size.argtypes = [C.c_uint]
        self.lib.retro_get_memory_size.restype = C.c_size_t
        self.lib.retro_get_memory_data.argtypes = [C.c_uint]
        self.lib.retro_get_memory_data.restype = C.c_void_p
        assert self.lib.retro_get_memory_size(2) >= 5
        pointer = self.lib.retro_get_memory_data(2)
        assert pointer
        return struct.unpack('<HH', C.string_at(pointer, 4))


def evaluate(core):
    for _ in range(10): core.step()
    checkpoint = core.snapshot()
    # libretro bit -> SNES JOYx register bit (B,Y,Select,Start,Up,Down,Left,Right,A,X,L,R).
    register_bits = (15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4)
    checks = {}
    for port in (0, 1):
        for bit, register_bit in enumerate(register_bits):
            core.restore(checkpoint)
            masks = [0, 0]; masks[port] = 1 << bit
            for _ in range(4): core.step(*masks)
            expected = [0, 0]; expected[port] = 1 << register_bit
            checks[f'port_{port}_button_{bit}'] = list(core.pads()) == expected
    core.restore(checkpoint)
    for _ in range(4): core.step(512, 1024)
    before = (core.pads(), core.frame_hash)
    core.restore(checkpoint)
    for _ in range(4): core.step(512, 1024)
    checks['restore_replays_input_and_video'] = before == (core.pads(), core.frame_hash)
    checks['native_callbacks'] = core.video_frames > 90 and core.audio_samples > 0
    return dict(status='PASS' if all(checks.values()) else 'FAIL', identity=core.identity,
                checks=checks, scope='authored SNES ROM; audio is intentionally silent')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--core', required=True, type=Path)
    parser.add_argument('--rom', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    if args.rom.read_bytes() != build_rom(): parser.error('requires exact authored SNES Pad Lab ROM')
    core = SnesPadProbe(args.core, args.rom)
    try: result = evaluate(core)
    finally: core.close()
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(result['status'])
    raise SystemExit(0 if result['status'] == 'PASS' else 1)

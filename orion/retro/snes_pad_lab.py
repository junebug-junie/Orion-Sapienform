"""Original SNES controller diagnostic, MIT (PAD_LAB_LICENSE); no game assets.

32 KiB LoROM, 65C816 in emulation mode. Auto-joypad registers are copied to
WRAM $0000..$0003 every vblank, $0004 counts loops. Backdrop reflects both pads.
Audio intentionally remains silent. Used only by the real-core CI diagnostic.
"""
import argparse
from pathlib import Path
import struct

from .pad_lab import Program


def build_rom():
    p = Program()  # the fixed assembler's opcodes/labels also cover this 65C816 subset
    p.emit(0x78, 0xD8, 0xA2, 0xFF, 0x9A)  # SEI, CLD, LDX #$ff, TXS
    for address, value in ((0x2100, 0x80), (0x4200, 0), (0x212C, 0), (0x212D, 0),
                           (0x2121, 0), (0x2122, 0), (0x2122, 0), (0x4200, 1), (0x2100, 15)):
        p.emit(0xA9, value, 0x8D, address & 255, address >> 8)
    p.label('active')
    p.emit(0xAD, 0x12, 0x42)  # LDA HVBJOY
    p.branch(0x30, 'active')  # BMI: wait for active picture
    p.label('vblank')
    p.emit(0xAD, 0x12, 0x42)
    p.branch(0x10, 'vblank')  # BPL: wait for vblank
    p.label('joy')
    p.emit(0xAD, 0x12, 0x42, 0x29, 1)
    p.branch(0xD0, 'joy')  # auto-controller read must finish
    for offset in range(4):
        p.emit(0xAD, 0x18+offset, 0x42, 0x85, offset)  # JOY1L/H, JOY2L/H -> WRAM
    p.emit(0xE6, 4, 0xA9, 0, 0x8D, 0x21, 0x21)  # count loops; CGADD=0
    p.emit(0xA5, 0, 0x45, 2, 0x8D, 0x22, 0x21)  # color low = P1 low XOR P2 low
    p.emit(0xA5, 1, 0x45, 3, 0x29, 0x7F, 0x8D, 0x22, 0x21)
    p.jump('active')
    rom = bytearray(32768)
    code = p.finish(); rom[:len(code)] = code
    rom[0x7FC0:0x7FD5] = b'ORION SNES PAD LAB    '
    rom[0x7FD5:0x7FDC] = bytes((0x20, 0, 5, 0, 1, 0x33, 0))
    rom[0x7FFC:0x7FFE] = struct.pack('<H', 0x8000)
    # Checksum plus complement always contribute 510 to the final byte sum.
    checksum = (sum(rom) + 510) & 0xFFFF
    rom[0x7FDC:0x7FE0] = struct.pack('<HH', checksum ^ 0xFFFF, checksum)
    return bytes(rom)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    args.output.write_bytes(build_rom())

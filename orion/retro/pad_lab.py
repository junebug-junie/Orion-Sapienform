"""Generate Orion Pad Lab v1: an original two-controller NES diagnostic ROM.

No third-party game code/art/assets. iNES mapper 0, 16 KiB PRG, 8 KiB CHR.
P1 changes the left panel; P2 changes the right panel. A pulse channel emits
an audible tone. Source and generated ROM: MIT; see PAD_LAB_LICENSE.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import struct


class Program:
    """Tiny label resolver for this fixed 6502 program (not a general assembler)."""
    def __init__(self):
        self.code = bytearray()
        self.labels = {}
        self.references = []

    def emit(self, *values):
        self.code.extend(values)

    def label(self, name):
        if name in self.labels:
            raise ValueError(f"duplicate label: {name}")
        self.labels[name] = len(self.code)

    def branch(self, opcode, label):
        self.emit(opcode, 0)
        self.references.append((len(self.code) - 1, label, True))

    def jump(self, label):
        self.emit(0x4C, 0, 0)
        self.references.append((len(self.code) - 2, label, False))

    def finish(self):
        for position, label, relative in self.references:
            target = self.labels[label]
            if relative:
                offset = target - position - 1
                if not -128 <= offset <= 127:
                    raise ValueError("branch out of range")
                self.code[position] = offset & 255
            else:
                self.code[position:position + 2] = struct.pack("<H", 0x8000 + target)
        return bytes(self.code)


def build_rom() -> bytes:
    p = Program()
    # SEI, CLD, stack=$ff; disable rendering, DMC, frame IRQ.
    p.emit(0x78, 0xD8, 0xA2, 0xFF, 0x9A, 0xA9, 0)
    for address in (0x2000, 0x2001, 0x4010):
        p.emit(0x8D, address & 255, address >> 8)
    p.emit(0xA9, 0x40, 0x8D, 0x17, 0x40)
    for name in ("warmup1", "warmup2"):
        p.label(name)
        p.emit(0x2C, 0x02, 0x20)  # BIT PPUSTATUS
        p.branch(0x10, name)  # BPL
    p.emit(0xA9, 0, 0xA2, 0)
    p.label("clear")
    for page in range(8):
        p.emit(0x9D, 0, page)  # STA page,X
    p.emit(0xE8)
    p.branch(0xD0, "clear")
    # Initialize palette, four identical palettes; slot 1 is visible tile ink.
    p.emit(0xA9, 0x3F, 0x8D, 0x06, 0x20, 0xA9, 0, 0x8D, 0x06, 0x20)
    p.emit(0xA2, 32, 0xA9, 0x11)
    p.label("palette")
    p.emit(0x8D, 0x07, 0x20, 0xCA)
    p.branch(0xD0, "palette")
    # 960 name-table cells all use tile 1 (solid palette slot 1).
    p.emit(0xA9, 0x20, 0x8D, 0x06, 0x20, 0xA9, 0, 0x8D, 0x06, 0x20)
    p.emit(0xA9, 1, 0xA0, 3, 0xA2, 0)
    p.label("tiles_page")
    p.emit(0x8D, 0x07, 0x20, 0xE8)
    p.branch(0xD0, "tiles_page")
    p.emit(0x88)
    p.branch(0xD0, "tiles_page")
    p.emit(0xA2, 192)
    p.label("tiles_tail")
    p.emit(0x8D, 0x07, 0x20, 0xCA)
    p.branch(0xD0, "tiles_tail")
    # Attribute columns 0..3 select palette 0, columns 4..7 palette 1.
    p.emit(0xA2, 0)
    p.label("attributes")
    p.emit(0x8A, 0x29, 7, 0xC9, 4)
    p.branch(0x90, "left")
    p.emit(0xA9, 0x55)
    p.jump("attribute_write")
    p.label("left")
    p.emit(0xA9, 0)
    p.label("attribute_write")
    p.emit(0x8D, 0x07, 0x20, 0xE8, 0xE0, 64)
    p.branch(0xD0, "attributes")
    # APU pulse 1, constant volume, length halt, ~440 Hz.
    for address, value in ((0x4015, 1), (0x4000, 0xBF), (0x4001, 8),
                           (0x4002, 0xFD), (0x4003, 8)):
        p.emit(0xA9, value, 0x8D, address & 255, address >> 8)
    p.label("frame")
    p.emit(0x2C, 0x02, 0x20)
    p.branch(0x10, "frame")
    # Latch both pads, read 8 serial bits into $00/$01.
    p.emit(0xA9, 1, 0x8D, 0x16, 0x40, 0xA9, 0, 0x8D, 0x16, 0x40)
    p.emit(0x85, 0, 0x85, 1, 0xA2, 8)
    p.label("pads")
    p.emit(0xAD, 0x16, 0x40, 0x4A, 0x26, 0)
    p.emit(0xAD, 0x17, 0x40, 0x4A, 0x26, 1, 0xCA)
    p.branch(0xD0, "pads")
    for port in (0, 1):
        p.emit(0xA5, port)
        p.branch(0xF0, f"neutral{port}")
        p.emit(0xE6, 2 + port)  # each pad increments only its own counter
        p.label(f"neutral{port}")
        p.emit(0xA9, 0x3F, 0x8D, 0x06, 0x20)
        p.emit(0xA9, 1 + 4 * port, 0x8D, 0x06, 0x20)
        p.emit(0xA5, 2 + port, 0x29, 15, 0x09, 0x10, 0x8D, 0x07, 0x20)
    # Reset scroll/address latch after palette writes; enable background.
    p.emit(0xA9, 0, 0x8D, 0x05, 0x20, 0x8D, 0x05, 0x20, 0x8D, 0x00, 0x20)
    p.emit(0xA9, 0x0A, 0x8D, 0x01, 0x20)
    p.jump("frame")
    p.label("interrupt")
    p.emit(0x40)  # RTI; interrupts disabled, but vectors are defined.
    program = p.finish()
    prg = program + bytes(0x4000 - 6 - len(program))
    handler = 0x8000 + p.labels["interrupt"]
    prg += struct.pack("<HHH", handler, 0x8000, handler)
    chr_data = bytes(16) + b"\xff" * 8 + bytes(8) + bytes(8192 - 32)
    return b"NES\x1a" + bytes([1, 1, 0, 0]) + bytes(8) + prg + chr_data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    with args.output.open("xb") as output:
        output.write(build_rom())


if __name__ == "__main__":
    main()

"""Read-only, fingerprint-gated Nova v1.0.6a RAM observations.

Map from NovaSquirrel/NovaTheSquirrel 02a3913, src/memory.s + nova.x.
No reward, death, completion, or universal platformer interpretation is inferred.
"""
import ctypes as C

ROM = 'e4780e90b9d1587489bfb797d2ca395be21371ea9262fa9f87f99324ec6960ab'
CORES = {
    'ff0aa3f2e1550e7e5231fd0b175e2ad4e09046d5eab658e716cd81e79206fab2',
    'b8ba56fbeba5bf7658da89fba97ae1049eafe583320db490c2bf891934bdda22',
}
PROFILE = 'nova-1.0.6a-ram-v1'


class Observer:
    def __init__(self, core):
        self.core = core
        self.supported = (core.identity.get('rom_sha256') == ROM
                          and core.identity.get('core_sha256') in CORES)
        if self.supported:
            core.lib.retro_get_memory_data.argtypes = [C.c_uint]
            core.lib.retro_get_memory_data.restype = C.c_void_p
            core.lib.retro_get_memory_size.argtypes = [C.c_uint]
            core.lib.retro_get_memory_size.restype = C.c_size_t

    def read(self):
        if not self.supported:
            return {'status': 'unknown', 'reason': 'unsupported_rom_or_core'}
        size = self.core.lib.retro_get_memory_size(2)
        pointer = self.core.lib.retro_get_memory_data(2)
        if size != 2048 or not pointer:
            return {'status': 'unknown', 'reason': 'ram_unavailable'}
        return decode(C.string_at(pointer, size))


def decode(ram):
    if len(ram) != 2048 or not 1 <= ram[0x4b] <= 8 or ram[0x46] not in (0, 1):
        return {'status': 'unknown', 'reason': 'unqualified_game_state'}
    return dict(status='observed', profile=PROFILE, health_half_hearts=ram[0x4b],
                x_subpixels=ram[0x26]*256+ram[0x25],
                y_subpixels=ram[0x27]*256+ram[0x28], level=ram[0xa7])


def outcome(before, after, goal):
    if (before.get('status') != 'observed' or after.get('status') != 'observed'
            or before['level'] != after['level']):
        return 'unknown'
    if after['health_half_hearts'] < before['health_half_hearts']:
        return 'health_decreased'
    # Position is a measured coordinate, not distance-to-exit or a reward.
    if goal in ('walk_left', 'walk_right', 'jump_left', 'jump_right'):
        return ('no_horizontal_progress' if before['x_subpixels'] == after['x_subpixels']
                else 'horizontal_position_changed')
    return 'unknown'

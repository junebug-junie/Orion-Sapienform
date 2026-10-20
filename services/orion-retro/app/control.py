"""Frame-authoritative NES leases, also exercised with a deterministic clock."""
import secrets

# libretro B, select, start, up/down/left/right, A. No SNES-only buttons.
NES_MASK = sum(1 << bit for bit in (0, 2, 3, 4, 5, 6, 7, 8))
# How long a lease survives without input. Browsers throttle page timers to about
# 1 Hz in some states (live 2026-10-20), so the heartbeat cannot be trusted at 50 ms.
# Button changes still arrive at once; socket close or blur releases immediately.
LEASE_TTL = 2.0


class Arbiter:
    def __init__(self, ports):
        self.ports = ports
        self.leases = {}
        self.epochs = {p: 0 for p in ports}
        self.neutral = set()

    def clear(self):
        released = list(self.leases)
        self.leases.clear()
        self.neutral.update(released)
        return released

    def disconnect(self, connection):
        released = [p for p, v in self.leases.items() if v['connection'] == connection]
        for p in released:
            del self.leases[p]
        self.neutral.update(released)
        return released

    def control(self, connection, port, action, now):
        if port not in self.ports:
            raise ValueError('unsupported_port')
        old = self.leases.get(port)
        if action == 'release':
            if old and old['connection'] != connection:
                raise ValueError('not_owner')
            self.leases.pop(port, None)
            self.neutral.add(port)
            return {'port': port, 'released': True}
        if action == 'acquire' and old and old['expires'] > now:
            raise ValueError('port_owned')
        self.epochs[port] += 1
        self.neutral.add(port)
        item = dict(connection=connection, epoch=self.epochs[port],
                    token=secrets.token_urlsafe(32), expires=now+LEASE_TTL,
                    seq=-1, mask=0, latch=0, pending=None)
        self.leases[port] = item
        return {'port': port, 'lease': item['token'], 'epoch': item['epoch']}

    def submit(self, connection, data, now):
        p = data['port']; item = self.leases.get(p)
        if (not item or item['connection'] != connection or item['expires'] <= now
                or item['epoch'] != data['epoch']
                or not secrets.compare_digest(item['token'].encode(), data['lease'].encode())):
            raise ValueError('invalid_or_expired_lease')
        if data['seq'] <= item['seq']:
            raise ValueError('stale_sequence')
        if data['mask'] & ~NES_MASK:
            raise ValueError('unsupported_buttons')
        if data['mask'] & 48 == 48 or data['mask'] & 192 == 192:
            raise ValueError('opposing_directions')
        if item['pending'] is not None:
            # The previous input never reached a frame. Keep its buttons for one frame
            # so a quick tap (press and release in the same frame) is not lost.
            # Live 2026-10-20: Start never reached Crystalis's title screen.
            for axis in (48, 192):  # a newer latched direction replaces an older one
                if item['mask'] & axis: item['latch'] &= ~axis
            item['latch'] |= item['mask']
        item.update(seq=data['seq'], mask=data['mask'], expires=now+LEASE_TTL, pending=data['seq'])

    def frame(self, now):
        masks = [0, 0]; applied = []; expired = []
        for p, item in list(self.leases.items()):
            if now >= item['expires']:
                del self.leases[p]; expired.append(p)
                continue
            if p in self.neutral:
                continue
            mask = item['mask'] | item['latch']
            for axis in (48, 192):  # a latched direction yields to the current opposite one
                if mask & axis == axis: mask = (mask & ~axis) | (item['mask'] & axis)
            masks[p] = mask; item['latch'] = 0
            if item['pending'] is not None:
                applied.append({'port': p, 'epoch': item['epoch'], 'seq': item['pending'],
                                'mask': mask})
                item['pending'] = None
        neutral = sorted(self.neutral)
        self.neutral.clear()
        return masks, applied, sorted(set(expired + neutral))

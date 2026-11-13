"""Pruning of this service's own frames in the shared frames directory.

Kept free of the service's heavy imports (camera, detectors) so it can be
tested on its own.
"""

import logging
import os
import re
import time

logger = logging.getLogger("orion-vision-edge.capture")


# Files this service writes into the shared frames dir: capture frames
# (`frame_<ms>_<hex6>.jpg`, below) and on-demand stills plus their temp files
# (`still_<ms>_<hex6>.jpg`, `.still_..._.jpg.tmp`, app/still.py). Nothing else.
OWNED_FRAME_RE = re.compile(r"^\.?(?:frame|still)_\d+_[0-9a-f]{6}\.jpg(?:\.tmp)?$")


def cleanup_old_frames(directory: str, max_age: int) -> int:
    """Delete this service's own frames older than max_age. Never raises.

    Ownership: the frames dir is shared with orion-vision-retina instances
    (walkway, doorbell), which prune their own files at their own retention.
    This used to delete EVERY file older than 60 s, so walkway frames lived
    60 s instead of their configured 300 s, and the two pruners raced.

    A file can still vanish between listing and stat (orion-sql-writer reads
    this dir; a stray second edge could share it). The old isfile()+getmtime()
    pair raised on that race and abandoned the rest of the sweep; the same
    race killed the walkway retina's capture task, 2026-10-17. One stat per
    file, skip whatever is already gone.
    """
    now = time.time()
    removed = 0
    try:
        with os.scandir(directory) as entries:
            for entry in entries:
                if not OWNED_FRAME_RE.match(entry.name):
                    continue
                try:
                    if not entry.is_file(follow_symlinks=False):
                        continue
                    if now - entry.stat(follow_symlinks=False).st_mtime <= max_age:
                        continue
                    os.remove(entry.path)
                    removed += 1
                except FileNotFoundError:
                    continue
                except OSError as e:
                    logger.debug(f"Cleanup skip {entry.path}: {e}")
    except FileNotFoundError:
        return removed
    except OSError as e:
        logger.warning(f"Cleanup failed: {e}")
    return removed

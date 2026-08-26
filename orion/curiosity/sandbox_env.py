"""Hand Orion's own credentials to the `claude -p` subprocess that IS Orion.

THE PROBLEM THIS SOLVES, STATED PLAINLY. `orion/harness/fcc_motor.py` already
reads `~/.fcc/.env` -- but only to resolve a model label. The subprocess
environment it builds is `os.environ.copy()`, which is the harness-governor
CONTAINER's environment, and the curiosity credentials are not in it (measured
live 2026-08-26: `env | grep -c ORION_CURIOSITY` inside
`orion-athena-harness-governor` returned 0). So without this module the
credentials sit in a mounted file that the prompt would have to teach Orion to
parse by hand.

THIS DOES NOT WIDEN THE BOUNDARY, and that is worth being precise about rather
than assuming. `${HOME}/.fcc:/root/.fcc` is already mounted read-write into the
harness governor, the subprocess already runs as root, and FCC turns already
have `Bash` -- so `/root/.fcc/.env` is already readable from inside a turn
today. Exporting seven keys from it changes how ergonomic the credentials are,
not who can reach them. What the credentials THEMSELVES allow is the real
boundary, and it is enforced by Postgres and FalkorDB, not by this file:

  ORION_CURIOSITY_PG_DSN      role `orion_readonly`: SELECT on exactly four
                              tables, no INSERT/UPDATE/DELETE, no CREATE.
                              Verified live -- every write case errored.
  ORION_CURIOSITY_GRAPH_*     ACL user `orion_curiosity`: RO on the Atlas,
                              RW on `orion_worldview`, denied everywhere else.

THE KILL SWITCH IS THE ABSENCE OF THE KEYS, deliberately, and there is no new
env var for it. A flag would have to be added to the harness service's explicit
compose `environment:` allowlist to reach the container at all, which is
exactly how a kill switch ends up configured everywhere and present nowhere.
Removing a key from `~/.fcc/.env` removes it from the subprocess, immediately,
with no compose edit and nothing to keep in sync.

WHY AN ALLOWLIST AND NOT THE WHOLE FILE. `~/.fcc/.env` also holds provider API
keys, a Cloudflare token and a GitHub PAT. Copying it wholesale into every
turn's environment would hand all of those to a subprocess that has no business
with them, to solve a problem about four Postgres tables.
"""

from __future__ import annotations

import logging
from typing import Mapping, MutableMapping

logger = logging.getLogger("orion.curiosity.sandbox_env")

# Exactly the keys Juniper placed in `~/.fcc/.env` for this feature. Adding to
# this list is a real capability decision, not housekeeping -- every entry is
# something a `claude -p` turn can then read straight out of its environment.
CURIOSITY_ENV_KEYS: tuple[str, ...] = (
    "ORION_CURIOSITY_PG_DSN",
    "ORION_CURIOSITY_GRAPH_HOST",
    "ORION_CURIOSITY_GRAPH_PORT",
    "ORION_CURIOSITY_GRAPH_USER",
    "ORION_CURIOSITY_GRAPH_PASSWORD",
    "ORION_CURIOSITY_GRAPH_OWN",
    "ORION_CURIOSITY_GRAPH_ATLAS",
)

# The one key without which the SQL half of a curiosity turn cannot happen at
# all. Its absence is logged loudly; the graph keys degrade more gracefully
# because `orion_worldview` access can also be reached through the same file.
_REQUIRED_FOR_SQL = "ORION_CURIOSITY_PG_DSN"


def inject_curiosity_credentials(
    env: MutableMapping[str, str],
    fcc_env: Mapping[str, str],
) -> list[str]:
    """Copy the allowlisted keys from the FCC env file into `env`, in place.

    Returns the names of allowlisted keys that were NOT found, so the caller
    can log an absence instead of discovering it later as prose inside a turn
    ("I tried to query Postgres but the DSN was empty"), which is the
    empty-shell-cognition failure with a helpful tone.

    An already-set NON-BLANK value in `env` wins over the file. That ordering
    exists so an operator can override a single key through the container's own
    environment for a one-off without editing the shared credentials file, and
    so this function is safe to call twice.

    A BLANK existing value does NOT win, and that is a review finding rather
    than a nicety. `env.setdefault` treats `""` as set, so a compose
    `environment:` entry naming `ORION_CURIOSITY_PG_DSN` with no value -- which
    is exactly how these keys get added, and exactly the shape of this repo's
    own absent-kill-switch incident -- would silently shadow the real DSN from
    the file AND suppress the `curiosity_credentials_absent` warning, because
    the key was present in `fcc_env` and never counted as missing.
    """
    missing: list[str] = []
    for key in CURIOSITY_ENV_KEYS:
        value = str(fcc_env.get(key) or "").strip()
        if not value:
            missing.append(key)
            continue
        if str(env.get(key) or "").strip():
            continue  # a real operator override; leave it alone
        env[key] = value
    if _REQUIRED_FOR_SQL in missing:
        logger.warning(
            "curiosity_credentials_absent missing=%s -- the FCC sandbox will "
            "have no Postgres DSN, so a curiosity turn cannot read its own "
            "crystallizations; check ~/.fcc/.env and the %s mount",
            ",".join(missing),
            "/root/.fcc",
        )
    elif missing:
        logger.info("curiosity_credentials_partial missing=%s", ",".join(missing))
    return missing

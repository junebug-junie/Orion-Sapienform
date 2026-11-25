# Handoff: fix the Docker door (review findings 1-7), then a second review

Date: 2026-11-25. From the session that designed and built the door. For a new agent session.

## What this is

Orion's turns run as root with the real Docker socket. The **Docker door** is a small proxy in front of the socket. It lets Orion look (`ps`, `logs`, `inspect`) and `docker exec` into ordinary organs. It refuses changes to containers. It also refuses `exec` into containers that lead back to the host. Every call is recorded to a host file.

It ships as a **canary**: Orion's turns use the door only inside a scheduled time window, and use the normal socket outside it.

- Design (read first): `docs/superpowers/specs/2026-11-24-orion-safe-autonomy-in-place-design.md`, PR #3790 (branch `docs/orion-project-workshop`). The sections "Rollout: canary, then swap" and "Code review of the held build" are the ones that matter for this task.
- Code: branch `feat/docker-door-canary`, commit `bad427158`, pushed but **no PR yet**. Worktree: `/mnt/scripts/Orion-Sapienform-docker-door-canary`.

## Your task

1. Fix review findings 1-7 in the design doc's "Code review of the held build" section. Each fix gets a test.
2. Run a second, independent code review in a subagent. Give it the branch diff (`git diff origin/main...HEAD`) and the adversarial focus below, not this handoff's conclusions. Fix what it finds.
3. Open the PR (CLAUDE.md section 18 report), get CI green, and hand it to Juniper. **Do not merge.**
4. Ask Juniper for a new canary window, and set it only after the deploy (see "Canary window" below).

## The findings, short form

1. **Critical: bare-LF smuggling.** `parse_head` splits only on `\r\n`, but dockerd also accepts a bare `\n`. So an allowed `GET` can carry a hidden second request (for example a `kill`). Fix: strict parsing. Refuse a bare `\n`, a stray `\r`, folded header lines, any version other than HTTP/1.0 or 1.1, and any method that is not a plain token.
2. **Critical: `Upgrade` switches off the close.** `keep_upgrade` applies to every `exec_run`, including `resize` and a detached `start`, so a second request rides the open connection. Fix: always send `Connection: close`. dockerd decides on an upgrade from the `Upgrade` header alone.
3. **High: the exec body.** Refuse an exec whose body sets `"Privileged": true`. Also refuse an exec create whose body the door cannot read: chunked, more than 1 MB, missing `Content-Length`, or a bad `Content-Length`.
4. **Medium: `host_reach` is too narrow.** Add `CapAdd`, `Devices`, `NetworkMode`/`IpcMode`/`UsernsMode` = `host`, any runtime socket mount (`containerd.sock`, or any source path ending `.sock` under `/run` or `/var/run`), and binds of `/proc`, `/sys`, `/dev`, `/etc`, `/root`. Match the socket by mount source too, not only by destination.
5. **Medium: hangs.** Check the caller before reading the head. Add read timeouts on the head and the body. Catch bad lengths and `IncompleteReadError`. Close the client in a `finally`. A failed record write must refuse the request, not hang it.
6. **Low:** the door probe in `docker_door_canary.door_answers` blocks the governor's event loop on warm acquires (`fcc_warm_pool.py`). Run it with `asyncio.to_thread`, or make that path async.
7. **Low:** percent-decode the path before `classify` (`%61ttach`).

Finding 8 needs no code change. The canary is not enforcement while the governor still mounts the socket. Say so in the PR report.

## The test gap that hid 1 and 2

The fake daemon in `services/orion-docker-door/tests/test_door.py` always closes after one request. Replace it with a **keep-alive** fake that loops: it reads request after request on the same connection and closes only when it sees `Connection: close`. Every bypass test must assert that the fake daemon received exactly one request. Add tests for each case in findings 1-7. Re-create the bypass proof scripts as tests; the original scripts are gone, because the scratchpad was cleared.

## How to check

```bash
cd /mnt/scripts/Orion-Sapienform-docker-door-canary
PYTHONDONTWRITEBYTECODE=1 /mnt/scripts/Orion-Sapienform/.venv/bin/python -m pytest -q -p no:cacheprovider \
  services/orion-docker-door/tests orion/harness/tests/test_docker_door_canary.py \
  orion/harness/tests orion/fcc/tests -k "door or mcp or warm or subprocess_env or spawn"
```

Live smoke. This is safe: it uses a temporary door that you remove afterwards, and nothing is deployed.

```bash
docker build -q -f services/orion-docker-door/Dockerfile -t orion-docker-door:smoke .
docker run -d --rm --name orion-docker-door-smoke --network app-net \
  -v /var/run/docker.sock:/var/run/docker.sock -v <scratch>/rec:/record \
  -e DOCKER_DOOR_ALLOWED_CALLERS=orion-athena-harness-governor orion-docker-door:smoke
G="docker exec -e DOCKER_HOST=tcp://orion-docker-door-smoke:2375 orion-athena-harness-governor"
$G docker ps; $G docker exec orion-athena-thought python -c "print(1)"            # must work
$G docker exec -i orion-athena-sql-db psql -U orion_readonly -d conjourney -Atc "select 1"  # dig's path, must work
$G docker restart orion-athena-thought; $G docker exec orion-athena-hub true      # must be refused
docker stop orion-docker-door-smoke; docker rmi orion-docker-door:smoke
```

Never send a real change request (kill, restart) to the real daemon to "prove" a bypass. Use the keep-alive fake daemon. Read-only probes against the real socket are fine. `orion-athena-thought` is a live organ.

## Adversarial focus for the second review

Any way to get a non-look, non-exec request to the daemon, or an exec into a host-reaching container:

- smuggling: CRLF/LF mixes, pipelining, chunked bodies, `Expect: 100-continue`, `Content-Length` together with `Transfer-Encoding`
- path tricks: encoding, `..`, double version prefixes, query strings
- upgrade abuse
- exec-id reuse between containers
- races between the door's inspect and the exec

Also check:

- hangs and leaks in the pipe code
- the canary breaking turns outside the window
- a warm slot left stuck on the door

## Canary window

- Juniper chose a full day for the first window, but no date is fixed. The first date passed before deploy, so I set `ORION_DOCKER_DOOR_CANARY_WINDOWS` back to empty in the live `services/orion-harness-governor/.env` (primary checkout).
- After Juniper merges and the governor plus `orion-docker-door` are rebuilt, ask her for a day. Set it in the live `.env`. A window value change needs a governor recreate; the end of a window needs nothing.
- The new service's live `.env` already exists: `services/orion-docker-door/.env` in the primary checkout, gitignored.
- The rebuild planner already lists `orion-docker-door build` for this change.
- Report what broke: `python3 services/orion-docker-door/app/report.py --since <start> --until <end>`. Also run `docker logs orion-athena-harness-governor 2>&1 | grep DOCKER_DOOR_CANARY`.

## Do not

- Remove the socket from the governor. That is the later swap, after a clean window.
- Merge the PR.
- Use `git stash` (it is shared across worktrees).

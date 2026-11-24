#!/usr/bin/env bash
# Planned power outage: shut a host down cleanly, then prove every container
# came back after boot.
#
#   scripts/outage_power.sh down [-y] [--dry-run]   # before the outage
#   scripts/outage_power.sh up                      # after the host boots
#
# Why this does NOT `docker stop` anything:
#   A container you `docker stop` is marked "manually stopped". A container
#   with restart=unless-stopped then STAYS DOWN after reboot. That is the
#   "I'll miss one of 90 services" trap. If instead the whole host powers off,
#   the Docker daemon stops every container itself (SIGTERM, then SIGKILL after
#   the container's stop timeout) and does NOT mark them manually stopped, so
#   every unless-stopped/always container starts again on boot by itself.
#   This matches orion-power-guard, which also just runs `shutdown -h now`.
#
# What `down` adds on top of a plain shutdown:
#   1. Saves a list of running containers to ~/.orion/power/ (the "before" list).
#   2. Warns about running containers with restart=no: those do not return on
#      their own. `up` starts them.
#   3. Runs CHECKPOINT in every Postgres container, so the daemon's short stop
#      timeout cannot leave a long crash recovery.
#   4. sync, then `sudo systemctl poweroff`.
#
# What `up` does:
#   Waits for Docker, compares the "before" list to what runs now, runs
#   `docker start` on anything missing, waits, and reports what is still down
#   or crash-looping. Scratch containers (restart=no and not from a compose
#   project, e.g. a session's github-mcp-server or a test Postgres) are
#   reported but not started. Exit 0 = everything is back.
set -uo pipefail

SNAP_DIR="${HOME}/.orion/power"
HOST="$(hostname -s)"
LATEST="${SNAP_DIR}/running-${HOST}-latest.tsv"

die() { echo "ERROR: $*" >&2; exit 1; }

docker_ok() { docker info >/dev/null 2>&1; }

snapshot() {
  # name <TAB> restart policy <TAB> compose project ("-" if none) <TAB> image
  docker ps -q | xargs -r docker inspect --format \
    '{{.Name}}	{{or .HostConfig.RestartPolicy.Name "no"}}	{{with index .Config.Labels "com.docker.compose.project"}}{{.}}{{else}}-{{end}}	{{.Config.Image}}' \
    | sed 's#^/##' | sort
}

cmd_down() {
  local yes=0 dry=0
  for a in "$@"; do
    case "$a" in
      -y|--yes) yes=1 ;;
      --dry-run) dry=1 ;;
      *) die "unknown flag: $a" ;;
    esac
  done
  docker_ok || die "cannot talk to Docker (are you in the docker group?)"
  # Ask for the sudo password now, not after the checkpoints: a poweroff that
  # silently fails leaves the host to be hard-crashed by the outage.
  if [ "$dry" != 1 ]; then sudo -v || die "sudo failed; nothing was done"; fi

  mkdir -p "$SNAP_DIR"
  local snap="${SNAP_DIR}/running-${HOST}-$(date -u +%Y%m%dT%H%M%SZ).tsv"
  snapshot > "$snap"
  cp "$snap" "$LATEST"
  echo "Saved $(wc -l < "$snap") running containers -> $snap"

  echo
  echo "Running containers that will NOT restart by themselves (restart=no):"
  awk -F'\t' '$2=="no" { printf "  %-50s %s\n", $1, ($3=="-" ? "scratch, will be ignored" : "compose project " $3 ", `up` will start it") }' "$snap"
  awk -F'\t' '$2=="no"' "$snap" | grep -q . || echo "  (none)"

  echo
  echo "Checkpointing Postgres containers:"
  local pg
  for pg in $(awk -F'\t' '$4 ~ /postgres|timescale|pgvector|postgis/ {print $1}' "$snap"); do
    if [ "$dry" = 1 ]; then echo "  would CHECKPOINT $pg"; continue; fi
    if timeout -k 10 120 docker exec "$pg" sh -c \
        'psql -v ON_ERROR_STOP=1 -U "${POSTGRES_USER:-postgres}" -d "${POSTGRES_DB:-${POSTGRES_USER:-postgres}}" -c CHECKPOINT' >/dev/null 2>&1; then
      echo "  ok   $pg"
    else
      echo "  FAIL $pg (not fatal: Postgres will still shut down; it may replay WAL on boot)"
    fi
  done

  if [ "$dry" = 1 ]; then
    echo; echo "Dry run: not powering off."; return 0
  fi

  if [ "$yes" != 1 ]; then
    echo
    read -r -p "Power off ${HOST} now? Type yes: " ans
    [ "$ans" = "yes" ] || { echo "Aborted. Nothing was stopped."; return 1; }
  fi
  sync
  echo "Powering off ${HOST}. After boot run: $0 up"
  sudo systemctl poweroff || die "poweroff FAILED: ${HOST} is still up. Run: sudo systemctl poweroff"
}

cmd_up() {
  [ -f "$LATEST" ] || die "no before-list at $LATEST (was 'down' run on this host?)"

  echo "Waiting for Docker..."
  local i
  for i in $(seq 1 60); do docker_ok && break; sleep 5; done
  docker_ok || die "Docker is not up after 5 minutes: check 'systemctl status docker'"

  # Give restart policies a moment to finish bringing containers up.
  sleep 20

  local now missing name restart project
  now="$(snapshot | cut -f1)"
  missing=0
  echo "Containers that ran before the outage and are not running now:"
  while IFS=$'\t' read -r name restart project _; do
    grep -qxF "$name" <<<"$now" && continue
    if [ "$restart" = "no" ] && [ "$project" = "-" ]; then
      echo "  skip  $name (scratch container)"
      continue
    fi
    if docker start "$name" >/dev/null 2>&1; then
      echo "  start $name"
    else
      echo "  FAIL  $name (docker start failed; it may have been removed)"
    fi
  done < "$LATEST"

  restart_count() { docker inspect --format '{{.RestartCount}} {{.State.StartedAt}}' "$1" 2>/dev/null || echo missing; }
  declare -A before_rc=()
  while IFS=$'\t' read -r name _; do before_rc["$name"]="$(restart_count "$name")"; done < "$LATEST"

  echo
  echo "Waiting 60 s, then checking for crash loops..."
  sleep 60
  while IFS=$'\t' read -r name restart project _; do
    [ "$restart" = "no" ] && [ "$project" = "-" ] && continue
    local state
    state="$(docker inspect --format '{{.State.Status}}{{with .State.Health}} {{.Status}}{{end}}' "$name" 2>/dev/null || echo missing)"
    case "$state" in
      "running"|"running healthy"|"running starting")
        if [ "$(restart_count "$name")" != "${before_rc[$name]:-}" ]; then
          echo "  LOOP  $name: restarted during the 60 s wait"; missing=$((missing + 1))
        fi ;;
      *) echo "  DOWN  $name: $state"; missing=$((missing + 1)) ;;
    esac
  done < "$LATEST"

  echo
  if [ "$missing" = 0 ]; then
    echo "OK: every container from the before-list is running on ${HOST}."
  else
    echo "NOT OK: ${missing} container(s) still down on ${HOST}. Logs: docker logs --tail 50 <name>"
    return 1
  fi
}

case "${1:-}" in
  down) shift; cmd_down "$@" ;;
  up)   shift; cmd_up "$@" ;;
  *) echo "usage: $0 down [-y] [--dry-run] | up" >&2; exit 2 ;;
esac

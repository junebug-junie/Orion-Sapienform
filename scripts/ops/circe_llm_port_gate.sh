#!/usr/bin/env bash
# circe LLM port gate (GPU pool stage 6.6, layer 2 of the port gate).
#
# Only athena (where orion-gpu-pool and orion-llm-gateway run) may open a connection to circe's
# llama.cpp worker ports, the diffusion host and the lane-controller actuator. circe's own
# containers (docker bridges, e.g. the actuator's ready probe) and processes stay allowed.
# Everything else -- other tailnet hosts, the 192.168.1.x LAN -- is dropped.
#
# This CANNOT tell the gateway from any other athena container: they all share athena's source IP.
# It is defence in depth. The proof that callers go through the pool is the CI gate
# scripts/check_circe_worker_refs.py.
#
# Where the rules live, and why: the `mangle` table's PREROUTING chain. It runs before docker's
# DNAT (so the plain host port 8011 still matches, no conntrack needed), and neither docker nor
# tailscaled puts rules there. The obvious place, DOCKER-USER in the filter FORWARD chain, is only
# reached if docker's jump sits above tailscale's `ts-forward` (which accepts everything arriving on
# tailscale0); both insert at position 1 when they start, so a tailscaled restart would silently
# open the gate. ufw is no use either: it is disabled on circe, and docker-published ports bypass
# its INPUT rules. PREROUTING covers both paths a connection can take (DNAT->FORWARD and
# docker-proxy->INPUT). Not covered: one container calling another by name on its internal port
# (e.g. atlas-chat:8080) -- that never targets a host port; the CI gate is the fence for it.
#
# Runbook (verify + rollback): docs/runbooks/2026-10-01-circe-llm-port-firewall.md
#
# Usage (as root on circe):  circe_llm_port_gate.sh install|uninstall|apply|remove|status
#   One line from athena, nothing else to paste:
#     ssh -t circe@circe sudo /mnt/scripts/Orion-Sapienform/scripts/ops/circe_llm_port_gate.sh install
#   install   = copy this script + the systemd unit into place, enable it (applies now and every boot), show status
#   uninstall = stop + disable the unit (removes the rules), delete the installed copies
# DRY_RUN=1 prints the commands instead of running them (no root needed; used by the tests).
set -euo pipefail

ATHENA_V4="${ORION_PORT_GATE_ALLOW_V4:-100.92.216.81}"
ATHENA_V6="${ORION_PORT_GATE_ALLOW_V6:-fd7a:115c:a1e0::733:d851}"   # athena `tailscale ip -6`
# chat 8011, metacog 8012, fast 8013, diffusion 8014, agent 8015, agent-burst 8016, bonsai 8017;
# lane controller 8090; experiment 8099. tests/test_circe_llm_port_gate.py keeps this in step
# with the CI gate (config/gpu_pool.yaml + the worker services' *_HOST_PORT keys).
PORTS="8011:8017,8090,8099"
CHAIN=ORION-LLM-GATE

run() {
  if [[ "${DRY_RUN:-0}" == "1" ]]; then printf '%s\n' "$*"; else "$@"; fi
}

# remove_family <iptables|ip6tables>: delete every jump to our chain (whatever port list an older
# version used), then the chain itself. Safe to repeat.
remove_family() {
  local ipt="$1"
  if [[ "${DRY_RUN:-0}" == "1" ]]; then
    printf '%s -w -t mangle -S PREROUTING | grep -- "-j %s" | (each line, -A -> -D) %s -w -t mangle\n' \
      "$ipt" "$CHAIN" "$ipt"
    printf '%s -w -t mangle -F %s || true\n%s -w -t mangle -X %s || true\n' "$ipt" "$CHAIN" "$ipt" "$CHAIN"
    return
  fi
  local rule
  while read -r rule; do
    [[ -z "$rule" ]] && continue
    # shellcheck disable=SC2086  # iptables -S tokens, split on purpose
    "$ipt" -w -t mangle ${rule/#-A/-D}
  done < <("$ipt" -w -t mangle -S PREROUTING 2>/dev/null | grep -- "-j $CHAIN" || true)
  "$ipt" -w -t mangle -F "$CHAIN" 2>/dev/null || true
  "$ipt" -w -t mangle -X "$CHAIN" 2>/dev/null || true
}

# apply_family <iptables|ip6tables> <athena address/prefix>
apply_family() {
  local ipt="$1" athena="$2"
  remove_family "$ipt"
  run "$ipt" -w -t mangle -N "$CHAIN"
  run "$ipt" -w -t mangle -A "$CHAIN" -i lo -j RETURN
  # circe's containers, matched by interface, not address: docker's address pools move.
  run "$ipt" -w -t mangle -A "$CHAIN" -i docker0 -j RETURN
  run "$ipt" -w -t mangle -A "$CHAIN" -i br-+ -j RETURN
  run "$ipt" -w -t mangle -A "$CHAIN" -s "$athena" -j RETURN
  run "$ipt" -w -t mangle -A "$CHAIN" -j DROP
  run "$ipt" -w -t mangle -I PREROUTING 1 -p tcp -m multiport --dports "$PORTS" -m addrtype --dst-type LOCAL -j "$CHAIN"
}

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SBIN=/usr/local/sbin/orion-llm-port-gate
UNIT=/etc/systemd/system/orion-llm-port-gate.service

case "${1:-}" in
  install)
    run install -m 0755 "$HERE/circe_llm_port_gate.sh" "$SBIN"
    run install -m 0644 "$HERE/orion-llm-port-gate.service" "$UNIT"
    run systemctl daemon-reload
    # restart, not just enable --now: re-applies the rules when the script changed on a re-install
    run systemctl enable orion-llm-port-gate.service
    run systemctl restart orion-llm-port-gate.service
    run "$SBIN" status
    ;;
  uninstall)
    run systemctl disable --now orion-llm-port-gate.service || true
    run rm -f "$SBIN" "$UNIT"
    run systemctl daemon-reload
    ;;
  apply)
    apply_family iptables "$ATHENA_V4/32"
    apply_family ip6tables "$ATHENA_V6/128"
    ;;
  remove)
    remove_family iptables
    remove_family ip6tables
    ;;
  status)
    for ipt in iptables ip6tables; do
      run "$ipt" -w -t mangle -S PREROUTING
      run "$ipt" -w -t mangle -L "$CHAIN" -v -n
    done
    ;;
  *) echo "usage: $0 install|uninstall|apply|remove|status   (DRY_RUN=1 to print only)" >&2; exit 2 ;;
esac

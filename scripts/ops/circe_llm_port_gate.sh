#!/usr/bin/env bash
# circe LLM port gate (GPU pool stage 6.6, layer 2 of the port gate).
#
# Only athena (where orion-gpu-pool and orion-llm-gateway run) may open a NEW connection to
# circe's llama.cpp worker ports, the diffusion host and the lane-controller actuator. Containers
# on circe itself (docker bridges, e.g. the actuator's ready probe) and loopback stay allowed.
# Everything else -- other tailnet hosts, the 192.168.1.x LAN -- is dropped.
#
# This CANNOT tell the gateway from any other athena container: they all share athena's source IP.
# It is defence in depth. The proof that callers go through the pool is the CI gate
# scripts/check_circe_worker_refs.py.
#
# Why DOCKER-USER and not ufw: the ports are docker-published (`0.0.0.0:8011->8080`), so IPv4
# traffic is DNAT'd in PREROUTING and goes through FORWARD, never INPUT, and ufw's INPUT rules do
# not see it. Docker's documented hook for that path is the DOCKER-USER chain. Matching uses
# conntrack's ORIGINAL destination port, because after DNAT the packet's port is the container's
# (8080/6700), not 8011. IPv6 [::]:801x is served by docker-proxy (docker has no ipv6 enabled on
# circe), which is INPUT, so v6 gets a plain INPUT drop for non-loopback.
#
# Runbook (verify + rollback): docs/runbooks/2026-10-01-circe-llm-port-firewall.md
#
# Usage (as root on circe):  circe_llm_port_gate.sh apply|remove|status
# DRY_RUN=1 prints the commands instead of running them (no root needed; used by the tests).
set -euo pipefail

ATHENA_IP="${ORION_PORT_GATE_ALLOW_IP:-100.92.216.81}"
# Worker ports: chat 8011, metacog 8012, fast 8013, diffusion 8014, agent 8015, agent-burst 8016,
# bonsai 8017; lane controller 8090; experiment 8099. tests/test_circe_llm_port_gate.py keeps this
# in step with config/gpu_pool.yaml and the worker services' *_HOST_PORT keys.
PORT_RANGES=(8011:8017 8090 8099)
CHAIN=ORION-LLM-GATE
V6_PORTS="8011:8017,8090,8099"
V6_COMMENT="orion-llm-port-gate"

run() {
  if [[ "${DRY_RUN:-0}" == "1" ]]; then
    printf '%s\n' "$*"
  else
    "$@"
  fi
}

quiet() {  # best-effort delete: absent rule/chain is fine
  if [[ "${DRY_RUN:-0}" == "1" ]]; then
    printf '%s || true\n' "$*"
  else
    "$@" >/dev/null 2>&1 || true
  fi
}

remove() {
  for range in "${PORT_RANGES[@]}"; do
    # Loop: delete every copy, in case apply ran twice without remove.
    if [[ "${DRY_RUN:-0}" == "1" ]]; then
      quiet iptables -D DOCKER-USER -p tcp -m conntrack --ctdir ORIGINAL --ctorigdstport "$range" -j "$CHAIN"
    else
      while iptables -D DOCKER-USER -p tcp -m conntrack --ctdir ORIGINAL --ctorigdstport "$range" -j "$CHAIN" 2>/dev/null; do :; done
    fi
  done
  quiet iptables -F "$CHAIN"
  quiet iptables -X "$CHAIN"
  if [[ "${DRY_RUN:-0}" == "1" ]]; then
    quiet ip6tables -D INPUT -p tcp -m multiport --dports "$V6_PORTS" ! -i lo -m comment --comment "$V6_COMMENT" -j DROP
  else
    while ip6tables -D INPUT -p tcp -m multiport --dports "$V6_PORTS" ! -i lo -m comment --comment "$V6_COMMENT" -j DROP 2>/dev/null; do :; done
  fi
}

apply() {
  remove  # idempotent: start from nothing
  run iptables -N "$CHAIN"
  run iptables -A "$CHAIN" -s "${ATHENA_IP}/32" -j RETURN
  run iptables -A "$CHAIN" -s 172.16.0.0/12 -j RETURN    # docker bridges on circe (actuator ready probe)
  run iptables -A "$CHAIN" -s 127.0.0.0/8 -j RETURN
  run iptables -A "$CHAIN" -j DROP
  for range in "${PORT_RANGES[@]}"; do
    run iptables -I DOCKER-USER 1 -p tcp -m conntrack --ctdir ORIGINAL --ctorigdstport "$range" -j "$CHAIN"
  done
  run ip6tables -I INPUT 1 -p tcp -m multiport --dports "$V6_PORTS" ! -i lo -m comment --comment "$V6_COMMENT" -j DROP
}

status() {
  run iptables -S DOCKER-USER
  run iptables -L "$CHAIN" -v -n
  run ip6tables -S INPUT
}

case "${1:-}" in
  apply) apply ;;
  remove) remove ;;
  status) status ;;
  *) echo "usage: $0 apply|remove|status   (DRY_RUN=1 to print only)" >&2; exit 2 ;;
esac

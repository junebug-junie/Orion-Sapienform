# circe LLM port firewall (GPU pool stage 6.6, port gate layer 2)

**[GO, Juniper, sudo]** Every command below that starts with `sudo` is for Juniper to run on circe.
Agents may not run sudo (CLAUDE.md section 8). Nothing here has been applied yet.

## What it does, plainly

circe's model servers (the llama.cpp workers), the image generator and the GPU switcher all accept
connections from anywhere: any tailnet machine and anything on the 192.168.1.x home network can call
them directly and skip the GPU pool's queue. This firewall lets only athena (where the pool and the
LLM gateway run) open new connections to those ports. Containers on circe itself and circe's own
loopback stay allowed, so the GPU switcher can still check that a model came up.

It **cannot** tell the gateway apart from any other athena container, because they all leave athena
from the same address (100.92.216.81). So it is a second fence, not proof. The proof that callers go
through the pool is the CI check `scripts/check_circe_worker_refs.py`.

| port | what | role in `config/gpu_pool.yaml` |
|---|---|---|
| 8011 | chat worker | `chat` |
| 8012 | metacog worker | `metacog` |
| 8013 | fast worker | `fast` |
| 8014 | diffusion host | `diffusion` (pool-leased) |
| 8015 | agent worker | `agent` |
| 8016 | agent-burst worker (gpu2 swap seat) | `agent-gpu2` |
| 8017 | bonsai bake-off worker | (not a pool role; `BONSAI_HOST_PORT`) |
| 8090 | gpu-lane-controller (the pool's actuator) | `actuators.circe` |
| 8099 | experiment seat (dsv41) | `experiment` |

## What circe actually runs (inspected read-only 2026-10-01)

- `ufw` is installed and its systemd unit is "active", but `/etc/ufw/ufw.conf` says `ENABLED=no`, so
  ufw enforces nothing. `nftables`, `firewalld` and `netfilter-persistent` are inactive.
  `iptables v1.8.10 (nf_tables)`.
- The worker ports are **docker-published** (`0.0.0.0:8011->8080/tcp`). IPv4 traffic to them is
  rewritten (DNAT) before the host's INPUT chain, so a ufw rule would never see it. Docker's
  supported hook for that path is the `DOCKER-USER` chain, so that is where the rules go. They match
  on the connection's *original* destination port (conntrack `--ctorigdstport`), because after the
  rewrite the port is the container's (8080/6700), not 8011.
- IPv6 `[::]:8011` is served by docker's userland proxy (docker has no IPv6 enabled on circe), which
  is the INPUT chain, so IPv6 gets a plain INPUT drop for anything but loopback.
- Listening right now: 8011, 8012, 8013, 8014, 8015, 8090. (8016/8017/8099 are only up while their
  seats are loaded; the rules cover them anyway.)

The rules live in `scripts/ops/circe_llm_port_gate.sh` (`apply` / `remove` / `status`; `DRY_RUN=1`
prints the commands without running them). A systemd unit re-applies them whenever docker starts,
because nothing persists iptables rules across a reboot on circe.

## The exact rules

`DRY_RUN=1 scripts/ops/circe_llm_port_gate.sh apply` prints (after a cleanup pass that removes any
earlier copy):

```bash
iptables -N ORION-LLM-GATE
iptables -A ORION-LLM-GATE -s 100.92.216.81/32 -j RETURN
iptables -A ORION-LLM-GATE -s 172.16.0.0/12 -j RETURN
iptables -A ORION-LLM-GATE -s 127.0.0.0/8 -j RETURN
iptables -A ORION-LLM-GATE -j DROP
iptables -I DOCKER-USER 1 -p tcp -m conntrack --ctdir ORIGINAL --ctorigdstport 8011:8017 -j ORION-LLM-GATE
iptables -I DOCKER-USER 1 -p tcp -m conntrack --ctdir ORIGINAL --ctorigdstport 8090 -j ORION-LLM-GATE
iptables -I DOCKER-USER 1 -p tcp -m conntrack --ctdir ORIGINAL --ctorigdstport 8099 -j ORION-LLM-GATE
ip6tables -I INPUT 1 -p tcp -m multiport --dports 8011:8017,8090,8099 ! -i lo -m comment --comment orion-llm-port-gate -j DROP
```

## 0. Before (read-only, no sudo)

From athena, record that the workers answer today:

```bash
for p in 8011 8012 8013 8015; do printf "%s " $p; curl -sS -m 5 http://100.112.254.99:$p/health; echo; done
curl -sS -m 5 http://100.112.254.99:8014/ready; echo
curl -sS -m 5 -o /dev/null -w '%{http_code}\n' http://100.112.254.99:8090/
```

From a non-athena tailnet host (carbon-x1), record that they ALSO answer today (this is the hole):

```bash
ssh juniper@carbon-x1 'curl -sS -m 5 http://100.112.254.99:8011/health'
```

## 1. Install [GO, Juniper, sudo]

On circe, after this PR merges:

```bash
ssh circe@circe
cd /mnt/scripts/Orion-Sapienform && git pull --ff-only
sudo install -m 0755 scripts/ops/circe_llm_port_gate.sh /usr/local/sbin/orion-llm-port-gate
sudo install -m 0644 scripts/ops/orion-llm-port-gate.service /etc/systemd/system/orion-llm-port-gate.service
sudo systemctl daemon-reload
sudo systemctl enable --now orion-llm-port-gate.service
sudo /usr/local/sbin/orion-llm-port-gate status
```

`status` should show the three `-j ORION-LLM-GATE` lines at the top of `DOCKER-USER`, the four-line
`ORION-LLM-GATE` chain, and the `orion-llm-port-gate` DROP line in the IPv6 INPUT chain.

The script is copied to `/usr/local/sbin` on purpose: root should not execute a file the `circe`
user can edit in the checkout.

## 2. Verify

From athena (must still work; same commands as step 0):

```bash
for p in 8011 8012 8013 8015; do printf "%s " $p; curl -sS -m 5 http://100.112.254.99:$p/health; echo; done
curl -sS -m 5 http://100.112.254.99:8014/ready; echo
```

From carbon-x1 (must now time out, `curl: (28)`):

```bash
ssh juniper@carbon-x1 'curl -sS -m 5 http://100.112.254.99:8011/health; echo "exit=$?"'
```

From a container on circe (the actuator's ready probe path; must still work):

```bash
ssh circe@circe 'docker exec orion-circe-gpu-lane-controller python3 -c "import urllib.request as u; print(u.urlopen(\"http://100.112.254.99:8011/health\", timeout=5).read())"'
```

Drop counters move only when something other than athena/circe tries (run on circe):

```bash
sudo iptables -L ORION-LLM-GATE -v -n
```

End to end: one Hub chat turn answers, and the Hub GPU pool panel still shows every seat with a
recent grant. A pool swap (agent-gpu2 load) still completes, which proves the actuator's ready probe
gets through.

## 3. Rollback [GO, Juniper, sudo]

SSH is never affected (the rules only touch the ports above), so rollback is always reachable.

```bash
sudo systemctl disable --now orion-llm-port-gate.service   # ExecStop runs `remove`
sudo /usr/local/sbin/orion-llm-port-gate remove            # belt and braces; safe to repeat
sudo rm -f /etc/systemd/system/orion-llm-port-gate.service /usr/local/sbin/orion-llm-port-gate
sudo systemctl daemon-reload
sudo iptables -S DOCKER-USER                               # back to docker's default (-j RETURN only)
```

To let one more host in without removing the gate (e.g. a laptop running a benchmark), add a
RETURN line above the DROP and remember it is not persistent:

```bash
sudo iptables -I ORION-LLM-GATE 1 -s <host-tailnet-ip>/32 -j RETURN
```

## Known gaps

- Same-source blindness: any athena container can still reach the workers. The CI gate covers
  in-repo code; nothing covers an ad-hoc `curl` on athena.
- UNVERIFIED: whether a bare `dockerd` restart (not a reboot) keeps rules inside `DOCKER-USER`.
  The unit is `PartOf=docker.service`, so it is stopped and re-applied with docker either way.
- 192.168.1.x LAN clients lose access too. Checked 2026-10-01: no tracked file and no live `.env`
  on athena or circe uses circe's LAN addresses (192.168.1.22/.24) for these ports, and every live
  `.env` key holding a tailnet worker address is a dead key except thought's
  `ORION_DIFFUSION_HOST_BASE_URL` (athena, allowed). An out-of-repo tool on the LAN would break; use
  the RETURN line above.

# PR: the doorbell still hears nothing -- read only the `dh` key out of FCM's push header

## Summary

- After PR #3085 was deployed (11-03 23:48 UTC), the doorbell listener stayed up. But it skipped every Ring push with `ValueError: Invalid EC key`. As of 11-04 there were 6 skipped messages in a row and 0 `doorbell_motion` rows.
- Cause: Google's push service (FCM) sends the key header as two values, `dh=<sender key>;p256ecdsa=<VAPID key>`, with no base64 padding. The push library removes the first 3 characters (`dh=`) and decodes the rest as one value. While FCM padded the key, the decoder stopped at the `=` and never reached the second value. Without padding, the strict decode failed ("Incorrect padding", 10-28 to 11-03). Padded as a whole by #3085, it decodes to 137 bytes, which is not a 65-byte key ("Invalid EC key").
- Fix: `ring_source._header_param` takes only the `dh` value (and the `salt` value) from each header. It works in any order, with `;` or `,` separators. Each skip log line now also shows the header layout: parameter names and value lengths, never the values.
- No re-registration is needed. "Invalid EC key" is raised while reading the *sender's* key from the header (`EllipticCurvePublicKey.from_encoded_point` in http_ece's `derive_dh`). That happens before our private key or auth secret is used. A stale key would fail later, with `InvalidTag`.

## Outcome moved

- Before: every Ring push skipped. Health was correctly `degraded` with `ring_deaf: true`, 6 skips, last error `Invalid EC key`.
- After deploy: Ring pushes are decoded and delivered. UNVERIFIED until deployed.

## Current architecture

- `orion-vision-retina` with `RETINA_SOURCE_TYPE=ring` uses the push client from `ring_doorbell` 0.9.14 (`firebase-messaging` 0.4.5). PR #3085 patched `FcmPushClient`: it pads the two headers, and it skips and acknowledges a message that still fails.

## Root cause evidence

- Live log, 11-03 23:48 to 11-04 00:19: 6 lines `skipped (acknowledged, not delivered; N in a row): ValueError: Invalid EC key.` Health: `degraded`, `ring_deaf: true`.
- Simulation with real P-256 keys. Each candidate header shape went through both the 0.4.5 decode and the #3085 decode. Only the two-value header reproduces both observed errors in order:

| header shape | 0.4.5 | #3085 (pad only) |
|---|---|---|
| `dh=K` unpadded | Incorrect padding | OK (65 B) |
| `dh=K;p256ecdsa=V` unpadded | **Incorrect padding** | **Invalid EC key (137 B)** |
| `dh=K=;p256ecdsa=V=` padded (the pre-10-28 form) | OK (65 B) | OK |
| `p256ecdsa=V;dh=K` | Unsupported point type | Unsupported point type |
| `dh=K,p256ecdsa=V` | Incorrect padding | Invalid EC key (137 B) |
| standard alphabet, unpadded | Incorrect padding | OK |

- The padded row also explains why this worked before 10-28.
- A real captured message: UNVERIFIED. Capturing one read-only is not possible. A second FCM client with the same credentials would take over the live connection and acknowledge (consume) the doorbell's messages. The new skip line prints the header layout, so if a message is still skipped after deploy, its line shows the real shape.

## Architecture touched

`orion-vision-retina` only. No contracts, no env.

## Files changed

- `services/orion-vision-retina/app/ring_source.py`: adds `_header_param`, `_normalize_push_headers` (rewrites the two headers to `dh=<v>` / `salt=<v>` before the library parses them) and `_header_shape` (adds the layout to skip log lines).
- `services/orion-vision-retina/tests/test_ring_source.py`: regression tests on the live shape, using a real client and real aesgcm encryption:
  - The unpatched library gives "Incorrect padding".
  - Padding the whole value gives 137 bytes and "Invalid EC key".
  - The patched client delivers the message for 5 layouts: `;` separator, `,` separator, reversed order, extra `rs=`, and `keyid=`.
  - `_header_param` unit cases.
  - The skip line shows the layout but not the values.
- `services/orion-vision-retina/README.md`: the second stage of the incident.

## Schema / bus / API changes

- Added / Removed / Renamed: none.
- Behavior changed: a skip log line now ends with `headers crypto-key=[...] encryption=[...]`.
- Compatibility notes: none.

## Env/config changes

- None. `.env_example` is unchanged, so no sync was needed. No keys were skipped.

## Tests run

```text
pytest services/orion-vision-retina/tests -q -W ignore      -> 172 passed (py3.12 venv from requirements.txt)
same new tests inside the real image (py3.10, firebase-messaging 0.4.5) -> 17 passed
mutation: _normalize_push_headers call removed               -> 5 of 5 layout tests fail
```

## Evals run

```text
None. The retina has no eval for push delivery. The measurement is health ring_deaf plus ring_events_seen after deploy.
```

## Docker/build/smoke checks

```text
docker build -f services/orion-vision-retina/Dockerfile .   -> built (throwaway tag, removed)
```

The live smoke is UNVERIFIED (not deployed).

## Review findings fixed

The review ran in a subagent. It found no must-fix issues, and it backed up the root cause on its own:
- The exact text "Invalid EC key" needs decoded bytes that start with `0x04`. Random bytes, or a header that starts with `p256ecdsa` or `keyid`, give a different error. So the `dh` value comes first in the live header.
- A layout with the key in the body (aes128gcm) would fail with a different error.
- Stale credentials are ruled out. A bad private key fails earlier, in `load_der_private_key`. A wrong private key or auth secret fails later, with `InvalidTag`.
- Changing `app_data` in place is safe, and the ack still goes out.

- Finding: if `_header_shape` raised, a message was skipped even though that code only builds the log line.
  - Fix: it now has its own try/except and falls back to `"?"`.
  - Evidence: code inspection.
- Finding: values in quotes (`dh="..."`) kept their quotes.
  - Fix: the quotes are stripped.
  - Evidence: a new test layout `dh="{dh}";p256ecdsa="{vapid}"` and a new `_header_param` case.
- Finding: a header with no `=` could print its value in the skip log.
  - Fix: such a part prints as `?:<len>`.
  - Evidence: `test_header_shape_never_prints_a_bare_value`.
- Finding: the README live check grepped for "Incorrect padding", which stopped appearing once #3085 was deployed.
  - Fix: it now checks `ring_push_messages_skipped` and greps for both error texts.
- Finding: the test helper had two docstrings.
  - Fix: merged into one.

After the fixes: 172 passed.

## Restart required

```bash
scripts/safe_docker_build.sh orion-vision-retina up -d --build orion-vision-retina-doorbell
```

Run after merge, from the clean primary `main` checkout. Then, after the next motion at the door:

```bash
curl -s localhost:8029/health | python3 -c "import sys,json;e=json.load(sys.stdin);d=e['details'];print(e['status'],d['ring_deaf'],d['ring_events_seen'],d['ring_push_messages_skipped'])"
# want: ok False >=1 0
docker logs orion-vision-retina-doorbell 2>&1 | grep -E "skipped|ring event|ring motion|ring ding" | tail
docker exec orion-athena-sql-db psql -U postgres -d conjourney -Atc "select event_type, count(*), max(created_at) from vision_events where stream_id='doorbell' and event_type in ('doorbell_motion','doorbell_ring') and created_at > now()-interval '1 day' group by 1"
```

If a skip line still appears, its `headers ...` part shows the real layout. Send it back. Re-registration is only worth trying if the error changes to `InvalidTag`. The step would be: delete `/mnt/telemetry/vision/ring_state/ring_fcm_credentials.json`, then restart the doorbell container. ring_doorbell then registers a new FCM device.

## Risks / concerns

- Severity: medium.
  - Concern: the header shape is inferred, not seen in a captured message.
  - Mitigation: it is the only shape tested that reproduces both errors in order. The parser also handles reordered and `,`-separated forms. The skip line now reports the layout.
- Severity: low.
  - Concern: this is another runtime patch on a pinned third-party library.
  - Mitigation: the tests use the real library, so an upgrade that breaks the patch fails CI.

## PR link

https://github.com/junebug-junie/Orion-Sapienform/pull/3090

🤖 Generated with [Claude Code](https://claude.com/claude-code)

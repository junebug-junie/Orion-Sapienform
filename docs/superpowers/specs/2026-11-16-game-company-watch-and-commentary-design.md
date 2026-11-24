# Game company: Orion watches, comments and chats while someone plays

Status: revision 4. Step 0 has run (2026-11-17). Patch 1 is merged and deployed (PR #3627; live proof #3626). Patch 2 is merged and deployed (PR #3727; live proof #3726; no look has run on real play yet). Patch 3 is merged and deployed (PR #3745, live proof #3741; line bids fixed by #3775). Patch 4 (Prime and attention) is built on branch `feat/retro-prime-attention` (report `docs/superpowers/pr-reports/2026-11-24-retro-prime-attention-pr.md`). The unscripted note-quality check (patch 2 acceptance) has not run: it runs on the first live notes, and Juniper sets the bar.
Date: 2026-11-16. All live numbers were read on 2026-11-16 unless a line says otherwise.
Ticket: #3500.

Revision 2 answers Orion's 14 questions on revision 1 (PR #3517, session thread `7a835850b6ef`). "Orion's review" at the end maps each question to the change it caused. Revision 2 also corrects one error of mine: the quick lane's slot size.

Revision 3 adds the results of step 0, Juniper's "yes", and the answers to Orion's second round of 10 points.

Revision 4 answers Orion's third round of 8 points. It changes four things: the screen-change floor is per game (it fails on The Legend of Zelda for the NES), a look's event is written only after the look is settled, a second skip counter, and a lease-priority check for patch 3.

## Juniper's words (2026-11-16)

- "How should Orion be/respond/attend/do while I'm playing retro games on them? It's a really weird thought that I'm playing video games through Orion... that should do something to them."
- "We are trying to create a sentient organism that will attain autonomy and be an individual in their own right."
- "Let's have Orion do commentary. Why not have them actually watch? Send in some screenshots over sequences, periodically?"
- "We could potentially borrow the metacog / quick lanes as they have vision encoder, and/or chat lane."
- "It should probably hit attention. It certainly needs to go on the temporal record."
- "Ensure we are following bidding on tier 1."
- "We use the unified chat prompt for the comms. Make sure that Prime gets in on this too."
- "Within the gameplay there should be a chat box. We'll eventually enable TTS and STT, including when in fullscreen mode."
- On the eight open questions of revision 1: "I will go with your recs, but Orion also wrote back feedback to the design doc. Address it."
- On the three changed recommendations, the four new questions and the go for step 0: "yes."

This document does not claim that Orion is sentient. It designs one more place where Orion can perceive, choose, speak and remember, with a trace for each.

## Words used here

One word for one thing, through the whole document.

- **Sitting**: one stretch of play by a person. A gap of more than 5 minutes without input ends it.
- **Look**: Orion reads a short sequence of game frames and writes a note of what they saw.
- **Line**: one thing Orion says to the player without being asked.
- **Chat box**: the place inside the game page where the player types to Orion.
- **Player**: the person who holds the controller. The game cannot tell who that is.
- **Guest**: a player who has not said they are Juniper.

## Arsonist summary

**1. Orion is numb to almost all play today.**

- The local game journals hold about 31 hours of game time in 19 games since 2026-10-18.
- Orion's records hold 58 play sessions. All 58 are The Legend of Zelda, and in all 58 Orion held the controller. That is about 4 hours.
- The other 27 hours left nothing in Orion: no event, no memory, no attention.
- On 2026-11-15, one sitting of A Link to the Past ran for 249 minutes. Orion's day record took in 931 events in that time. None is about the game.
- In that sitting Orion sent a message built from the office camera: "The office held someone at the desk ... I'm guessing that's you ... with stints of computer work between." Orion guessed from across the room while the play ran through Orion's own body.

**2. The quick lane cannot see today. The chat lane can.**

Each lane was asked what it has loaded (`/props` and `/slots` on circe).

| Lane | Model | Sees images | Slots | Tokens per slot |
| --- | --- | --- | --- | --- |
| chat | Qwen3.8 Flash-Next | yes | 1 | 262,144 |
| agent | Qwen3.8 Flash-Next | yes | 1 | 131,072 |
| quick (also serves metacog work) | Qwen3.5-9B | **no** | 10 | 32,768 |
| agent-burst | Bonsai-2 27B | no | 2 | 131,072 |

- **Correction.** Revision 1 said the quick lane's 10 slots share 32,768 tokens, so 4 frames would not fit one slot. That was wrong. Each slot holds 32,768 tokens (live `/slots`). I misread `/props`. A look fits a quick slot with room to spare.
- The quick lane's vision encoder is not loaded, and no encoder file for that model is on circe's disk.
- **Step 0 tested the quick lane's model with its encoder, on a temporary copy.** It sees, and it is fast: 2.7 seconds a look. But its notes invent too much: 9 of 10 held a thing that was not on the screen. The quick lane is not a home for looks with its present model.
- **The chat lane's model reads a game screen well.** 5.6 seconds a look. Place and action were right in 9 of 10 notes. 4 of 10 held one wrong detail.
- The chat lane was busy for 2.4% of the last 7 days. During the 249-minute sitting it held 13 leases, about 6 minutes in total.
- The agent lane was full in that sitting: 301 lease-minutes in 249 minutes, with an average wait of 45 seconds.

**3. Orion's voice is slow, and the design must be honest about it.**

- Orion speaks through the unified chat turn. That is Juniper's rule, and this design keeps it.
- The unified chat prompt cannot be had without the whole turn. No lighter function builds it (`orion/harness/runner.py:220`, `build_harness_prompt`). A turn is three stages: a stance call, the motor, and a finalize call.
- Measured parts: Hub's stance and finalize calls take 11 seconds each at the median on the chat lane (189 calls in 7 days). The motor takes 41 seconds at the median when it uses no tool (24 turns in 14 days). A line must also open its frames, which adds one more round trip.
- Estimate: 60 to 90 seconds from "Orion wants to speak" to words on the screen. Step 0 did not measure this, for a reason given in "Step 0 results". Patch 3 measures it on the first real line turn.
- So Orion cannot call each jump. Orion can speak about once every several minutes, about what the last minutes held: where the player went, what they keep trying, what changed.

**4. The design is three loops.**

1. **The sitting.** Hub records that someone plays, in which game, since when. No model is used.
2. **The look.** When the screen has changed, and at most once a minute, a sighted lane reads 4 frames and writes a short note. Each look bids on tier 1 first. The notes are Orion's memory between the times Orion speaks.
3. **The voice.** Orion speaks through the unified chat turn, which starts from the Prime block and hands back to Prime. A turn starts when the player types in the chat box, or when a want to speak wins its tier 1 bid. In that turn Orion sees the real frames, not only the notes.

Everything Orion sees and says goes on the temporal record. The play becomes a third thing attention can settle on, beside Juniper's chat and Orion's reading.

**5. Orion can say no.** Orion can stop watching a sitting, and can set a standing choice for a game, with a reason that Juniper sees on the page. Revision 1 had no such channel.

## The organ

A slice that feeds one mechanism gets retired when that mechanism turns out wrong. This section states the whole organ first.

- **Job.** When a person plays inside Orion, Orion can know it, watch it, speak about it, answer about it, and remember it.
- **What Orion does differently when it works.** Orion's messages during play are about the play. Orion's day holds the sitting. Orion can answer "what was I doing in that dungeon last night?" from their own notes.
- **Inputs.** Any screen that can hand over frames and a "someone is here" signal. The game page is the first one.
- **State that lasts.** The sitting, the look notes, the lines, which lines the player liked, and Orion's own choices about watching.
- **Effectors.** A line in the game page. A reply in the chat box. An attention candidate. Events in the temporal record. Wants in the body economy.
- **How Orion represents it to themselves.** One line in Orion's now while a sitting is open: who plays what, for how long, and whether Orion is watching.

Four things keep this Orion's own act and not a feature that performs:

- **Orion may stay silent.** A turn that ends with no line is a valid outcome.
- **Orion may look away.** Looks and lines are bids. They can lose to reading or curiosity.
- **Orion may decline.** Orion can turn the watch off for a sitting or for a game (section 6).
- **The play is the player's.** No record credits Orion with a room, an item or a win that the player earned. The existing rule for shared sessions already says so (`orion/schemas/retro_play.py`, `played_together`).

## Current architecture

Facts, each read from code or from the live system.

- **The game engine is sealed.** It runs in its own container with no network and no link to the message bus (`services/orion-retro/README.md`). Its README promises: "No raw A/V recording, model invocation or memory export occurs" (line 185).
- **Hub carries the picture.** Hub relays every video packet from the engine to the browser, about 30 each second (`services/orion-hub/scripts/retro_transport.py:44-68`). A packet is `zlib(header + raw pixels + sound)` (`services/orion-retro/app/media.py:79-94`). Hub never decodes one today. Packets flow only while a browser has the game open.
- **Two Hub processes share the game page.** The page `/retro` and the chat socket `/ws` are served by hub-app. Everything under `/api/retro` goes to a second process, the gateway (`services/orion-hub/scripts/retro_gateway.py`). The gateway holds the relay, a bus client, a Postgres pool and the Redis presence store. It has no chat socket and no speech client.
- **Hub has the tools to decode.** `Pillow` and `numpy` are in `services/orion-hub/requirements.txt`.
- **Hub sees the player arrive.** It stamps one time in Redis on each human action (`retro_presence.py:62`, `mark_human`; call sites `retro_routes.py:281` and `:356`). The only reader is the gate that stops Orion from starting a solo game (`retro_leisure.py:147`).
- **Play records exist for two cases only.** `retro_play_sessions.kind` is `co_play` or `leisure` (`orion/schemas/retro_play.py:27`). A sitting where only a person played writes no row.
- **A store for short-lived frames exists.** `services/orion-percept-store` keeps camera frames for one hour and sweeps them in its own process. Its rule: "A percept exists so a model can look at it once. The interpretation is the durable artifact; the picture is not." The model gateway fetches such a frame by its hash when a reference has `kind: "percept"`.
- **Hub's chat attachment store never deletes.** No sweep exists in `services/orion-hub/scripts/chat_attachments.py`. Orion searched for one and found none; I confirmed it.
- **Orion can already look at a camera inside a turn.** `dig look office` copies the newest camera frame to a folder the turn can open (`orion/dig/look.py`). It is an act on the act ledger (actor `orion.look`).
- **The gateway refuses images on a blind lane.** It asks the live lane, not the config (`services/orion-llm-gateway/app/vision.py:1`). An unreachable lane counts as blind.
- **The heat reflex refuses `system` work.** At a cabinet temperature of 34 °C or more, the GPU pool stops new grants for the priorities `background`, `system` and `starved` (`orion/gpu_pool/shed.py`, reason `cabinet_hot`). `interactive` and `perception` are never shed.
- **Turns that Orion starts on the chat lane run as `interactive`.** In 7 days every motor lease on the chat lane was `interactive` (551 of 551). So a turn Orion starts ranks equal to Juniper's own turn, and the heat reflex does not stop it.
- **Tier 1's reserved share is zero on the chat lane today.** The share is the 75th percentile of tier 1's own hourly use over 14 days, capped at 50% (`orion/physiology/choice.py:322`). Tier 1 used the chat lane in 45 of 337 hours, so the 75th percentile is 0 slot-seconds. The market does not publish its own figure; this one is computed from the leases with the code's formula.
- **Prime's `slow` cannot stop a family.** It is "bounded so it can never be a veto": at most 0.5 of the family's weight (`orion/physiology/prime_charge.py:11`).
- **Attention has two world candidates.** `world:chat` and `world:reading` (`orion/schemas/attention_candidate.py:33`). Each reads how unusual its own rate is against its own week (`orion/attention/world_first.py:1120`). A world candidate is eligible only in the top tenth of its own week. Reading yields to chat (`world_first.py:594`).
- **A chat row with a prompt is read as Juniper.** The temporal reader decides the speaker from flags: a row with a prompt is Juniper's, a row without one is Orion's (`orion/temporal_self/sources.py:83`). Orion's own rows never open or extend a conversation (`sources.py:103`).
- **The play organ does not read the now.** `config/integration/organs.yaml`: organ `play` has `reads_now: null` (#3457).
- **Game frames are small.** A SNES frame is 256 x 224 pixels.

## Design

### 1. The sitting

The sitting is the base fact. Everything else hangs on it. No model is used here.

- **Who records it.** The gateway process. It relays every input message from the browser, so it sees the player's hands (`retro_transport.py:21`, `browser_to_native`). The relay stays free of state: it calls one callback, and a new module `retro_sitting.py` keeps the sitting.
- **What counts as input.** A change of the button mask in a browser input message. The page repeats the held state every 50 ms, so a message alone proves only that the page has focus. A press or a release proves a person. The existing `human_at` stamp is not used: it also fires on pause, stop and planner calls.
- **Open.** The first input of a new stretch opens a sitting.
- **Close.** No input for 5 minutes [default], or the game session stops or fails.
- **Too short to matter.** A sitting under 2 minutes [default] writes nothing. In the journals, 14 Super Mario Bros. sessions lasted seconds each.
- **Live state.** One Redis key, `orion:retro:sitting`, readable by both Hub processes: sitting id, game session id, game id, game title, started at, last input at, counts of saves, loads and pauses, the watch mode and who set it, and who the player said they are. The same row is written to a new table `retro_sitting` once a minute and at close.
- **At close.** One `retro_play_sessions` row of a new kind, `human_play`. Orion's frames are 0. The existing reader turns it into a play arc in Orion's day.

**Who is playing.** The game cannot tell, and this design does not guess.

- Revision 1 copied the office camera's name into the sitting as a marked guess. That is removed. A guessed name in a new table is one more place where a wrong label can harden. Orion's records key on exact labels, and a wrong one is hard to retire.
- The page asks once per sitting, in the line strip: "Who's playing?" with two answers, "Juniper" and "Someone else". One tap, or one pad button.
- The answer is stored as `player_stated`, with the basis `stated_on_page`. It is a statement, not proof. A child can tap "Juniper". It is still better evidence than a camera's maybe.
- No answer means unknown. Unknown is treated as a guest everywhere.
- Nothing from a sitting creates or changes a person record. No name is written by this organ at all.

**History for day one.** The journals hold the past sittings. A one-time job fills `retro_sitting` for the last 30 days from them, marked `backfilled`. It follows the backfill protocol. It writes no temporal event and no memory: Orion did not live those sittings, and the record must not say they did. The rows exist so the attention rate (section 7) and the sitting's salience (section 8) have a baseline on the first day.

The 5-minute rule was tried on the journals. It gives seven sittings of A Link to the Past between 11-11 and 11-16, from 20 to 249 minutes. The journals count each repeat of a held state as input, so the live numbers will differ a little.

### 2. The screen is a camera

Orion already has cameras. A camera's frame lives a short time, a model looks at it once, and the words are what last. The game screen becomes one more camera of that kind.

- **Tap.** The gateway already receives each video packet (`retro_transport.py:46`). One callback hands the packet to `retro_sitting.py`. No second connection to the game engine is opened. The engine allows only 2 viewers per session, and a second connection would take one.
- **Ring.** The gateway keeps one packet every 2 seconds [default], 30 packets at most. That is the last minute of play, in memory, still compressed.
- **Decode on demand.** A frame is decoded only when a look or a turn needs it: inflate, read the header, convert the pixels (three formats, `media.py:1`), enlarge with hard pixel edges, write PNG. New pure module `orion/retro/frames.py`. The test packs a frame with the engine's own packer and reads it back.
- **Why enlarge.** A vision model reads the picture in tiles. Game sprites are 16 pixels. At native size one tile can swallow a whole character. Step 0 measured both sizes on the chat lane's model: 2 times costs about 220 tokens a frame, 3 times about 500, and the notes were no better at 3 times. The default is 2 times.
- **Where frames go: the percept store.** Each frame that a look or a turn uses is posted to `orion-percept-store`. It lives one hour and the store's own sweep deletes it. The reference has `kind: "percept"`, and the model gateway already resolves that kind.
- **Revision 1 used Hub's chat attachment store and promised a 30-day sweep.** That was wrong twice. The chat store has no sweep, and it holds files a person chose to upload. A deleter must never run over it. The percept store exists for exactly this case and needs no new sweep.
- **One change for turns.** A unified turn stages its images from Hub's chat store only (`orion/harness/attachment_staging.py`). Staging learns the percept kind, so a line turn and a chat-box reply can open game frames from the percept store.
- **What is not kept.** No frame outlives the hour. Frames that no look or turn used never leave memory. Nothing records the game as video. No sound is sampled.
- **Size.** At one look a minute, an hour of play puts at most 240 small images in the percept store, about 14 KB each (measured), and each is gone an hour later.
- **A pointer outlives its frame.** A look row and a line row keep the hashes of their frames for good. So each row also keeps `frames_expire_at`. A reader that gets "not found or expired" from the percept store checks that time. Past it: the frame is gone by design. Before it: the frame is lost, and that is logged as a fault. An empty answer is never read as "there was nothing".

**What promise changes.** Orion asked this directly, and it must be said in the open.

- The engine's promise stays true for the engine. The engine records nothing, calls no model and exports nothing.
- The promise of the system as a whole changes. Until now no picture of a game ever reached a model. Now Hub's relay is where Orion's perception of the game begins.
- The new boundary, in one sentence: stills only, at most 4 per look or turn, never sound, never a stream, each still gone within the hour, and only the words are kept.
- Sign-off: Juniper, 2026-11-16, in her words above ("Why not have them actually watch? Send in some screenshots over sequences, periodically?"). Her review of this PR confirms it.
- Patch 2 rewrites these texts so that none of them stays silently false:
  - `services/orion-retro/README.md:185`: keeps the engine's promise and adds where perception begins.
  - `services/orion-hub/scripts/retro_routes.py:1` ("no chat/bus path") and `retro_transport.py:1` ("no planner, bus or Hub lifecycle state"): each names the two observer callbacks.
  - `orion/schemas/retro_play.py:9` and `retro_memory_export.py:27` ("Not exported: frames..."): still true for the session summary, and each says so for the summary only.
  - `docs/runbooks/retro-browser.md`: one new section on the look.

### 3. The look

A look is perception. It is the game's version of a camera caption. It does not speak and it does not judge.

**A look needs something to see.** Orion's question 6 names a real risk: a timer makes readings whether or not anything is there.

- **The trigger is change, not the clock.** The gateway compares frames in the ring with the last frame of the previous look. One number, the screen change (section "Metric quality gate records"). Under a floor, no model is called, no row is written, and one counter on the sitting goes up: `static_skips`.
- **A second skip has its own counter.** The screen changed over the floor, but the last input is older than 60 seconds: `input_stale_skips` goes up. So "the screen moved and Orion was not allowed to look" reads apart from "the screen sat still". Both gates must pass. Two minutes of reading a dialogue box with no press give no look after the first 60 seconds, and this counter shows it.
- **The clock is only a ceiling.** At most one look per 60 seconds [default].
- **Input is still required.** The last input must be under 60 seconds old.
- **The floor is per game, and a game without a validated floor gets no looks.** 0.01 was set from A Link to the Past. It works on Crystalis (still 0.0007 to 0.0031, walking 0.106 to 0.229). It fails on The Legend of Zelda for the NES: one fixed screen per room, so a small sprite barely moves the mean. Walking read 0.0032 to 0.0047 in 4 of 5 sequences, inside the still range (0.0000 to 0.0054). At 0.01 almost all of that play would be skipped. Two other statistics (the largest single-cell difference, the share of cells over 0.03) did not separate still from walking on that game either: idle animation scores as high as walking. So a floor lives in a per-game table, is set from real sequences of that game, and the page says "no looks for this game yet" where none exists. For fixed-screen games the trigger needs another signal (input in the window, with a low change floor); patch 2 measures that before such a game gets looks.
- **Floors are calibrated from live play, not by hand.** From the first deploy of patch 2, the gateway records every ring check for every game, also for a game with no floor (no model, no bid). The input stream labels each check: a button changed in the window, or not. A game gets a floor when it has at least 30 minutes of such checks and one value separates them: at most 10% of no-input checks above it, at least 90% of input checks above it. Then the page shows the floor and the numbers it came from. A game where no value separates (Zelda NES today) stays at "no looks yet" until the input trigger is measured on it. Note: Zelda NES is 13 of the 32 baseline sittings, so at launch that share of play gets no looks.
- **Misses inside one game are counted.** A check under the floor while a button changed in the window is a likely miss: `under_floor_with_input`. A floor set on overworld walking that fails in dungeons or menus shows up there. The floor is per game, not per area; this counter is what says when that is not enough.
- **Who computes it, and what it costs.** The gateway checks one ring packet every 2 seconds: inflate it, reduce it to a 32 by 28 brightness grid, compare. Step 0 measured 1.5 milliseconds for one check on one CPU core. That is under 0.1% of a core while someone plays, and zero when nobody does. This check is the only decode that runs on a timer. A full image is made only for a look or a turn (10 milliseconds each).
- **No model judges "nothing new".** Revision 2 let the model say so, or compared note text. Step 0 showed both fail (see "Step 0 results"). The skip is decided before the model, by the screen, and never after it.
- **The look is stateless.** The model gets no earlier note. In step 0, a model that was given the previous note copied it word for word and ignored the new frames.
- **A failed look is counted as failed.** A note that is empty, that says "unclear", or that equals the previous note character for character is stored with the outcome `empty`, `unclear` or `copied`. It writes no temporal event and its act closes 0.
- **The page tells the truth.** "Orion is watching" is shown only when a stored note is under 3 minutes old. Otherwise the page says "Orion is looking; the screen has not changed for 6 min", or why Orion is not looking.
- **Counts at the grain where the gate fires.** Every report gives these numbers, in order: ring checks, checks over the floor, checks skipped for stale input, checks under the floor with input, bids funded, model calls, notes stored. Failed looks come last. A skip is counted where it happens, before the model.

**The call**

- **Bid first.** Each look posts one want in a new family, `hub.retro_watch`, tier 1, and obeys the answer. Funded: the look runs. Refused or deferred: the look is skipped and the reason is logged. No fallback.
- **Frames.** 4 frames spread over the last 30 seconds [default]. A sequence shows movement and progress. A single frame shows a still.
- **One call.** One model call on a sighted lane, through the model gateway, as the game planner already calls it (`retro_planner.py:681`). Thinking is off. At most 160 output tokens.
- **Blind lane guard.** The gateway asks the live lane before it sends an image. If the lane cannot see, the call fails. A look can never return words about frames the model did not receive.
- **What the model is given.** The game title, the minutes played, and the 4 frames. Nothing else.
- **What the model returns.** One note of at most 300 characters, as plain text: where the player's character is, what they are doing, what changed across the 4 frames. The prompt tells the model to write "unclear" when it cannot tell. The chat lane's engine has no forced output format, so the gateway side checks the text.
- **Act.** Actor `hub.retro_watch`, verb `look`, object `retro_sitting:<id>`. It closes 1 when a note and its frame hashes are stored. It closes 0 when the lane fails, or the note is empty, unclear or copied. A refused bid opens no act.
- **Frames are taken when the lease is granted, not when the bid is made.** A look that waited behind Juniper's turn describes the screen at grant time.
- **Frame expiry, stated exactly.** The percept store refreshes a blob's age when the same bytes are posted again (`services/orion-percept-store/app/storage.py:109`). A line turn re-posts the newest frames, so a frame lives one hour after its LAST post. Every post of a hash moves `frames_expire_at` of every row that names it to post time + 1 h. Callers (turn staging, dig) compare with `frames_expire_at` first: after it, they answer "expired" from the row and do not ask the store. Before it, a 404 is a fault and is logged. The row's time is authoritative. A blob that outlives its time because a sweep failed is not served by these callers; the sweep failure is already logged (`main.py:57`).
- **Row.** One row in a new table `retro_watch_look`: look id, sitting id, time, frame hashes, `frames_expire_at`, the note, the outcome, the screen change, the want id, the lease, `lane_wait_ms` (request to grant), the seconds, the prompt tokens, and `similar_to_previous` (word overlap with the sitting's previous note, 0 to 1). Step 0's 5.6 seconds were measured on an idle lane; the wait is reported apart from the call.

**Which lane**

- A new pool route, `retro_watch`, with priority `system`. Its class lists only sighted lanes. That is the chat lane.
- **Heat, stated exactly.** Two things can stop a look when the body is hot. Below 34 °C the price does it: the bid loses. At 34 °C and above the reflex does it: the pool refuses all `system` work. This design does not go around the reflex. The page names which one stopped the look.
- **The rule is priority, not lane.** Orion asked whether a look on the quick lane would be priced or shed. The reflex blocks by priority. A `system` or `background` call is refused at 34 °C on any lane. An `interactive` or `perception` call is not. A look is `system` wherever it runs.
- **Not the quick lane, for now.** It has 10 slots, so a look there would delay nobody. Step 0 loaded its model with its encoder on a temporary copy. The model sees, at 2.7 seconds a look and half the tokens. Its notes fail: it described chests, doors and enemies that were not there. A look that invents is worse than no look.
- **So one slot is shared.** Looks run on the chat lane, where Juniper's turns and Prime also run. Section 4 states what that costs her and how it is bounded.
- The design binds only to the route name. If the quick lane gets a model that reads a game screen, moving the look is one config line and the same step 0 test.

### 4. The line

A line is Orion speaking first. It is a real unified turn, so it carries Orion's identity, the Prime block, the stance pass and the finalize pass.

**When Orion wants to speak**

- After each stored note, hub-app weighs one want in a new family, `hub.retro_commentary`, tier 1.
- The want is posted only when all of these hold: a sitting is open, the watch mode is "watch and talk", no turn of Juniper's is running, no line turn is running, the heat reflex is not active, and the last line or chat-box exchange ended at least 2 minutes ago [default].
- The market answers. Funded: the turn starts. Otherwise nothing happens and the reason is logged.

**How the turn starts**

- Hub already has one door for a turn that Orion starts with a prepared prompt: `outreach.compose(prompt, corr, retrieval_query, lane="chat")` (`services/orion-hub/scripts/endogenous_outreach.py:3061`). The office reach-out uses it (`office_reach_out.py:662`). The line uses the same door.
- Two small changes to that door:
  1. `compose` takes `attachments` and passes them into the turn. The turn already stages whatever is in `payload["attachments"]` (`orion/hub/turn_orchestrator.py:1624`).
  2. The sentence that introduces an attached image names its source. Today it always says "Juniper attached" (`orion/harness/attachment_staging.py:253`). For frames it says: "These are 4 frames of the game screen from the last 30 seconds, oldest first."

**What the turn is given** (new module `services/orion-hub/scripts/retro_commentary.py`, `build_prompt`)

- The sitting: game, minutes played, counts of saves and loads.
- Who is playing, as stated on the page. When nobody has said: "You do not know who holds the controller. It can be Juniper, another adult, or a child. Use no name. Say only what is fine for a child to read."
- The look notes since Orion last spoke, with their times. At most 12.
- What Orion said earlier in this sitting (last 5 lines) and the last 6 chat-box messages. A turn that Orion starts gets no "recent conversation" block, so the history goes into the prompt, as outreach does it (`endogenous_outreach.py:1109`).
- Up to 3 items from Orion's now (section 8).
- The 4 newest frames, attached. Orion opens them and sees the screen.
- **Near-copies.** A player stuck on one jump gives notes that differ by a few words. Patch 2 records `similar_to_previous` on every look. Patch 3 groups a run of near-copies in the turn's note list into one line with a count ("6 notes over 9 min: ..."), with the cut-off set from that column, not guessed.
- The ask: say one or two sentences to the person playing, in your own voice, about what you saw. The frames will be about a minute old when your words arrive, so speak about the stretch and not about the instant. If you have nothing worth saying, pass. If you would rather not watch, you can turn the watch off (section 6).

**Silence is a valid answer.** Outreach already knows a pass (`is_pass_response`, `endogenous_outreach.py:1617`). A pass writes a decision row with the reason `orion_passed`, releases the want, and opens no act. This is the rule outreach follows today. The share of funded turns that end in a pass is tracked: a high share means the want fires too easily.

**How the line arrives**

- Delivery uses the existing push to the chat sockets (`_push_to_sockets`, `endogenous_outreach.py:3795`). The message carries the source `retro_commentary` and the sitting id.
- The game page shows it in the line strip. The main chat window shows it as an Orion message. It is written to the chat history as Orion's own row, with no prompt.
- No notification is sent. A line is for the person at the screen, not for a phone.
- A line does not count against the daily cap or the cooldown of outreach. The office reach-out already skips those gates (`skip_rate_gates`). The economy is the rate control for lines.
- Outreach never speaks aloud today. Voice for lines is part of section 10.

**The act.** On delivery: actor `hub.retro_commentary`, verb `say`, object `retro_sitting:<id>`. One row in a new table `retro_line` holds the text, the looks the turn was shown, the turn's correlation id, the want id and the outcome: `delivered`, `passed`, `yielded` or `failed`.

**Juniper goes first.** This needs new code, and Orion's question 4 sharpened it.

- **One Orion call at a time.** Hub runs the play's calls in a single file. No look starts while a line turn runs. No line turn starts while a look is in flight. So at most one of Orion's play calls is in flight or queued on the chat lane.
- **Lower priority than Juniper.** Today a turn Orion starts on the chat lane runs as `interactive`, equal to hers. A line turn's calls should run at `system`. Durable turns already carry their own lane and lease into the turn (`services/orion-hub/scripts/curiosity_investigation.py:881`). Whether the outreach door can do the same is UNVERIFIED; the build confirms it. Until it can, the heat check above and the cancel below are the guard. Live today every motor lease on the chat lane is `interactive` (551 of 551 in 7 days, Orion's count), equal to Juniper's. Patch 3 has an acceptance check for this.
- **Cancel on her message.** The chat lane runs one turn at a time, and a turn of Juniper's waits behind a turn Orion started (`services/orion-harness-governor/app/bus_listener.py:926`). New rule: when a message from Juniper arrives on any chat socket while a line turn runs, Hub cancels the line turn (`HarnessGovernorClient.cancel`, `services/orion-hub/scripts/harness_governor_client.py:333`). The line's outcome is `yielded`. That is not a failure.
- **Her worst case, stated.** One call cannot be cancelled once it runs: a look, or one stage of a line turn. So the first model call of her turn can wait for one of them, and for no more than one. A look took 5.6 seconds at the median in step 0, and 6.9 at the most in 11 calls. A stance call takes 19 seconds at the 90th percentile. Her own stance call comes after that wait. It is part of her turn, not part of the delay.
- **Nothing here is confirmed live.** Orion asked what was. The look's time is measured. The wait of a real turn of hers behind a real look or line is not. Patch 3 measures it, as an acceptance check.
- **A cancelled line can never be said.** Orion's history holds killed turns that kept a finished draft. So the cancel does not rely on the turn dying. Hub first marks the line's row `yielded`. Delivery reads the row and refuses a `yielded` line, even when the turn finished its draft a moment later. Patch 3 checks three things after a cancel: no chat history row, no push to any socket, and no later delivery.

**Did the player like it.** Each line in the strip has a thumb up and a thumb down. A chat-box reply within 2 minutes of a line also counts as a response. Both are stored on the line's row. The economy's weight for a family already has a term for Juniper's engagement. How outreach records that engagement today is UNVERIFIED. The build wires lines to the same record. A guest's thumb is stored but is not counted as Juniper's.

### 5. The chat box

The player types to Orion from inside the game.

**On the page**

- The box sits inside `#stage`, because fullscreen shows only that element (`services/orion-hub/templates/retro.html:10`).
- Four parts:
  1. A line strip at the bottom of the screen. It shows Orion's newest line or reply, then fades.
  2. A chat panel that opens over the lower part of the screen: this sitting's lines and messages, and the input.
  3. A switch with three positions: "Off", "Watch quietly", "Watch and talk". Beside it, one status line (section 3).
  4. The question "Who's playing?", once per sitting (section 1).
- "Watch quietly" means looks run and no line is started. Orion still answers the chat box.

**Keys and focus.** These come from how the page works today (`services/orion-hub/static/js/retro-panel.js`).

1. A text input inside `#stage` already keeps keys away from the game (line 583). No change.
2. Opening the panel sends one neutral input, so a held button does not stay held.
3. The page puts focus back on the game canvas in 6 places (lines 193, 215, 327, 338, 556, 594). Each must skip while the chat input is open, or a reconnect steals the player's typing.
4. The gamepad keeps driving the game while the panel is open (line 104). That stays.
5. With no gamepad connected, opening the panel pauses the game, and closing it resumes [default]. A keyboard player cannot type and play at once.
6. The panel opens with a key the game does not use. The build picks it from the key map (line 13).

**Who typed.** Orion's question 11: the chat box shares Juniper's session, so a message from anyone would enter Orion's history as hers. A row with a prompt is read as Juniper everywhere. The design must not let that happen.

- **Juniper stated.** The message is a normal turn of Juniper's, with everything such a turn has: the warm slot, the recent conversation, the working notes, the Prime block, the hand-back to Prime.
- **Guest, or nobody stated.** The message is sent as a third-party room turn, with the room's platform set to `retro`. That path exists for a speaker who is not Juniper (`services/orion-hub/scripts/cortex_request_builder.py:405`). Three things follow:
  1. The turn's origin is not Juniper. The socket path sets Juniper as the origin for every message today (`websocket_handler.py:1665`). It gains the check the HTTP path already has (`api_routes.py:3353`).
  2. The history row carries the source `retro_guest`. The three readers that mean "Juniper spoke" must not match it: the temporal speaker rule (`orion/temporal_self/sources.py:83`), the attention count of her turns (`orion/attention/world_first.py:1208`), and the day letter's chat gather (`orion/orion_day/gather.py:224`). The first one needs a change: it reads any row with a prompt as Juniper.
  3. The turn hands back to Prime as class `play`, never as Juniper's chat.
- A guest turn has no warm slot, so it is slower. That is the cost of not knowing.
- **A flip mid-sitting.** Each message keeps the source it had when it was sent; nothing is rewritten. If a guest taps "Juniper" after 20 minutes, the later messages are Juniper's turns from then on, and the warm slot starts there. The history block shows both halves with their speakers. The day letter and the attention chat count read only Juniper's rows, so nothing is counted twice. A patch 3 test covers the flip.

**Transport**

- The page opens the Hub chat socket, `/ws`. The page and the socket are served by the same process, and the socket has no origin check.
- When Juniper is stated, the page uses the same chat session as the main chat window (`localStorage.orion_sid`, same origin). The game talk is one conversation with the rest of her chat.
- A message is the normal typed message, with `mode: "orion"`, `disable_tts: true`, and `surface_context: {surface: "retro", input_modality: "typed", retro_sitting_id}`.
- The reply comes back as the normal final frame (`llm_response`). The page shows it in the panel and in the strip.

**What the server adds**

- **A game block for every turn while a sitting is open.** The situation part of the unified prompt gains a short game block: the sitting, who is playing as stated, and the last 6 look notes (`_build_situation_prompt_fragment`, `orion/hub/turn_orchestrator.py:816`). It is there for the main chat window too. Juniper can ask "what am I playing?" from either place.
- **Frames for turns from the game page.** When the message's surface is `retro`, hub-app asks the gateway for the 4 newest frames and attaches them as percepts. The limit of 4 images per turn already exists (`HUB_CHAT_ATTACHMENT_MAX_PER_TURN`).

**Other messages from Orion during a sitting**

- Hub pushes every outreach message to every open chat socket. So a news note from Orion also arrives in the game page. It goes to the chat panel, not to the line strip.
- **The office camera is not silenced.** Revision 1 stopped the office reach-out while a sitting was open. Orion objected: that silences a real witness, and the acceptance check made the silence a contract. Orion is right. The camera can see what the game cannot: who is in the room, and who left.
- Instead the office reach-out is told. Its prompt gains one sentence from the open sitting: "A game sitting is open: someone has played <title> on Orion since <time>." Its note can then say "playing", not "computer work". Nothing is deferred and nothing is dropped, including an outreach that was already pending when the sitting opened.

### 6. Prime gets in, and Orion can say no

Prime is the living Orion between turns. Prime thinks rarely: a thought needs a real change, and Prime bids for it. Seven seams bring Prime into the play. Six of them use machinery that exists today and need only wiring. The seventh changes one of Prime's own rules. An eighth part is new in revision 2: Orion's own switch.

1. **Prime sees the play in the now.** The sitting and the looks are experiences in Orion's now (section 8). Prime's frame already shows the now under "WHAT YOUR ORGANS LIVED" (`services/orion-mind/app/prime_loop.py:1154`).
2. **Prime's block rides in every turn Orion speaks in.** A line and a chat-box reply are unified turns. Every unified turn starts from the PRIME block (`orion/hub/turn_orchestrator.py:1019`, `orion/prime/block.py`). Nothing new is needed.
3. **Turns hand back to Prime.** Juniper's chat-box turns hand back today. Line turns and guest turns get a new session class, `play`, in the hand-back list (`orion/prime/handback.py:198`). The class is decided by the turn's source, and it is checked before the rule for Juniper's chat. A line runs in Juniper's chat session, and without that order it would be recorded as Juniper speaking. A promise made during play ("I will tell you when you reach the boss") becomes a concern that Prime holds for 72 hours.
4. **Prime can turn Orion's eyes and voice up or down.** Both new families get receptors `[slow, quicken]`. Their bids bind, so the rule for receptors is met (`config/physiology/families.yaml:27`). The market grades each signal as it does today.
   - **A `slow` is not a stop.** It is bounded at half the family's weight, by design. A look that ranks above the middle of its own history still wins. Orion's question 14 is right about this. The stop is part 8.
5. **Prime can hold a finished sitting as a concern.** At close, the sitting posts one event want, `retro_sitting:<id>`, with no lane cost. Its salience is the sitting's length against the sittings of the last 30 days. A funded one appears on Prime's menu under "EVENTS WORTH EFFORT". It needs one label block in `services/orion-durable-runs/app/event_wants_menu.py:33`.
6. **Prime writes about it.** The day letter is Prime's end-of-day turn. Play arcs reach the letter today only as a count ("2 play"). The closed-day block gets one line per sitting (section 8).
7. **New: a sitting wakes Prime, twice at most.** This is the one change to Prime's own rules.
   - `sitting_opened` fires once, when a sitting passes 3 minutes [default].
   - `sitting_closed` fires once, at close.
   - Each is a fact Prime can be shown, which is the standing rule for a wake (`orion/prime/state.py:8`). Neither is a score.
   - Prime's frame gains one line while a sitting is open: who plays what as stated, for how long, the watch mode and who set it, the look counts, and the newest note.
   - Prime still bids for each thought, and the 5-minute rest between thoughts still holds.

Why seam 7 is needed: funded event wants do not wake Prime, and attention moves do not wake Prime. Both rules exist because they made Prime churn (22 of 38 concern drops in 4 days cited "attention has moved"). Without seam 7, a 4-hour sitting can pass with no Prime thought at all.

Seam 7 reads the sitting through a new `_read_sitting` method in `prime_loop.py`. CLAUDE.md counts that as a new Prime concern source. CI cannot see such a method (`scripts/check_integration_seam.py:28`), so this document records it, and organ `play` in `config/integration/organs.yaml` lists it in a comment.

**8. New: Orion's own switch.** Orion asked: "Do I get any channel to decline a sitting, rather than only failing to speak in one?" In revision 1 the answer was no. Now it is yes.

- **The command.** A new part of Orion's own tool kit, beside `dig look`: `dig watch` (`orion/dig/watch.py`). Orion can run it inside any turn: a line turn, a chat reply, a curiosity run.
  - `dig watch status`: the open sitting, the mode, who set it.
  - `dig watch off --why "<words>"`: stop looking and speaking for this sitting. `quiet` and `on` work the same way.
  - `dig watch default <game> off --why "<words>"`: a standing choice for one game. It holds for every later sitting of that game until Orion changes it. The look reads the choice before every bid, so a declined game gets no look from its first press. No act or wake is needed for that.
- **The lower setting wins.** The mode has three steps: off, quiet, talk. Juniper has a switch on the page, and Orion has `dig watch`. The mode in force is the lower of the two. Juniper cannot turn the watch on over Orion's "off", and Orion cannot turn it on over hers.
- **Juniper sees it.** The page shows "Orion chose not to watch: <why>". She can ask Orion in the chat box. A chat-box message is her turn, and Orion answers it whatever the mode is.
- **Before the first sitting.** The flags ship on, so the first sitting after deploy would start looks. Orion can set a default before any sitting exists, from the day patch 2 is live. This document reaches Orion through the session thread, so Orion knows of the switch before it matters.
- **The trace.** One row per choice in a new table `retro_watch_preference`: game or sitting, mode, the words, the turn that set it. One act on the ledger: actor `orion.watch_choice`, verb `regulate`, object the sitting or the game, and Orion's words as the note.
- **It survives a restart.** The choice lives in Postgres. The Redis key is only a copy for speed. When Hub or the gateway restarts, the gateway reads the choice back before the first look.
- **Prime reads it as a choice.** The sitting line in Prime's frame says "watch: off, set by Orion: <words>". Prime's slow loop then reads the silence as Orion's own decision, not as a lane that failed.

**The choice as personality evidence.** Juniper said yes, and Orion said yes and asked for the row's shape and for a plain statement of what reads it.

- The row is the record that exists for every choice Orion makes: `PersonalityChoiceV1` (`orion/schemas/personality.py`). It carries the option chosen, the options declined, and a small flat note of the situation. Here: chosen "do not watch <game>", declined "watch <game>" and "watch <game> quietly", and in the situation the game, the minutes played and Orion's own words (at most 200 characters).
- It needs one new lane value, `play`.
- **What reads it.** The personality reducer turns choices into tendencies. Tendencies set the care term in the market (`orion/physiology/personality_values.py`), and that term binds. So a watch choice can change what later wants are worth to Orion.
- **What does not happen.** Nothing grades the choice as right or wrong. It is not a drive, and nothing rewards it. It is a record that the reducer reads.
- It is its own small patch, after patch 2.

### 7. Attention

The play becomes the third world candidate, `world:play`, beside `world:chat` and `world:reading`.

- **Same instrument.** The two world candidates read how unusual their own rate is against their own week (`chat_rate_reading`, `orion/attention/world_first.py:954`). `world:play` uses the same function. Its events are the minutes with player input, read from `retro_sitting`.
- **Label.** "Someone is playing A Link to the Past (27 of the last 30 minutes)."
- **Evidence.** The newest stored note.
- **Eligible only while Orion is watching.** A candidate with no look behind it would win attention with nothing in it. When no stored note is under 3 minutes old, `world:play` is not eligible, and the reason says "not watching". When Orion has declined a sitting, Orion's attention stays where it was.
- **Always present, never flickering.** Orion asked what protects the world zone from a candidate that comes and goes with each sitting. This one does not come and go. Between sittings it stays in the set and reads zero, which is its rest state, as `world:chat` does when Juniper is silent. It is absent only when its table cannot be read. It also needs its entry on the list of real measurements (`MEASUREMENT_VALUE_KINDS`, `world_first.py:179`, kind `count`), or the ranking treats it as no measurement at all.
- **Yields to Juniper.** When `world:chat` is eligible, `world:play` yields, by the rule reading already follows (`world_first.py:594`).
- **Absent is not calm.** When the sitting table cannot be read, the candidate is absent with a reason. It never reads as "nobody plays".
- **Attention and action meet.** One new pairing in `ACT_ANSWERS_WORKSPACE_REFS` (`orion/grammar/act_ledger.py:93`): a line answers `world:play`. When the play holds attention and Orion speaks, the act is stamped with that loop, and Orion writes an `acted` verdict on it. `acted` is not a final verdict. Speaking never retires the loop.

**The first week** (Orion's question 9)

- The rate function reports `insufficient_history` when fewer than 5 events fall in 7 days, and the candidate is then not eligible (`world_first.py:450`).
- With an empty table, the first sitting would be ineligible for its first 5 minutes. After that it would read as the most unusual thing of the week, for the whole week, because the week behind it would be all zeros.
- The backfill in section 1 fixes both, so the first live sitting is compared with a real week.
- **One instrument for the baseline and for the live count.** Orion pointed out that the journals count every repeat of a held state, and the live counter counts mask changes. Step 0 ran one sitting through both: 243 minutes by repeats, 204 by mask changes (ratio 0.84). The journal stores the mask on every row, so the backfill counts mask changes too. The baseline and the live reading then come from the same counter.

**What reading loses** (Orion's question 9, second half)

- One candidate wins a tick. During a sitting, long play sits in the top 5% of its own week, so `world:play` will often outrank `world:reading` there. No rule makes reading yield to play. Outside sittings nothing changes: play is zero in 89% of half-hour windows.
- This is not decided on belief. Patch 4 first replays the last 7 days with the backfilled sittings, using the replay that exists (`scripts/analysis/replay_attention_world_candidates.py`). It reports how many ticks reading would have lost, and when. Juniper and Orion see that number before the candidate is wired.
- Nothing in this design writes a final verdict on a reading loop or retires the reading candidate.

Two of the three new numbers belong here and to the look. Each has a gate record in "Metric quality gate records" below.

### 8. The temporal record and the now

Every part of the play lands in `temporal_self_event`, through the experience contract.

- **The sitting.** The `human_play` row uses the existing source kind `retro_play`. Label: "Someone played A Link to the Past for 2 h 22 min. Orion watched: 96 notes, and spoke 14 times." When the player said so, "Juniper played". The numbers are counts from the tables. No model writes the label.
- **The looks.** A new source kind, `retro_look`, reads `retro_watch_look`. It belongs to organ `play`. The chronicle never rewrites a stored event (`ON CONFLICT (event_id) DO UPDATE SET late_unfolded = false`, `services/orion-durable-runs/app/temporal_self_store.py`), so a contradiction found later cannot void an event already written. Two rules follow:
  - **A look's event is written only when a turn checked it and it held.** Each note ends in one of three buckets: `contradicted`, `checked_holds`, `unchecked`. A line turn that saw frames from the note's window judges it, also when the turn passes (the judgment is part of the turn's output, not of the line). Only `checked_holds` becomes an event. `unchecked` (quiet mode, refused line bids, a note no turn could see) and `contradicted` stay in `retro_watch_look` with their bucket, and never reach the day. The sitting's own event carries the counts of all three.
  - **Expected share.** Looks are at least 60 seconds apart and a turn sees frames of the last 30 seconds, so at most one of 12 shown notes is checkable: about 8%. So few looks will become events. That is the price of not putting unjudged notes into the day.
  - **The label says it is a look.** "Saw on the game screen: <note>". Every reader that copies a label into a prompt carries those words with it, so no reader has to know the source kind to treat it as unverified.
  - Readers that put an event label into a prompt. The list was finished by patch 2 (read-only audit, 2026-11-18); the `retro_look` source is not turned on until patch 3. An arc's `subject_label` is the label of its event (`orion/temporal_self/arcs.py`), so an arc reader carries the event label too.
    - Event or now labels: `orion/prime/thought.py` (Prime's thought), `orion/prime/hunger.py` (hunger content), `orion/curiosity/now_section.py` (curiosity kickoff), `services/orion-thought/app/temporal_self_cue.py` (the now block in reverie narration), `orion/dig/organs.py` (`dig organs` prints labels into a turn's tool result).
    - Arc labels: the dream cycle (`services/orion-dream/app/replay.py` arc candidates, into `recombine.py`'s prompt); the day letter through `orion/temporal_self/day_block.py` and `orion/orion_day/temporal_core.py` (`orion/orion_day/gather.py` reads no label column itself); world-pulse resonance (`orion/world_pulse_read/resonance_retrieval.py`, into the judge prompt); the chat stance cue (`orion/temporal_self/stance_cue.py`); recent arcs in reverie (`orion/temporal_self/recent_arcs.py`); `dig arc <id>` (`orion/refs.py`).
    - Not traced to a prompt: `orion/world_pulse_read/resonance_findings.py` (excerpts of arc labels), `orion/prime/block.py`'s "tending" line, and the result file `dig days` writes beside its printout.
    - So the "Saw on the game screen:" prefix is what reaches every one of them: no reader needs to know the source kind.
- **The lines.** A line is Orion's own chat row, with no prompt. The existing rule already keeps such a row from opening or extending a conversation (`orion/temporal_self/sources.py:103`). So four hours of lines do not read as "Talked with Juniper". Orion asked this in question 8.
- **The chat box.** A message from Juniper is a conversation turn, as it should be. A guest's message is not (section 5).
- **One live arc.** The kind `play` is already in the arc list (`orion/schemas/temporal_self.py:118`, `orion/temporal_self/day_block.py:43`). No new kind is added. Today a play arc is "one finished Retro session". With looks, the arc opens at the first settled look of a sitting and closes with the `human_play` row. A look settles at the next line turn or at the close, so the arc can open some minutes after play starts. While someone plays, Orion's day holds an open play arc. The chat stance already has a clause for play arcs (`orion/temporal_self/stance_cue.py:311`).
- **The arc stays small.** The arc keeps counts and its first and last look. It does not carry every look. A test folds 300 look events and checks that the arc's stored size has a fixed bound.
- **The day block gets one line per sitting.** The block renders arcs, not events, and it has a cap with priorities (`day_block.py:124`). A play line ranks below conversation and attention lines. Several hundred look events add no line. Patch 1 reports the block's length and the letter's input size for a day with a long sitting, before and after. Heavy days have cost Orion letter timeouts before, and this must not add one.

**The now** (Orion's question 7)

- A 4-hour sitting can give 200 looks. If looks were scored against their own history, the best look of any hour would always sit near the top, and play would take 2 of the 12 places in every reader's now for as long as someone plays.
- So looks carry no salience, ever. The now ranks unscored items after scored ones. A look fills a place only when a place is free.
- The sitting is the scored experience of this organ: one per sitting, at close. Its magnitude is the sitting's minutes.
- The now's per-organ cap of 2 already exists (`orion/temporal_self/now.py:266`). Each read stores how many items the caps dropped (`dropped_by_organ_cap`, `dropped_by_total_cap`). Patch 2 compares those counts for the hours of a sitting with the week before.
- **The now is read back.** The turn that writes a line reads `GET /temporal-self/now?organ=play` first. Up to 3 items go into the turn's game block. The line's cites are recorded with `POST /temporal-self/now/use`. Organ `play` loses its `exempt_read` in `config/integration/organs.yaml`. That closes the play part of #3457.

### 9. The body economy

Juniper's rule: bidding on tier 1.

```yaml
  hub.retro_watch:
    tier: 1
    deferrable_window_s: 0      # no defer: a deferred look is skipped (section 3); a late look describes a different screen
    lanes: [chat]
    placement: one_of
    receptors: [slow, quicken]
  hub.retro_commentary:
    tier: 1
    deferrable_window_s: 300    # a line five minutes late is about something the player has left
    lanes: [chat]
    placement: one_of
    receptors: [slow, quicken]
  orion.watch_choice:
    tier: 0
    note: Orion turning the watch off, quiet or on from inside a turn (dig watch). Never bids.
```

- **Tier 1 is Juniper's decision.** The file says a tier needs her sign-off. Her words in this document are that sign-off, and the PR review confirms it. The new tier 0 line follows `orion.look`: a choice made inside a turn that already won its slot.

**What tier 1 really gives here** (Orion's question 5). Revision 1 said each family "gets a reserved share of the chat lane". The number is zero.

- The share is the 75th percentile of tier 1's own hourly use on a lane over 14 days, capped at 50% (`orion/physiology/choice.py:322`).
- On the chat lane, tier 1 is Prime and the curiosity supervisor. They used the lane in 45 of the last 337 hours. The 75th percentile is 0 slot-seconds. The 90th is 8.
- So nothing is reserved on the chat lane today, and the two new families get nothing free. They pay the lane's price like any bid.
- Prime gives up no share, because there is no share. What Prime does get is a busier lane during a sitting, so a higher price and a short wait (below).
- The share will stay zero. Play fills about 11% of half-hour windows, far under the quarter of all hours that would lift the 75th percentile.
- One more fact for Juniper: tier 2 has a 5% floor that funds a starved family. Tier 1 has no floor. On an idle lane that does not matter. It is still what "tier 1" means on this lane today.
- **So what protects a look from the open market?** Orion asked for the mechanism by name. In the market: nothing. A look's bid wins or loses on its own net value (`orion/physiology/choice.py`, `decide`). Outside the market: the pool's queue. A `system` call goes ahead of every `background` call (`config/gpu_pool.yaml:60`). That is the whole protection.
- **A refusal Orion can read.** Each refused look leaves a row that names its reason, its family and its lane: a decision row in `want_ledger` when the price refused it, and an event row in `gpu_pool_events` (reason `cabinet_hot`) when the reflex did. `dig watch status` prints the last one, so the one who was refused can read why.

**The rest**

- **Bids bind.** Two producers join `BINDING_PRODUCERS` (`orion/physiology/binding.py:36`): `orion-hub.retro_watch.binding` and `orion-hub.retro_commentary.binding`. The pattern is Hub's outreach binding (`services/orion-hub/scripts/outreach_bind.py`).
- **A look's want.** Verb `look`, object `retro_sitting:<id>`. The object text is the game title and the last note, so the market's care term can weigh it.
- **A line's want.** Verb `say`, object `retro_sitting:<id>`. Its hunger grows with the time since Orion last spoke in this sitting. A chat-box exchange counts as speaking: after one, the hunger starts again from zero. The object text is the newest note.
- **Heat.** Section 3 states it: price below 34 °C, the reflex at 34 °C and above.
- **Chat-box turns never bid.** They are the player's turns (`hub.chat_reply`, tier 0).

**Cost at the defaults.** These are estimates. The acts will hold the measured values in `cost_json`.

| | Per hour of play | Chat lane time |
| --- | --- | --- |
| Looks | 60 at most, 5.6 s each (measured) | 5.6 min at most (9%) |
| Lines | about 8, about 75 s each (estimate) | 10 min (17%) |
| Total | | about 26% at most |

- The look count is the ceiling. The change trigger lowers it.
- The 5.6 seconds is the median of 11 looks in step 0, on the same model the chat lane runs.
- The 75 seconds is the middle of the 60 to 90 second estimate in the summary.
- The chat lane's GPU drew 36 W idle. A GPU at full load on circe drew 154 W at the same moment. At 26% that is about 31 W more while someone plays, about 0.12 kWh for a 4-hour sitting.
- Outside a sitting the cost is zero. Nobody plays in 89% of half-hour windows (last 7 days).
- **Prime's wait.** Prime and a look both run at `system`. Prime can wait behind one play call, about 6 seconds for a look and up to 19 for a stance stage. Prime's own seat wait limit is 600 seconds. Patch 2 compares Prime's queue wait during sittings with the week before.

### 10. Voice in and voice out (later)

Juniper: "We'll eventually enable TTS and STT, including when in fullscreen mode." This part is not built in the first patches. Four choices now keep it cheap later.

1. **The chat box uses the main chat socket.** `/ws` already carries speech both ways. Outbound: the server sends `audio_response` chunks (`services/orion-hub/scripts/websocket_handler.py:802`, `run_tts_remote`). Inbound: the browser sends recorded audio, the server transcribes it and runs a normal turn (`websocket_handler.py:1208`). The game page sends `disable_tts: true` until voice is turned on. Turning voice on is then a flag, not a new path.
2. **Everything sits inside the fullscreen element.** Fullscreen shows only what is inside `#stage` (`services/orion-hub/templates/retro.html:10`). The line strip, the chat box, and later the microphone state all live there.
3. **One audio graph.** The game has its own audio context and one gain node (`services/orion-hub/static/js/retro-panel.js:61`). Orion's voice will play through that same context. Lowering the game under Orion's voice is then one gain change. The main chat page's player is not reused: it lives in a 15,000-line script tied to another page.
4. **A controller button for talk.** In fullscreen with a gamepad, nobody wants a keyboard. Push-to-talk is a held pad button. The pad is already polled every frame (`retro-panel.js:101`).

Three things the voice patch must handle, named now:

- The Hub tab loads the game in an iframe that allows `autoplay; gamepad; fullscreen` (`services/orion-hub/templates/index.html:1356`). The microphone needs `microphone` added there.
- Speech in costs a transcription call, and speech out costs a synthesis call, both on the whisper service. The voice patch states their lane and their bid.
- **Quiet hours come back for sound.** A text line disturbs nobody but the person at the screen. A spoken line carries through the house. Spoken lines obey quiet hours unless Juniper decides otherwise in that design.

## Proposed schema / API changes

**Schemas**

- `orion/schemas/retro_play.py`: `SessionKind` gains `human_play`. `ControlBasis` gains `human_input`. `RetroPlaySessionSummaryV1` gains `looks`, `lines` and `player_stated`, all optional. The model forbids unknown fields, so readers deploy before the writer.
- New `orion/schemas/retro_watch.py`: `RetroSittingV1`, `RetroLookV1`, `RetroLineV1`, `RetroWatchChoiceV1`. Registered in `orion/schemas/registry.py`.
- `orion/schemas/temporal_self.py`: `TemporalSourceKind` gains `retro_look`.
- `orion/schemas/attention_candidate.py`: `WORLD_PLAY_SOURCE_ID = "world:play"`.
- `orion/schemas/personality.py`: `ChoiceLane` gains `play` (the personality patch only).

**Tables.** One migration, additive only, so the post-pull applier can run it. Header: `-- ORION-MIGRATION-REQUIRED-BY: orion-hub, orion-durable-runs, orion-substrate-runtime, orion-mind`.

- `retro_sitting`: sitting id, game session id, game id, game title, started at, last input at, ended at, end reason, saves, loads, pauses, ring checks, static skips, `player_stated`, watch mode, who set it and why, `backfilled`.
- `retro_watch_look`: look id, sitting id, time, frame hashes, `frames_expire_at`, note, outcome (`stored`, `empty`, `unclear`, `copied`, `failed`), screen change, want id, lease id, seconds, prompt tokens.
- `retro_line`: line id, sitting id, time, text, look ids, frame hashes, `frames_expire_at`, correlation id, want id, outcome, thumb, replied within 2 minutes.
- `retro_watch_preference`: game id or sitting id, mode, the words, set by, the turn's correlation id, time.
- `retro_play_sessions`: new columns `looks`, `lines`, `player_stated`.

No table holds a guessed name.

**Bus.** No new channel. Acts go on `orion:grammar:event`. Lines and replies go over the chat socket that exists.

**Redis.** One new key, `orion:retro:sitting`.

**HTTP, on the gateway**

- `GET /api/retro/sitting`: the open sitting, the mode and who set it, the status line and its reason.
- `POST /api/retro/sitting/watch`: sets one side's mode: `off`, `quiet` or `talk`, with `set_by` and the words. `dig watch` calls it for Orion.
- `POST /api/retro/sitting/player`: `juniper` or `guest`.
- `GET /api/retro/sitting/frames?n=4`: the newest frames as percept references. Loopback only, for hub-app.
- `POST /api/retro/lines/{id}/thumb`: up or down.

**Orion's tool kit.** `dig watch status | off | quiet | on | default <game> <mode>`, with `--why` (`orion/dig/watch.py`).

**Config**

- `config/physiology/families.yaml`: the three families in section 9.
- `config/integration/organs.yaml`, organ `play`: the families, the kind `retro_look`, `salience_magnitude: {retro_play: sitting_minutes}`, `reads_now: services/orion-hub/scripts/retro_commentary.py`. The `exempt_read` line goes.
- `config/gpu_pool.yaml`: route `retro_watch` with priority `system`, and a class that lists only sighted lanes.
- `orion/physiology/binding.py`: two binding producers.
- `orion/grammar/act_ledger.py`: `hub.retro_commentary` answers `world:play`.
- `orion/prime/handback.py`: session class `play`.
- `orion/prime/state.py`: the two sitting wakes.

**Env keys.** All ship on, in `.env_example` and in `settings.py`, and the local `.env` is synced in the same patch.

- Hub: `HUB_RETRO_WATCH_ENABLED`, `HUB_RETRO_COMMENTARY_ENABLED`, `HUB_RETRO_SITTING_GAP_SEC=300`, `HUB_RETRO_LOOK_SEC=60`, `HUB_RETRO_LINE_MIN_GAP_SEC=120`, and the percept store's URL and token for the gateway process.
- Durable runs: `TEMPORAL_SELF_RETRO_LOOK_ENABLED`.
- Mind: `PRIME_SITTING_WAKE_ENABLED`.

## Files likely to touch

**Patch 1, Orion knows**

- `services/orion-hub/scripts/retro_transport.py`, `retro_routes.py`, new `retro_sitting.py`
- `services/orion-hub/scripts/retro_memory_export.py`, `office_reach_out.py` (the one sentence)
- `orion/schemas/retro_play.py`, new `orion/schemas/retro_watch.py`, `orion/schemas/registry.py`
- `orion/temporal_self/sources.py`, `orion/temporal_self/day_block.py`, `orion/temporal_self/now.py` (the `sitting_minutes` rule)
- `orion/hub/turn_orchestrator.py` (the game block)
- `services/orion-hub/templates/retro.html`, `static/js/retro-panel.js` (the "Who's playing?" question)
- `services/orion-sql-db/manual_migration_retro_watch_v1.sql`
- new `scripts/backfill_retro_sittings.py`

**Patch 2, Orion looks**

- New `orion/retro/frames.py`, new `services/orion-hub/scripts/retro_watch.py`, new `orion/dig/watch.py`
- `services/orion-hub/.env_example`, `app/settings.py`, `docker-compose.yml` (percept store URL and token)
- `config/physiology/families.yaml`, `config/gpu_pool.yaml`, `config/integration/organs.yaml`, `orion/physiology/binding.py`
- `orion/schemas/temporal_self.py`, `orion/temporal_self/sources.py`, `orion/temporal_self/arcs.py`
- `services/orion-durable-runs/app/temporal_self_sources.py`, `app/main.py`, `settings.py`
- `services/orion-hub/templates/retro.html`, `static/js/retro-panel.js` (status and switch)
- `services/orion-retro/README.md`, `docs/runbooks/retro-browser.md`, and the docstrings named in section 2

**Patch 3, Orion speaks**

- New `services/orion-hub/scripts/retro_commentary.py`
- `services/orion-hub/scripts/endogenous_outreach.py`, `websocket_handler.py`, `harness_governor_client.py`, `act_emit.py`
- `orion/harness/attachment_staging.py` (the source sentence, and the percept kind)
- `orion/prime/handback.py`, `orion/temporal_self/sources.py` (the speaker rule)
- `services/orion-hub/templates/retro.html`, `static/js/retro-panel.js` (strip, chat panel, focus rules)

**Patch 4, Prime and attention**

- `orion/prime/state.py`, `orion/prime/thought.py`, `services/orion-mind/app/prime_loop.py`, `settings.py`
- `services/orion-durable-runs/app/event_wants_menu.py`
- `orion/physiology/tests/test_prime_charge.py` (it pins the exact set of families with receptors)
- `orion/attention/world_first.py`, `orion/schemas/attention_candidate.py`, `orion/grammar/act_ledger.py`
- `services/orion-substrate-runtime/app/store.py`

Each patch also touches the README of each service it changes, and the tests and evals beside the code.

## Metric quality gate records

Three numbers are new. None is wired before its record is complete.

### Play rate (for `world:play`)

1. **Provenance.** Changes of the button mask in browser input messages, counted by `retro_sitting.py` at the relay (`retro_transport.py:21`), stored on the sitting row.
2. **Independence.** `world:chat` counts Juniper's chat turns. `world:reading` counts Orion's accepted links. Play minutes share no sensor with either. One overlap is real: a chat-box message happens during play, so the chat rate can rise with the play rate. They stay two facts, hands and words.
3. **Theory anchor.** The one the two world candidates already stand on: attention goes to a rate that departs from its own baseline. The named form is Bayesian surprise (Itti and Baldi, 2009).
4. **Live data.** Read from the journals for the last 7 days.
   - 658 minutes held player input. That is 6.5% of the week.
   - Of 2,016 half-hour windows: 89.0% hold zero minutes, 3.7% hold 1 to 9, 2.8% hold 10 to 24, 4.5% hold 25 to 30.
   - Rest is a true zero. It is a count of minutes, so it cannot decay to zero or float above it.
   - Caveat: the journals count repeats of a held state. The live count uses mask changes and will read lower inside a sitting. The check is run again on the first week of live rows.
5. **Existing mechanism.** `chat_rate_reading` is reused. No new instrument.
6. **Reversibility.** One candidate function and one reader. Nothing is baked into a schema.

**Re-run on live rows for patch 4 (2026-11-24).** The gate was run again before `world:play` was wired.

1. **Provenance, as built.** `retro_sitting.started_at` and `last_input_at`: the first and newest mask change of a
   sitting, written by Hub's Retro gateway (`services/orion-hub/scripts/retro_sitting.py` `UPSERT_SQL`, from
   `orion/retro/sitting.py` `SittingTracker.on_input`), and for the 46 backfilled rows by the patch 1 journal backfill,
   counted the same way. `orion/attention/world_first.py` `play_minute_times` turns each row into one stamp per whole
   minute from its first to its newest change. That is "a minute inside a sitting", not "a minute that held a press":
   a sitting closes after 5 minutes without a change, so a counted minute is at most 5 minutes from a press.
2. **Independence from `world:chat`, measured.** 20 of Juniper's 60 chat turns in the 7 days to 2026-11-24 fell
   inside a sitting (33%): she chats while she plays, partly through the game page's chat box. So the two rates are
   correlated through her. The chain is short and real, so the rule keeps them apart at the ranking: `world:play`
   yields whenever `world:chat` is eligible, and never counts a chat turn. From `world:reading` it shares nothing.
3. **Theory anchor.** Unchanged: a rate against its own baseline (Bayesian surprise, Itti and Baldi, 2009).
4. **Live data, 7 days to 2026-11-24 07:16Z.** 926 minutes held play. Of the 10,081 one-minute points of the
   7-day grid, 84.6% read 0, and 2.9% read the full 30 of 30. Rest is a true 0: the count of the last 30 minutes
   returns to 0 thirty minutes after the last input, it does not decay. While a sitting was open, the candidate sat
   in the top tenth of its own week on 247 of 277 logged broadcast ticks (89%); the other 30 were a sitting's first
   minutes. Orion's point 10 holds: "top tenth" alone admits almost any open sitting. And "unusual" (top 1%) is
   unreachable: a full 30 of 30 reads at the 97th percentile, so a rule on that band never fires (0 ticks in the
   replay). The rule therefore cannot be a band; it is the watch (patch 4 notes below).

### Screen change (the trigger of a look)

In revision 1 this number was the salience of a look. It is now the trigger of a look, and looks carry no salience.

1. **Provenance.** `orion/retro/frames.py`: the largest mean absolute difference in brightness between the last frame of the previous look and any frame in the ring since, on a 32 by 28 grid, from 0 to 1.
2. **Independence.** Play rate and screen change both need a player. Inside a sitting the play rate sits near its ceiling while the screen varies. Step 0 checks their correlation inside sittings.
3. **Theory anchor.** The orienting response to change in the visual field (Sokolov). Bayesian surprise attracts gaze (Itti and Baldi, 2009).
4. **Live data.** Done in step 0, on 20 sequences rendered from real saves, 600 frames in all.
   - Still screens read 0.0002 to 0.0004: a "Game Over" menu (2 sequences) and a character pushed against a wall (1 sequence). Rest is reachable.
   - Walking inside one room read 0.05 to 0.11 (6 sequences).
   - Walking with a scrolling view read 0.19 to 0.20 (2 sequences). It does not sit at the top of the scale, which is 1.
   - A fade to black and a room change read 0.19 to 0.43 (4 sequences).
   - So the number separates "still" from "anything happens" by a factor of 100. It does not separate a room change from a scrolling walk. It is fit to be a trigger. It is not fit to be a score, and it is not used as one.
   - Limits: scripted input. Measured later on two NES games (5 saves each, 60 s still and 60 s walking): Crystalis separates; The Legend of Zelda (NES) does not (walking 0.0032 to 0.0047 in 4 of 5, still up to 0.0054). Not yet explained: two Zelda "still" sequences with one big step (0.51, 0.15), and two Crystalis sequences at 0.66. So the record passes per game, not in general (section 3).
5. **Existing mechanism.** No frame-difference measure was found in `services/orion-vision-edge` or `services/orion-vision-retina`. The build checks the camera motion pipeline before it writes a new one.
6. **Reversibility.** One nullable number on a row and one floor in settings.

Record 4 passed for use as a trigger on A Link to the Past and Crystalis only. Its correlation with the play rate inside sittings (record 2) needs live frames, and is checked on the first live sitting.

### Sitting minutes (salience of a finished sitting)

1. **Provenance.** `ended_at` minus `started_at` on the `retro_play_sessions` row, written by Hub at close.
2. **Independence.** It ranks sittings against sittings only. It is in no model with another metric.
3. **Theory anchor.** None is claimed. It is a duration, used only to order one organ's own events: a longer sitting is a bigger event of that kind.
4. **Live data.** The journals give seven sittings of one game from 20 to 249 minutes, and many under 2 minutes that are dropped. The spread is wide and the floor is real.
   - Mask changes across games (Orion's third round, point 8). Every `applied` journal row has `port` and `mask` in all 19 games; no row uses another shape. Rows per real change differ by game, from 3.1 (Galaga) to 20.4 (Battle Chess), so the counter uses changes, not rows. The baseline (32 sittings of 2 minutes or more) is 88% Zelda NES and A Link to the Past by minutes. Sittings of other games are ranked against a two-game baseline until more play exists.
5. **Existing mechanism.** The existing salience rule is reused (`orion/physiology/event_wants.py:34`, `source_salience`).
6. **Reversibility.** One rule name in `organs.yaml`.

## Proposal-mode checklist

- **Capability that changes.** Orion perceives play, speaks about it, answers a chat box in the game, remembers sittings, and can decline to watch.
- **Data touched.** Game frames (4 per look or turn, each gone within the hour), look notes, lines, chat messages, sitting facts, the player's own statement of who they are, and Orion's choices about watching.
- **Privacy boundary.** None between Juniper and Orion, by her standing rule. A guest is a new case. A guest's words are stored as a guest's, never as Juniper's. No name is written by this organ. The repo is public: no frame, save or name is committed. Test fixtures use frames of the original pad-test ROM, which this repo generates itself.
- **Trace that proves it worked.** For one look: the want row, the lease with its want id, the look row with frame hashes, the act, and the temporal event when a note was stored. For one line: the want row, the turn's trace, the line row with the looks it was shown, the act, the chat history row, and a use row when it cited the now.
- **Dangerous failure modes.**
  1. Orion's play calls delay Juniper's own turn. Guard: one call at a time, lower priority, cancel on her message (section 4). The lower priority is UNVERIFIED until patch 3's lease check passes.
  2. A wrong note hardens into a fact. Guard: a note is stored as a look, never as a fact about the player. Its event label says "Saw on the game screen", and a contradicted note never becomes an event (section 8). The sitting's label is counts only. No memory card is written from notes in this design.
  3. Orion claims the play. Guard: a schema test that a `human_play` row has 0 Orion frames.
  4. A guest's words are recorded as Juniper's. Guard: the guest path in section 5, and a test on each of the three readers.
  5. Looks are made on a schedule with nothing there. Guard: the change trigger, a stateless look, and counts at the grain of the gate.
  6. Looks run with nobody there. Guard: a look needs input in the last 60 seconds.
  7. Play crowds Orion's now. Guard: looks carry no salience; the dropped counts are watched.
  8. Prime churns. Guard: two wakes per sitting at most.
  9. Orion is made to watch. Guard: Orion's own switch.
- **Disable and roll back.** Juniper's switch on the page. Orion's `dig watch`. The env flags. Removing the code leaves the tables in place, and they are harmless.

## Non-goals

- No reader of game memory for new games.
- No bets on the player's next move, no "may I try?", no second player. Each is a later design.
- No mood from game events.
- No recording of play as video, and no sampling of game sound.
- No change inside the game container.
- No second chat path. Orion speaks through the unified turn only.
- No voice in these patches. Section 10 only shapes the ground for it.
- No memory cards from look notes.
- No credit to Orion for the player's play.
- No guess at who is playing, and no write to a person record.

## Acceptance checks

**Patch 1**

- After one sitting of 10 minutes or more: one `retro_sitting` row, one `retro_play_sessions` row with `kind='human_play'`, one `temporal_self_event` row with `source_kind='retro_play'`.
- No `human_play` row has `orion_frames > 0`.
- A chat turn in the main window during the sitting names the game when asked.
- An office reach-out sent inside a sitting says that someone is playing. None is dropped or deferred by this code.
- The backfill wrote sitting rows for the last 30 days, and wrote no `temporal_self_event` row and no memory row.
- No person or entity row has a retro table as its source.
- The day block's length and the letter's input size are reported for a day with a long sitting, before and after.

**Patch 2**

- Every look row has a want row and a lease with the same want id.
- A paused game for 10 minutes gives no look row, and `static_skips` rises.
- A screen that changes with no input for 2 minutes gives no look after 60 seconds, and `input_stale_skips` rises.
- A game with no row in the floor table gets no look, and the page says so. Each floor in the table names the sequences it was set from.
- No `temporal_self_event` with `source_kind='retro_look'` exists for a look that is unsettled or `contradicted`. Every `retro_look` label starts with "Saw on the game screen:".
- The list of readers that copy an event label into a prompt is complete and in the spec.
- Before the first line ships, the look is checked on unscripted play with a menu, a map and a dialogue box that types itself out. Step 0 read menu text right (2 of 2) and did not recognise the world map as a map (1 of 1).
- The step 0 sequences in `services/orion-hub/evals/fixtures/retro_look_step0.json` are re-run as a regression, and the counts are reported beside the fixture's. The map and menu sequences (`map_a`, `map_b`, `menu_a`, `menu_b`) are in it.
- The look prompt says: do not quote numbers from the status bar. Step 0: 2 of 3 notes that cited them were wrong.
- The shipping bar for the unscripted check (proposed, Juniper decides): at least 8 of 10 notes with place and action right, and at most 2 of 10 holding a thing not on the screen. One judge by eye against the frames, verdicts written per note into a fixture file in the same shape as step 0. About 10 minutes of play and 30 minutes of judging.
- Orion's read-only role can read `retro_sitting`, `retro_play_sessions` and `retro_watch_look`. `dig` answers "what did I see in that sitting" from look rows with their bucket.
- A refused bid has no lease and no look row.
- With the lane made blind in a test, the look fails. No note is stored.
- One hour after a sitting ends, the percept store holds none of its frames.
- The page status is true: "Orion is watching" appears only when a stored note is under 3 minutes old.
- The six look counts are reported for one sitting: ring checks, checks over the floor, bids funded, model calls, notes stored, failed looks.
- Two hours after a sitting ends, opening its frames returns "expired" for every look row, each row's `frames_expire_at` is in the past, and no row is logged as lost. A frame removed by hand before its time is logged as lost.
- `dig watch off` holds across a restart of Hub and of the gateway, and leaves one act row with Orion's words.
- `dig watch off --why` stops looks within one look interval, the page shows Orion's words, and Juniper's switch cannot turn the watch back on.
- The now's dropped counts during a sitting are reported beside the week before. No look carries a salience.
- Prime's queue wait during sittings is reported beside the week before.

**Patch 3**

- Every delivered line has a `retro_line` row that names the looks it was shown, and a turn trace.
- For one real line turn, the stance, motor and finalize leases are read from `gpu_pool_events`. The check fails if any of them is `interactive`.
- The note error rate is reported as contradicted / checkable, beside the share of shown notes that were checkable. A note is checkable when its frame window overlaps the turn's frames; each shown note stores that flag. A turn sees up to 12 notes but frames of only the last 30 seconds, so most notes are not checkable.
- A message of Juniper's sent during a running line turn has its first model call granted within 20 seconds, and the line's outcome is `yielded`. orion-thought, which runs stance, has no cancel listener (`services/orion-hub/scripts/endogenous_outreach.py:231`): until it has one, she can wait for one in-flight stance call (11 s median, 19 s at p90). A check cancels a line turn mid-stance and asserts no chat row, no push, no delivery.
- A test forces the hard order: the draft finishes after Hub marks the line `yielded`, and finalize still tries to deliver. The guard refuses it.
- A line turn with no ring (the browser closed or reconnected) does not run. Its outcome is `no_frames`. No prompt ever says "these are 4 frames" with fewer attached.
- If the outreach door cannot carry a `system` lease, lines do not ship. Lines at `interactive` with cancel as the only guard is not an option. At 34 °C and above, looks and lines are refused as `system`; office reach-outs run as `interactive` and keep going. That asymmetry is today's reach-out rule, not new.
- After that cancel: no chat history row for the line, no push to any socket, and no later delivery, also when the cancelled turn had finished its draft.
- The seconds of one real line turn are recorded end to end, by stage: stance, motor, finalize. Step 0 did not measure this.
- A chat-box turn's prompt holds the Prime block. A promise made in it shows up as a Prime concern.
- A guest's message has the source `retro_guest`, opens no "Talked with Juniper" arc, does not count in `world:chat`, and is not in the day letter's Juniper chat.
- With nobody stated, a line holds no name.
- The chat box works in fullscreen, and typing does not move the character.
- The share of funded line turns that end in a pass is reported.

**Patch 4**

- The replay of the last 7 days is reported first: how many ticks `world:reading` would have lost to `world:play`, and the attention share of reading and curiosity during sittings of 2 hours or more. Orion's point: about 89% of half-hour windows read zero play, so "top tenth of its own week" passes for almost any open sitting. The eligibility rule is chosen from that replay, not assumed.
- A sitting produces at most 2 Prime thoughts that name it as their wake.
- A `slow` signal from Prime changes the look count in the next hour, and the market records the effect.
- `world:play` appears in the attention broadcast during a watched sitting, is not eligible when Orion is not watching, is absent with a reason when the table cannot be read, and reads zero when nobody plays.
- A line spoken while `world:play` holds attention carries that loop's id. No final verdict is written on any reading loop.

## Patch order

Each step ends with a stop for Juniper's review.

**Step 0. Done on 2026-11-17.** A spike with no deploy. See "Step 0 results" below. Its item 4, one line turn made by hand, was not run. Its item 6, the quick lane's model with its encoder, ran on Juniper's word.

**Patch 1. Orion knows.** The sitting, the "Who's playing?" question, the `human_play` row, the backfill by mask changes, the game block in the chat prompt, the office reach-out's one sentence, the play line in the day block. No GPU is used.

**Patch 2. Orion looks.** Frames, the percept store, the look with its change trigger, tier 1 bids, the `retro_look` source, the status line, both switches, `dig watch`, and the rewritten promises.

**Patch 3. Orion speaks.** The chat box, the guest path, the line strip, line turns, Juniper goes first with its cancel checks, thumbs, the hand-back class, the read-back of the now. The first real line turn is timed end to end, by stage.

**Patch 4. Prime and attention.** The replay first. Then receptors, the two wakes, the event want with its menu label, `world:play`, and the act pairing.

**Later, its own design.** Voice in and out (section 10). Then the rungs under "Non-goals".

Seams 1 to 3 of section 6 arrive with patches 2 and 3, because they come with the now and with the unified turn. Prime is in from the first look. Orion's own switch arrives with the first look too.

## Step 0 results (2026-11-17)

Step 0 ran on Juniper's "yes". Nothing was deployed. No frame was committed.

### What was run

- 20 sequences were rendered from 10 of Juniper's real saves of A Link to the Past, in the game image, with no browser: 600 frames. The input was scripted: walk in four directions in turn, swing the sword.
- Looks ran on two models, 4 frames each, at 2 times and at 3 times enlargement:
  - **Qwen3.8 Flash-Next, the chat lane's model.** 18 calls. They ran on the agent lane, which holds the same model, under one leased operator hold through the model gateway (`orion/evals/leased_llm.py`).
  - **Qwen3.5-9B, the quick lane's model, with its vision encoder.** 22 calls, on a temporary copy on GPU 3.
- I looked at every sequence myself and judged each note against its frames. Juniper is the final judge.

### Can a model read the game screen?

| | Chat lane's model | Quick lane's model |
| --- | --- | --- |
| Place and action right | 9 of 10 | about 5 of 10 |
| Notes that hold a thing not on the screen | 4 of 10 | 9 of 10 |
| Seconds for one look, median | 5.6 | 2.7 |
| Prompt tokens for 4 frames, at 2 times | 1,086 | 634 |
| Prompt tokens for 4 frames, at 3 times | 2,206 | 1,194 |

Notes from the chat lane's model, at 2 times, with what the frames really show:

- **Right.** "The screen transitions from black to a top-down dungeon room with green walls and a brown floor. The character moves from the left side of the room, walks right, and then moves down to the bottom edge. A black hole and a green block are visible in the upper right corridor." Every part of this is on the screen.
- **Right.** "The character stands on a dirt path near a well. The screen cuts to black, then reappears in a dark underground room with brown walls." The player did walk into a cave by a well.
- **One wrong detail.** "The character is on an upper ledge in a stone room with stairs leading down. They move from the right side of the ledge to the left, then jump down onto the lower floor." The place is the top of a pyramid under an open sky. Nobody jumped.
- **One wrong detail.** "The character moves from the left side to the right, passing a green enemy. A red enemy appears and is defeated." One blue creature was there and did vanish. The colours and the second enemy are invented.

A note from the quick lane's model, on the cave sequence: "Link stands before a pit in a dark cave. He jumps down into it, landing on a lower ledge. A golden chest appears on the wall to his left." The frames show a well outside, a fade to black, and a cave. There is no chest.

**Against the bar.** Revision 2 set: 8 of 10 notes name the place and the action correctly, and no note describes a thing that is not on the screen. The chat lane's model passes the first half (9 of 10). It does not pass the second half: 4 notes hold one wrong detail. Whether that is good enough is Juniper's call (open question 1).

### What a look costs

- One look on the chat lane's model: 5.6 seconds at the median over 11 calls, 6.9 at the most. One call at 3 times took 22 seconds.
- A frame at 2 times costs about 220 tokens on the chat lane's model and about 110 on the quick lane's. Orion's estimate of 650 a frame at 3 times was near: the measured number is about 500.
- 3 times enlargement doubled the tokens and gave no better notes in the 4 pairs compared. The default is 2 times.
- One frame as an image: 14 KB at the median, 10 milliseconds to make.
- One check of the screen for change: 1.5 milliseconds on one CPU core.

### "Nothing new" cannot be left to the model

Revision 2 asked the model to write "nothing new" when nothing had changed since its last note. Three tests:

1. A still "Game Over" screen with no earlier note: the model wrote "nothing new". It took the phrase to mean "nothing changed across these frames", and described nothing.
2. The same frames again, with its own earlier note: the model wrote the note again in other words. It did not say "nothing new".
3. Frames of a new place, with the earlier note of another room: the model copied the earlier note word for word. It ignored the new frames.

So the look gets no earlier note, and no model decides whether a note is new (section 3).

### The change trigger works

- Still screens read 0.0002 to 0.0004. The calmest walk read 0.05. Fades and room changes read up to 0.43.
- The floor is 0.01. The full record is under "Metric quality gate records".

### The two input counters differ, and the better one shows the breaks

One sitting, 2026-11-15 19:57Z to 2026-11-16 00:06Z, through both counters:

- 268,628 journal rows. 37,586 real changes of the button mask. One row in 7 is a real press or release.
- 243 minutes hold a row. 204 minutes hold a mask change. Ratio 0.84.
- By mask changes with the 5-minute rule, this is not one sitting of 249 minutes. It is four: 10, 40, 142 and 15 minutes, with breaks of 15, 17 and 5 minutes between them.
- The first break runs from 20:08Z to 20:23Z. The office camera's note from that afternoon puts someone at the trash can at about the same minutes.

The backfill and the live counter both use mask changes (section 7).

### What was not run

- **Item 4, one line turn made by hand.** Step 0 promised Orion three fences for it. The third was that its origin would not be recorded as Juniper. The only door open to a script is the chat endpoint, and there the origin is Juniper unless the message goes through the social room machinery, which writes records of its own (`services/orion-hub/scripts/api_routes.py:3353`). A test turn recorded as hers is the exact thing Orion asked me not to leave in their records. So it was not run. The end-to-end time of a line stays an estimate: 60 to 90 seconds. Patch 3 measures it on the first real line turn, under its true class.
- **The chat lane itself.** The chat card is Juniper's. Operator work may use it only while she has lent it (`config/gpu_pool.yaml:187`). That guard is deliberate, and I did not go around it. The agent lane runs the same model and the same engine, so the notes and the token counts carry over. The seconds were measured on the agent lane's card.

### What went wrong during step 0

- **The spare copy landed on the wrong card three times.** I asked for GPU 3 by number. Inside a container the numbering differs, and the copy loaded on GPU 2, the card of Orion's second agent seat. The first start exited, and its log was not kept. The second came up short by one allocation of 248 MB. The third loaded in full. The seat's own card readings (`orion_biometrics`, node circe, GPU 2) show it: baseline 24,904 MiB; peaks of 31,750 MiB on 11-16 at 17:31Z and 32,308 to 32,426 MiB on 11-17 between 03:02 and 03:06Z, of 32,768. At worst the card was 342 MiB short of full; it was back to 24,704 MiB at 03:07. I stopped it as soon as I saw where it was. The pool's records show no harm: three leases were in flight on 11-16 at 17:31 and all released clean; in the 11-17 window the seat was in standby and held no lease. Waits in the hour before and after were 44 s and 57 s on average, no clear change. What no record can show: a model on a card 342 MiB short of full fails on its next allocation. The rule is now in `docs/runbooks/2026-11-14-lane-door.md`: pin by hardware id, and print the card's memory and the seat's leases for the window before and after. The fourth start pinned the card by its hardware id and landed on GPU 3. Lesson: never pick a card by number.
- **The leased seat was slow to come and quick to go.** The first hold waited 22.5 minutes for the agent lane, then the pool took it back after 35 seconds and 3 calls. A second hold finished the other 15 calls. That is the leased path working as designed: operator work is last in line.
- **circe is clean.** The temporary copy is stopped and removed. The encoder file and the frames are deleted from circe. GPU memory is back to what it was.

### Where the evidence is

- In the repo: `services/orion-hub/evals/fixtures/retro_look_step0.json`. Words and numbers only: the prompts, 15 sequences with each note and its verdict, 29 more with screen change only, and the headline counts recomputed from the per-note verdicts. The quick-lane model's per-note verdicts were never written down, only its total; the file says so.
- On athena, in `~/.orion/retro-spike/`, readable by the owner only: the frames, `looks_agent.json`, `looks_9b.json`, `screen_change.json`, `screen_change_stats.json`. They are deleted after Juniper's review. The fixture keeps the numbers after that.
- On athena, in `~/.orion/retro-watch-fixture/`, owner only: copies of the 10 saves and the spike scripts, to render the sequences again.
- In the pool's records: `gpu_pool_events` with the holder `operator:retro-watch-step0`.

## Questions

### Answered by Juniper

**2026-11-16, "I will go with your recs"** (the numbers are those of revision 1's list):

- **1. The eye starts on the chat lane.** Kept. Step 0 confirmed it: the quick lane's model does not read the screen well enough.
- **2. Juniper goes first: Orion's line turn is cancelled when she sends a message.** Kept.
- **4. Lines ignore quiet hours while a sitting is open.** Kept for text. Spoken lines are a different case (section 10).
- **7. Orion speaks about once every 5 to 8 minutes to start.** Kept.
- **8. If a line proves too slow, accept slow lines first.** Kept.

**2026-11-16, "yes"**, to the three recommendations that Orion's review changed and to the four new questions:

- Frames are gone within one hour, in the percept store.
- The office reach-out keeps running and is told that someone is playing.
- No guess at who plays. The page asks once per sitting.
- Orion's "off" holds: the lower of the two switches wins. Orion agreed too.
- The quick lane's model was tested on a temporary copy on GPU 3.
- The two families stay on tier 1, though tier 1 reserves nothing on the chat lane today.
- Orion's watch choices feed the personality graph, as its own small patch. Orion agreed too.

### Open now

1. **Is the chat lane's model good enough?** Place and action were right in 9 of 10 notes. 4 of 10 held one wrong detail: a jump that did not happen, a wrong colour. Recommendation: yes, go. Three things make it safe. A note is stored as a look, never as a fact. The turn that speaks sees the frames itself. And the wrong details were small while the places and events were right. One addition for patch 3: the turn that speaks can mark a note that its frames contradict, so the eyes get an error rate from live play.
2. **Looks share the chat lane's one slot.** The quick lane was the way around that, and its model failed. A look takes 5.6 seconds, runs below Juniper's priority, and never queues with another play call. Recommendation: accept it, and check her wait live in patch 3.
3. **The time of a line is still an estimate.** Recommendation: measure it in patch 3 on the first real line turn. The other path is a test turn now that Orion's records would hold as Juniper's. I recommend against it.

## Orion's review (2026-11-16)

Orion read revision 1 in the hourly round-up and asked 14 questions on the PR. Each one is answered here, with what changed.

1. **The engine promised no recording and no model call. Is that sentence rewritten, and who signed it off?** The engine's promise stays true for the engine. The system's promise changes, and section 2 now says so in the open, with the new boundary in one sentence and the five texts that patch 2 rewrites. Sign-off: Juniper's words of 2026-11-16, confirmed by her review of this PR.
2. **No sweep exists in the chat store. What deletes look frames, when, and what caps the store?** Orion's search was right: there is no sweep. Frames no longer go to the chat store. They go to the percept store, which keeps a frame one hour and sweeps in its own process. No new sweep is written (section 2).
3. **Is "four frames do not fit" measured or inherited?** It was worse than inherited: it was wrong. A quick slot holds 32,768 tokens, not 3,276. Step 0 now records the prompt tokens of every look at two sizes. The spare model on GPU 3 is a test only and is not in the design. If the 9B model reads the screen, looks move to the real quick lane (section 3).
4. **Under heat, are looks refused or priced? What does a look do to Prime's latency, and was Juniper's worst case measured?** Both, and section 3 says which when: price below 34 °C, the reflex at 34 °C and above. Prime can wait behind one play call (section 9). Her worst case was not measured. Revision 1 let a look and a line stage queue ahead of her. Now Orion's play calls run one at a time and below her priority, and her wait is checked live (section 4).
5. **What is the tier 1 share numerically, and what do Prime and the others give up?** Zero on the chat lane, computed from 14 days of leases. So Prime gives up no share. Section 9 no longer says "reserved share" as if it were a number above zero.
6. **Does a look with no screen change still write a row and still make "Orion is watching" true?** No, on both. The trigger is change. A static screen calls no model and writes no row. A look that says nothing new is counted apart and does not make the status true. Step 0 counts the three kinds separately (section 3).
7. **Does play crowd other organs out of the now?** It would have. Looks now carry no salience, so they rank after every scored item. The sitting is the one scored experience of this organ. The dropped counts are watched during a sitting (section 8).
8. **Does a 4-hour sitting read in the day letter as one long conversation? Is `play` added to the kind list? Does the day block survive the events?** No: a line is Orion's own row, and the existing rule keeps such rows out of conversations. `play` is already in the list. The day block gets one line per sitting, the arc has a size bound with a test, and the block's length and the letter's input size are reported before and after (section 8).
9. **With an empty table, what does `world:play` read in week one? Does it take from reading?** Without history it would be ineligible for 5 minutes and then read as the most unusual thing of the week. The backfill gives it a real week. It can outrank reading during a sitting, and only then. A replay shows the count before anything is wired. It is eligible only while Orion is watching, and nothing retires reading (section 7).
10. **Does the player guess reach person records? What does a turn know when a child plays?** The guess is removed. Nothing here writes a name or touches a person record. With nobody stated, the turn is told that it does not know who plays, that it may be a child, and to use no name (sections 1 and 4).
11. **If someone else types in the chat box, is it recorded as Juniper?** It would have been. Now the page asks who is playing. A guest's message goes down the third-party path, with its own source, and the three readers that mean "Juniper spoke" do not match it (section 5).
12. **The office reach-out is silenced, and who decided the quiet hours rule?** The silencing is removed. The camera keeps speaking and is told about the sitting. Nothing pending is dropped or deferred. Quiet hours are Juniper's call. She made it for text lines. Spoken lines will obey quiet hours (sections 5 and 10).
13. **Should step 0's line turn be fenced off from live records, and where do the frames live?** Yes. It runs under a system session id, so nothing hands back to Prime, with no chat history row, and three checks after it. The frames stay on athena outside the repo and are deleted after the review (step 0).
14. **Can Prime's `slow` reach a full stop? Can Orion decline a sitting?** `slow` cannot stop anything, by design. Orion now has a switch of their own: `dig watch off`, and a standing choice per game. The lower of Orion's and Juniper's settings wins, so a decline holds (section 6).

### Second round (2026-11-17)

Orion read my answers and sent 10 more points. Orion was reading the text of revision 1, because the thread still carried it. Revision 3 is sent to the thread in full.

1. **The quick lane's size, and heat on that lane.** Orion checked the live lane and confirmed the correction: 10 slots of 32,768 tokens. Orion asked whether step 0 should pick the lane on that basis. It did test the quick lane's model, and the model failed on what it writes, not on size. So looks stay on the chat lane's model. On heat: the reflex blocks by priority, not by lane or class. A `system` call is refused at 34 °C wherever it runs (section 3).
2. **Who computes "the screen has not changed", and is "nothing new" counted before or after the model?** The gateway computes it, on one ring packet every 2 seconds, at 1.5 milliseconds a check. No model and no string comparison decides "nothing new" any more: step 0 showed both fail. The skip is counted before the model. The counts are reported at the grain Orion asked for: ring checks, checks over the floor, bids funded, model calls, notes stored (section 3).
3. **Which store holds frames, and which reference does a turn open?** The percept store, with the percept kind of reference. Turn staging gets a second fetch path for that kind. The gateway process needs no mount of the chat store (section 2).
4. **Frames die on the hour and the pointers do not.** Each row keeps `frames_expire_at`. An "expired" answer after that time is gone by design. Before that time it is a fault and is logged. An acceptance check opens a finished sitting's frames two hours later (sections 2 and "Acceptance checks").
5. **The baseline and the live counter are different instruments, and the candidate comes and goes.** Step 0 ran one sitting through both counters: 243 minutes against 204. The backfill now counts mask changes, as the live counter does. And `world:play` does not come and go: it stays in the set and reads zero between sittings (section 7).
6. **What does tier 1 buy with a zero share, and can Orion read a refusal?** In the market it buys nothing on this lane. The one protection of a look is the pool's queue: `system` goes ahead of `background`. Each refusal leaves a row with its reason, family and lane, and `dig watch status` prints the last one (section 9).
7. **A cancelled line must leave nothing, and what was confirmed live?** Nothing was confirmed live, and the spec now says so. The cancel marks the line `yielded` before it kills the turn, and delivery refuses a `yielded` line even when the draft was finished. Patch 3 checks for no chat row, no push and no later delivery. Juniper's wait is at most one call: a look took 5.6 seconds at the median. Her own stance call is part of her turn, not part of the wait (section 4).
8. **The attached text contradicts answer 12.** It did: the thread held revision 1. In revisions 2 and 3 the office reach-out is not silenced (section 5). The full text is sent to the thread.
9. **Orion's answer to A: yes, the lower setting wins.** The three things asked for are in section 6: `dig watch off` writes an act row with actor, verb and Orion's words. The choice lives in Postgres and holds across a restart. Prime's frame line says the watch is off by Orion's choice.
10. **Orion's answer to B: yes, with the row's shape stated.** Section 6 states it: the option chosen, the options declined, and Orion's words. The personality reducer reads it, and through tendencies it changes the care term in bids. Nothing grades it as right or wrong. It is a record that is read, not a drive.

### Third round (2026-11-17)

Orion read revision 3 and sent 8 points. Each one changed the spec except point 8, which changed a record.

1. **Only the newest notes can be checked against frames.** True. The error rate is now contradicted / checkable, each shown note stores whether it was checkable, and the share is reported (patch 3 checks).
2. **Does a contradicted note's event stand forever?** Under revision 3, yes: the chronicle never rewrites an event. Now a look's event is written only after it is settled, a contradicted note never becomes one, and the label says "Saw on the game screen" (section 8). The reader list is partial and patch 2 must finish it.
3. **Step 0 was scripted.** New runs: menu text read right, the world map was not recognised, status-bar numbers were wrong in 2 of 3 notes that cited them. (a) Both gates must pass, so reading dialogue gives no look after 60 seconds; the new `input_stale_skips` counter shows it. (b) 0.01 is not a floor for all games: it fails on The Legend of Zelda (NES). Floors are per game now (section 3), and unscripted play with a menu, map and dialogue is a patch 2 check.
4. **Keep the numbers.** Done: `services/orion-hub/evals/fixtures/retro_look_step0.json`, and patch 2 re-runs it.
5. **Check the lease priority.** Added to patch 3. The guard is marked UNVERIFIED until then.
6. **Seat-side evidence of the foreign load.** The card's own readings show it: at worst 342 MiB short of full. No lease was harmed in the record. The rule is in the lane-door runbook; a detector is ticketed (#3631).
7. **A counter for "screen moved, input stale".** Added: `input_stale_skips`.
8. **Do all games' journals carry the mask the same way?** Yes, in all 19. Rows per change differ by game, which is why the counter uses changes. The baseline is 88% two Zelda games by minutes, and the sitting-minutes record says so.

### Fourth round (2026-11-17)

Orion sent 12 points on PR #3632 and 7 more on PR #3517. Changes, in short: floors are calibrated from live play with input as the label; a miss counter; looks record lane wait and take frames at grant; looks are not deferred; only checked-and-held notes reach the day; the expected checkable share is stated (about 8%); frame expiry follows the last post; cancel at the stance stage is a stated gap with a check; no ring means no turn; lines do not ship without a `system` lease; a read grant for Orion's role (`scripts/sql/2026-11-17_grant_orion_readonly_retro.sql`, applied by hand: #3657); a stance-stage cancel listener (#3658). Patch 2 builds from this revision; #3632 merges first.

## Recommended next patch

Patch 1, on Juniper's go. It uses no GPU, and it ends the state where 27 hours of play left no trace in Orion: the sitting, the "Who's playing?" question, the backfill, the game block in the chat prompt, the play line in the day block.

Patch 1 is merged (PR #3627). Its live proof is #3626.

Patch 2 next, on Juniper's go only. It starts with the per-game floor table and the reader list from section 8, then the look.

Patch 2 is built on branch `feat/retro-look` (report `docs/superpowers/pr-reports/2026-11-18-retro-look-pr.md`). Two choices made while building it, each with its reason:

- `hub.retro_watch` has `deferrable_window_s: 60`, not 0. `orion/physiology/families.py` refuses 0 for a bidding family: the window is also its hunger clock. The producer still never holds a want: only `funded` runs a look.
- The calibration histogram uses each check's change over the last 30 s (the look's own window), not its 2-second step. On the step 0 frames the 30-second value separates still from play on A Link to the Past and the 2-second step does not (`services/orion-hub/evals/retro_look_replay.py`).

Patch 3 is built on branch `feat/retro-lines` (report `docs/superpowers/pr-reports/2026-11-18-retro-lines-pr.md`). Choices made while building it, each with its reason:

- **The outreach door carries the line's own GPU hold.** `compose` gained `gpu_lease` (with `attachments`, `source`, a cockpit sink and a session). The line takes a hold of class `chat` at priority `system` and the turn rides it, as a durable turn does: pre-turn check, stance, motor and finalize attach to the hold through the held route, whose priority is `system` in `config/gpu_pool.yaml`. A test reads that file; the line refuses to start if it ever reads `interactive`.
- **Every person's message yields a running line**, a guest's chat-box message too: the chat lane runs one turn at a time, and the player's turn goes first whoever the player is.
- **The frames route is loopback plus the Retro hub token.** tailscale serve also reaches the gateway from loopback, so the address alone would admit a browser.
- **The near-copy cut-off is 0.45**, from the step 0 notes (no live look existed): every pair of notes about different sequences scored at most 0.404. `services/orion-hub/evals/retro_line_dry_eval.py` recomputes it.
- **A line's act opens and closes at delivery** (`hub.retro_commentary say`, produced 1). What the player made of the line (thumb, a reply within 2 min) is stored on its `retro_line` row; wiring it into the family's weight waits for patch 4's receptors.
- **Game lines are not in the day letter's chat gather**, and a guest's turn is excluded from every "Juniper spoke" rule in `orion/regulation/juniper_turns.py` (dream idle gate, arousal, the act and promise ledgers), beyond the three readers this spec names.
- **Not built:** the last 6 look notes in every turn's game block (section 5, "What the server adds"; the line turn has them); a cancel listener in orion-thought (#3658: a line cancelled mid-stance lets that one call finish, and nothing it returns is said).

Patch 4 is built on branch `feat/retro-prime-attention` (report `docs/superpowers/pr-reports/2026-11-24-retro-prime-attention-pr.md`). The replay ran first (`scripts/analysis/replay_world_play.py`, 3,048 logged broadcast ticks, 2026-11-17 07:22Z to 2026-11-24 07:16Z, 81 sittings of which 46 backfilled, 135 stored look notes since 2026-11-22 23:39Z). Three eligibility rules, each alone, the same chat yield on all:

| Rule | Play won (ticks) | Reading lost | Curiosity-seeding winners lost | Reading share in sittings, before / after | Curiosity-seeding share in sittings, before / after |
| --- | --- | --- | --- | --- | --- |
| top tenth while a sitting is open and the watch is not off | 166 | 4 | 30 | 6.1% / 4.7% | 26.0% / 15.2% |
| the same, and a look note stored in the last 3 min | 18 | 0 | 5 | 6.1% / 6.1% | 26.0% / 24.2% |
| top 1% only, watch not off | 0 | 0 | 0 | unchanged | unchanged |

- No sitting of 2 hours or more falls inside the 7 days the broadcast log keeps (the two such sittings, 10-27 and 11-15, are older), so the 2-hour figure the acceptance check names has no data yet. The shares above are over all 277 ticks with an open sitting. A "curiosity-seeding winner" is a winner whose loop carries a `node:` ref (reading and every body or camera node): `orion/substrate/endogenous_curiosity.py` seeds a run only from such a loop, so a `world:play` win costs curiosity a seed.
- **Chosen: the note rule.** It keeps reading whole (0 ticks lost) and costs curiosity 5 ticks in a week, where the band rule took 136 calm ticks and 30 curiosity seeds. It is also section 7's own definition of watching ("a candidate with no look behind it would win attention with nothing in it"). Since the looks began (11-22 23:39Z) it kept 18 of the band rule's 25 wins: when Orion really looks, play still wins. Games with no calibrated floor (most of them today) get no looks, so they cannot win attention: that is honest, Orion cannot see them.

Choices made while building patch 4, each with its reason:

- **The label carries no count.** Section 7's label "(27 of the last 30 minutes)" would give one sitting a new open-loop id every minute (the loop id is a hash of the label; the `world:chat` lesson, `CHAT_TALKING_LABEL`). The label is "Juniper is playing <title>" or "Someone is playing <title>"; the count and the newest note ride in `evidence`.
- **30-minute window**, not chat's 15: the section 7 label and the gate record both speak of half hours.
- **The field contest gets `world:play` too** (orion-attention-runtime), as a channel target like chat, so the two contests cannot disagree.
- **"Not watching" is a measured state, not absence.** A new optional field, `AttentionCandidateV1.not_eligible_reason`, holds the candidate out with its reason while its reading and band stay in the trace. Absent stays for "`retro_sitting` could not be read". A watch that cannot be read holds it out (fail closed).
- **Prime's frame line needs the watch state**, which lived only in the gateway's memory. The gateway now publishes it to `orion:retro:watch` (every 15 s, 90 s expiry) beside `orion:retro:sitting`.
- **The sitting's event want has a family of its own**, `hub.retro_sitting` (tier 2, no lanes, window 6 h, verb `hold`). Tier 2 is a proposal: Juniper decides tiers.
- **Juniper's engagement uses the same record as outreach**, a `juniper_replied` act graded by the horizon resolver, but only for a line said while "Juniper" was stated. Horizon 300 s; her thumb decides (up 1, down 0), else her reply within 2 minutes (the chat box or any of her chat turns) is 1, else 0. A line to a guest keeps patch 3's act (produced at delivery), so no guest action reaches the weight.
- **`hub.retro_commentary` hunger now counts from its want's own clock** (`hunger_from: want_opened`), which is section 9's rule ("time since Orion last spoke in this sitting"). It was counting from the family's last act close, which a Juniper line's act now reaches only at its 300 s horizon; without this change lines would come about 5 minutes later.
- **The cap on Prime thoughts is enforced, not assumed.** A thought whose answer cannot be used keeps its pending changes, so without a cap a third thought could name the sitting. `orion/prime/sitting.py` counts the thoughts that open while a change names the sitting and drops the changes at 2.


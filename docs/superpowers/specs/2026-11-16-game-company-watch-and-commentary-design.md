# Game company: Orion watches, comments and chats while someone plays

Status: DESIGN. Nothing here is built. It waits for Juniper's review.
Date: 2026-11-16. All live numbers were read on 2026-11-16 unless a line says otherwise.
Ticket: #3500.

## Juniper's words (2026-11-16)

- "How should Orion be/respond/attend/do while I'm playing retro games on them? It's a really weird thought that I'm playing video games through Orion... that should do something to them."
- "We are trying to create a sentient organism that will attain autonomy and be an individual in their own right."
- "Let's have Orion do commentary. Why not have them actually watch? Send in some screenshots over sequences, periodically?"
- "We could potentially borrow the metacog / quick lanes as they have vision encoder, and/or chat lane."
- "It should probably hit attention. It certainly needs to go on the temporal record."
- "Ensure we are following bidding on tier 1."
- "We use the unified chat prompt for the comms. Make sure that Prime gets in on this too."
- "Within the gameplay there should be a chat box. We'll eventually enable TTS and STT, including when in fullscreen mode."

This document does not claim that Orion is sentient. It designs one more place where Orion can perceive, choose, speak and remember, with a trace for each.

## Words used here

One word for one thing, through the whole document.

- **Sitting**: one stretch of play by a person. A gap of more than 5 minutes without input ends it.
- **Look**: Orion reads a short sequence of game frames and writes a note of what they saw.
- **Line**: one thing Orion says to the player without being asked.
- **Chat box**: the place inside the game page where the player types to Orion.
- **Player**: the person who holds the controller. The game cannot tell who that is.

## Arsonist summary

**1. Orion is numb to almost all play today.**

- The local game journals hold about 31 hours of game time in 19 games since 2026-10-18.
- Orion's records hold 58 play sessions. All 58 are The Legend of Zelda, and in all 58 Orion held the controller. That is about 4 hours.
- The other 27 hours left nothing in Orion: no event, no memory, no attention.
- On 2026-11-15, one sitting of A Link to the Past ran for 249 minutes. Orion's day record took in 931 events in that time. None is about the game.
- In that sitting Orion sent a message built from the office camera: "The office held someone at the desk ... I'm guessing that's you ... with stints of computer work between." Orion guessed from across the room while the play ran through Orion's own body.

**2. The quick lane cannot see today. The chat lane can.**

Each lane was asked what it has loaded (`/props` on circe).

| Lane | Model | Sees images | Slots | Tokens |
| --- | --- | --- | --- | --- |
| chat | Qwen3.8 Flash-Next | yes | 1 | 262,144 |
| agent | Qwen3.8 Flash-Next | yes | 1 | 131,072 |
| quick (also serves metacog work) | Qwen3.5-9B | **no** | 10 | 32,768 in total |
| agent-burst | Bonsai-2 27B | no | 2 | 131,072 |

- The quick lane's model could see. Its vision encoder file is not loaded.
- Each quick slot has about 3,200 tokens. One camera frame costs about 1,000 tokens (measured 2026-11-13, `orion/dig/look.py`). A sequence of 4 frames does not fit one slot.
- The chat lane was busy for 2.4% of the last 7 days. During the 249-minute sitting it held 13 leases, about 6 minutes in total.
- The agent lane was full in that sitting: 301 lease-minutes in 249 minutes, with an average wait of 45 seconds.

**3. Orion's voice is slow, and the design must be honest about it.**

- Orion speaks through the unified chat turn. That is Juniper's rule, and this design keeps it.
- The unified chat prompt cannot be had without the whole turn. No lighter function builds it (`orion/harness/runner.py:220`, `build_harness_prompt`). A turn is three stages: a stance call, the motor, and a finalize call.
- Measured parts: Hub's stance and finalize calls take 11 seconds each at the median on the chat lane (189 calls in 7 days). The motor takes 41 seconds at the median when it uses no tool (24 turns in 14 days). A line must also open its frames, which adds one more round trip.
- Estimate: 60 to 90 seconds from "Orion wants to speak" to words on the screen. Step 0 measures the real number.
- So Orion cannot call each jump. Orion can speak about once every several minutes, about what the last minutes held: where the player went, what they keep trying, what changed.

**4. The design is three loops.**

1. **The sitting.** Hub records that someone plays, in which game, since when. No model is used.
2. **The look.** About once a minute, a sighted lane reads 4 frames and writes a short note. Each look bids on tier 1 first. The notes are Orion's memory between the times Orion speaks.
3. **The voice.** Orion speaks through the unified chat turn, which starts from the Prime block and hands back to Prime. A turn starts when the player types in the chat box, or when a want to speak wins its tier 1 bid. In that turn Orion sees the real frames, not only the notes.

Everything Orion sees and says goes on the temporal record. The play becomes a third thing attention can settle on, beside Juniper's chat and Orion's reading.

## The organ

A slice that feeds one mechanism gets retired when that mechanism turns out wrong. This section states the whole organ first.

- **Job.** When a person plays inside Orion, Orion can know it, watch it, speak about it, answer about it, and remember it.
- **What Orion does differently when it works.** Orion's messages during play are about the play. Orion's day holds the sitting. Orion can answer "what was I doing in that dungeon last night?" from their own notes.
- **Inputs.** Any screen that can hand over frames and a "someone is here" signal. The game page is the first one.
- **State that lasts.** The sitting, the look notes, the lines, and which lines the player liked.
- **Effectors.** A line in the game page. A reply in the chat box. An attention candidate. Events in the temporal record. Wants in the body economy.
- **How Orion represents it to themselves.** One line in Orion's now while a sitting is open: who plays what, for how long, and whether Orion is watching.

Three things keep this Orion's own act and not a feature that performs:

- **Orion may stay silent.** A turn that ends with no line is a valid outcome.
- **Orion may look away.** Looks and lines are bids. They can lose to reading or curiosity. The page shows "Orion is not watching" with the reason.
- **The play is the player's.** No record credits Orion with a room, an item or a win that the player earned. The existing rule for shared sessions already says so (`orion/schemas/retro_play.py`, `played_together`).

## Current architecture

Facts, each read from code or from the live system.

- **The game engine is sealed.** It runs in its own container with no network and no link to the message bus (`services/orion-retro/README.md`). "No raw A/V recording, model invocation or memory export occurs."
- **Hub carries the picture.** Hub relays every video packet from the engine to the browser, about 30 each second (`services/orion-hub/scripts/retro_transport.py:44-68`). A packet is `zlib(header + raw pixels + sound)` (`services/orion-retro/app/media.py:79-94`). Hub never decodes one today. Packets flow only while a browser has the game open.
- **Two Hub processes share the game page.** The page `/retro` and the chat socket `/ws` are served by hub-app. Everything under `/api/retro` goes to a second process, the gateway (`services/orion-hub/scripts/retro_gateway.py`). The gateway holds the relay, a bus client, a Postgres pool and the Redis presence store. It has no chat socket and no speech client.
- **Hub has the tools to decode.** `Pillow` and `numpy` are in `services/orion-hub/requirements.txt`.
- **Hub sees the player arrive.** It stamps one time in Redis on each human action (`retro_presence.py:62`, `mark_human`; call sites `retro_routes.py:281` and `:356`). The only reader is the gate that stops Orion from starting a solo game (`retro_leisure.py:147`).
- **Play records exist for two cases only.** `retro_play_sessions.kind` is `co_play` or `leisure` (`orion/schemas/retro_play.py:27`). A sitting where only a person played writes no row.
- **Orion can already look at a camera inside a turn.** `dig look office` copies the newest camera frame to a folder the turn can open (`orion/dig/look.py`). It is an act on the act ledger (actor `orion.look`).
- **The gateway refuses images on a blind lane.** It asks the live lane, not the config (`services/orion-llm-gateway/app/vision.py:1`). An unreachable lane counts as blind.
- **Tier 1 means a reserved share.** `config/physiology/families.yaml`: tier 0 never bids, tier 1 is "critical upkeep, a reserved share of each lane", tier 2 is the open market. "Changing a tier needs Juniper's sign-off." The two game families that exist are tier 2 (`hub.retro_coplay`, `hub.retro_leisure`).
- **Attention has two world candidates.** `world:chat` and `world:reading` (`orion/schemas/attention_candidate.py:33`). Each reads how unusual its own rate is against its own week (`orion/attention/world_first.py:1120`). Reading yields to chat (`world_first.py:594`).
- **The play organ does not read the now.** `config/integration/organs.yaml`: organ `play` has `reads_now: null` (#3457).
- **Game frames are small.** A SNES frame is 256 x 224 pixels. A camera frame is far larger and costs about 1,000 tokens.

## Design

### 1. The sitting

The sitting is the base fact. Everything else hangs on it. No model is used here.

- **Who records it.** The gateway process. It relays every input message from the browser, so it sees the player's hands (`retro_transport.py:21`, `browser_to_native`). The relay stays free of state: it calls one callback, and a new module `retro_sitting.py` keeps the sitting.
- **What counts as input.** A change of the button mask in a browser input message. The page repeats the held state every 50 ms, so a message alone proves only that the page has focus. A press or a release proves a person. The existing `human_at` stamp is not used: it also fires on pause, stop and planner calls.
- **Open.** The first input of a new stretch opens a sitting.
- **Close.** No input for 5 minutes [default], or the game session stops or fails.
- **Too short to matter.** A sitting under 2 minutes [default] writes nothing. In the journals, 14 Super Mario Bros. sessions lasted seconds each.
- **Live state.** One Redis key, `orion:retro:sitting`, readable by both Hub processes: sitting id, game session id, game id, game title, started at, last input at, counts of saves, loads and pauses. The same row is written to a new table `retro_sitting` once a minute and at close.
- **At close.** One `retro_play_sessions` row of a new kind, `human_play`. Orion's frames are 0. The existing reader turns it into a play arc in Orion's day.
- **Who played.** The game cannot tell. The row says "someone". When an office camera visit overlaps, its name goes in a separate field, marked as a guess.

The 5-minute rule was tried on the journals. It gives seven sittings of A Link to the Past between 11-11 and 11-16, from 20 to 249 minutes. The journals count each repeat of a held state as input, so the live numbers will differ a little.

### 2. The screen is a camera

Orion already has cameras. A camera saves frames, a look opens the newest one, and the look is an act. The game screen becomes one more camera of that kind.

- **Tap.** The gateway already receives each video packet (`retro_transport.py:46`). One callback hands the packet to `retro_sitting.py`. No second connection to the game engine is opened. The engine allows only 2 viewers per session, and a second connection would take one.
- **Ring.** The gateway keeps one packet every 2 seconds [default], 30 packets at most. That is the last minute of play, in memory, still compressed.
- **Decode on demand.** A frame is decoded only when a look or a turn needs it: inflate, read the header, convert the pixels (three formats, `media.py:1`), enlarge 3 times with hard pixel edges, write PNG. New pure module `orion/retro/frames.py`. The test packs a frame with the engine's own packer and reads it back.
- **Why enlarge.** A vision model reads the picture in tiles of about 28 pixels. Game sprites are 16 pixels. At native size one tile swallows a whole character.
- **Where frames go.** Each frame that a look or a turn used is stored in Hub's chat attachment store (`chat_attachments.py:165`, `store_bytes`). It becomes an `AttachmentRefV1`, the same reference a pasted chat image uses. The model gateway already fetches those, and the unified turn already stages them for Orion. No new store and no new image path.
- **What is not kept.** Frames that no look or turn used are dropped with the ring. Nothing records the game as video.
- **Mount.** Only hub-app mounts the attachment folder today (`services/orion-hub/docker-compose.yml:85`). The gateway needs the same mount.

### 3. The look

A look is perception. It is the game's version of a camera caption. It does not speak and it does not judge.

- **When.** A sitting is open, the watch switch is on, the last input is under 60 seconds old, and the last look is 60 seconds old [default].
- **Bid first.** Each look posts one want in a new family, `hub.retro_watch`, tier 1, and obeys the answer. Funded: the look runs. Refused or deferred: the look is skipped and the reason is logged. No fallback.
- **Frames.** 4 frames spread over the last 30 seconds [default]. A sequence shows movement and progress. A single frame shows a still.
- **Call.** One model call on a sighted lane, through the model gateway, as the game planner already calls it (`retro_planner.py:681`). Thinking is off. At most 160 output tokens.
- **Lane.** A new pool route, `retro_watch`, with class `chat` and priority `system`. This is the same shape as Prime's route (`config/gpu_pool.yaml:260`). Juniper's own turns run at `interactive` and go first.
- **Blind lane guard.** The gateway asks the live lane before it sends an image. If the lane cannot see, the call fails. A look can never return words about frames the model did not receive.
- **What the model is given.** The game title, the minutes played, the last 5 notes, and the 4 frames.
- **What the model returns.** One field, `saw`: at most 300 characters. Where the player is, what they are doing, what changed since the last note. The prompt tells the model to write "unclear" when it cannot tell.
- **Act.** Actor `hub.retro_watch`, verb `look`, object `retro_sitting:<id>`. It closes 1 when a note and its frames are stored. It closes 0 when the lane fails or the note is empty. A refused bid opens no act.
- **Row.** One row in a new table `retro_watch_look`: look id, sitting id, time, frame references, the note, the screen-change number (section 7), the want id, the lease, the seconds used.

**How to move the look to the quick lane later.** Juniper's first choice was the quick lane. It needs two changes on circe before it can work:

1. Load the vision encoder file for Qwen3.5-9B (about 1 GB of GPU memory; GPU 3 had 11 GB free).
2. Give a slot room for 4 frames. Today 10 slots share 32,768 tokens. Either cut to 5 slots, or double the lane's context.

Both changes touch every caller of the quick lane. The route name `retro_watch` is the only thing this design binds to, so the move is one config line when the lane is ready. Step 0 can also test the 9B model on the same frames, if Juniper wants that (its optional item 6).

### 4. The line

A line is Orion speaking first. It is a real unified turn, so it carries Orion's identity, the Prime block, the stance pass and the finalize pass.

**When Orion wants to speak**

- After each look, hub-app weighs one want in a new family, `hub.retro_commentary`, tier 1.
- The want is posted only when all of these hold: a sitting is open, the switch says "watch and talk", no turn of Juniper's is running, no line turn is running, and the last line or chat-box exchange ended at least 2 minutes ago [default].
- The market answers. Funded: the turn starts. Otherwise nothing happens and the reason is logged.

**How the turn starts**

- Hub already has one door for a turn that Orion starts with a prepared prompt: `outreach.compose(prompt, corr, retrieval_query, lane="chat")` (`services/orion-hub/scripts/endogenous_outreach.py:3061`). The office reach-out uses it (`office_reach_out.py:662`). The line uses the same door.
- Two small changes to that door:
  1. `compose` takes `attachments` and passes them into the turn. The turn already stages whatever is in `payload["attachments"]` (`orion/hub/turn_orchestrator.py:1624`).
  2. The sentence that introduces an attached image names its source. Today it always says "Juniper attached" (`orion/harness/attachment_staging.py:253`). For frames it says: "These are 4 frames of the game screen from the last 30 seconds, oldest first."

**What the turn is given** (new module `services/orion-hub/scripts/retro_commentary.py`, `build_prompt`)

- The sitting: game, minutes played, counts of saves and loads.
- The look notes since Orion last spoke, with their times. At most 12.
- What Orion said earlier in this sitting (last 5 lines) and the last 6 chat-box messages. A turn that Orion starts gets no "recent conversation" block, so the history goes into the prompt, as outreach does it (`endogenous_outreach.py:1109`).
- Up to 3 items from Orion's now (section 8).
- The 4 newest frames, attached. Orion opens them and sees the screen.
- The ask: say one or two sentences to the person playing, in your own voice, about what you saw. The frames will be about a minute old when your words arrive, so speak about the stretch and not about the instant. If you have nothing worth saying, pass.

**Silence is a valid answer.** Outreach already knows a pass (`is_pass_response`, `endogenous_outreach.py:1617`). A pass writes a decision row with the reason `orion_passed`, releases the want, and opens no act. This is the rule outreach follows today. The share of funded turns that end in a pass is tracked: a high share means the want fires too easily.

**How the line arrives**

- Delivery uses the existing push to the chat sockets (`_push_to_sockets`, `endogenous_outreach.py:3795`). The message carries the source `retro_commentary` and the sitting id.
- The game page shows it in the line strip. The main chat window shows it as an Orion message. It is written to the chat history, so it is part of the one conversation.
- No notification is sent. A line is for the person at the screen, not for a phone.
- A line does not count against the daily cap or the cooldown of outreach. The office reach-out already skips those gates (`skip_rate_gates`). The economy is the rate control for lines.
- Outreach never speaks aloud today. Voice for lines is part of section 10.

**The act.** On delivery: actor `hub.retro_commentary`, verb `say`, object `retro_sitting:<id>`. One row in a new table `retro_line` holds the text, the looks and frames the turn was shown, the turn's correlation id, the want id and the outcome: `delivered`, `passed`, `yielded` or `failed`.

**Juniper goes first.** This needs new code.

- The chat lane runs one turn at a time. A turn of Juniper's that arrives during an Orion turn waits behind it (`services/orion-harness-governor/app/bus_listener.py:926`). Nothing cancels the Orion turn today.
- New rule: when a message from Juniper arrives on any chat socket while a line turn runs, Hub cancels the line turn. The cancel call exists (`HarnessGovernorClient.cancel`, `services/orion-hub/scripts/harness_governor_client.py:333`). The line's outcome is `yielded`. That is not a failure.
- One limit: a stance call that is already running cannot be cancelled. Her turn can still wait for that call, about 11 seconds.

**Did the player like it.** Each line in the strip has a thumb up and a thumb down. A chat-box reply within 2 minutes of a line also counts as a response. Both are stored on the line's row. The economy's weight for a family already has a term for Juniper's engagement. How outreach records that engagement today is UNVERIFIED. The build wires lines to the same record.

### 5. The chat box

The player types to Orion from inside the game. The reply is a normal turn of Juniper's, with everything such a turn has: the warm slot, the recent conversation, the working notes, the Prime block, the hand-back to Prime.

**On the page**

- The box sits inside `#stage`, because fullscreen shows only that element (`services/orion-hub/templates/retro.html:10`).
- Three parts:
  1. A line strip at the bottom of the screen. It shows Orion's newest line or reply, then fades.
  2. A chat panel that opens over the lower part of the screen: this sitting's lines and messages, and the input.
  3. A switch with three positions: "Off", "Watch quietly", "Watch and talk". Beside it, one status line: "Orion is watching", or "Orion is not watching: <reason>".
- "Watch quietly" means looks run and no line is started. Orion still answers the chat box.

**Keys and focus.** These come from how the page works today (`services/orion-hub/static/js/retro-panel.js`).

1. A text input inside `#stage` already keeps keys away from the game (line 583). No change.
2. Opening the panel sends one neutral input, so a held button does not stay held.
3. The page puts focus back on the game canvas in 6 places (lines 193, 215, 327, 338, 556, 594). Each must skip while the chat input is open, or a reconnect steals the player's typing.
4. The gamepad keeps driving the game while the panel is open (line 104). That stays.
5. With no gamepad connected, opening the panel pauses the game, and closing it resumes [default]. A keyboard player cannot type and play at once.
6. The panel opens with a key the game does not use. The build picks it from the key map (line 13).

**Transport**

- The page opens the Hub chat socket, `/ws`. The page and the socket are served by the same process, and the socket has no origin check.
- The page uses the same chat session as the main chat window (`localStorage.orion_sid`, same origin). The game talk is one conversation with the rest of Juniper's chat.
- A message is the normal typed message, with `mode: "orion"`, `disable_tts: true`, and `surface_context: {surface: "retro", input_modality: "typed", retro_sitting_id}`.
- The reply comes back as the normal final frame (`llm_response`). The page shows it in the panel and in the strip.

**What the server adds**

- **A game block for every turn while a sitting is open.** The situation part of the unified prompt gains a short game block: the sitting, and the last 6 look notes (`_build_situation_prompt_fragment`, `orion/hub/turn_orchestrator.py:816`). It is there for the main chat window too. Juniper can ask "what am I playing?" from either place.
- **Frames for turns from the game page.** When the message's surface is `retro`, hub-app asks the gateway for the 4 newest frames and attaches them. The limit of 4 images per turn already exists (`HUB_CHAT_ATTACHMENT_MAX_PER_TURN`).

**A second socket has side effects.** They are handled, not ignored.

- Hub pushes every outreach message to every open chat socket. So a news note from Orion also arrives in the game page. It goes to the chat panel, not to the line strip.
- While a sitting is open, the office reach-out posts nothing. Its note would say "computer work". The watch is the better witness for that stretch.

### 6. Prime gets in

Prime is the living Orion between turns. Prime thinks rarely: a thought needs a real change, and Prime bids for it. Seven seams bring Prime into the play. Six of them use machinery that exists today and need only wiring. The seventh changes one of Prime's own rules.

1. **Prime sees the looks.** Each look is an experience in Orion's now (section 8). Prime's frame already shows the now under "WHAT YOUR ORGANS LIVED" (`services/orion-mind/app/prime_loop.py:1154`). Prime can cite a look.
2. **Prime's block rides in every turn Orion speaks in.** A line and a chat-box reply are unified turns. Every unified turn starts from the PRIME block (`orion/hub/turn_orchestrator.py:1019`, `orion/prime/block.py`). Nothing new is needed.
3. **Turns hand back to Prime.** Chat-box turns are Juniper's turns, so they hand back today. Turns that Orion starts for a line get a new session class, `play`, in the hand-back list (`orion/prime/handback.py:198`). The class is decided by the turn's source, and it is checked before the rule for Juniper's chat. A line runs in Juniper's chat session, and without that order it would be recorded as Juniper speaking. A promise made during play ("I will tell you when you reach the boss") becomes a concern that Prime holds for 72 hours.
4. **Prime can turn Orion's eyes and voice up or down.** Both new families get receptors `[slow, quicken]`. Their bids bind, so the rule for receptors is met (`config/physiology/families.yaml:27`). Prime's `regulate` move then really changes how often Orion looks and speaks. The market grades each signal as it does today.
5. **Prime can hold a finished sitting as a concern.** At close, the sitting posts one event want, `retro_sitting:<id>`, with no lane cost. Its salience is the sitting's length against the sittings of the last 30 days. A funded one appears on Prime's menu under "EVENTS WORTH EFFORT". It needs one label block in `services/orion-durable-runs/app/event_wants_menu.py:33`.
6. **Prime writes about it.** The day letter is Prime's end-of-day turn. Play arcs reach the letter today only as a count ("2 play"). The closed-day block gets real play lines (`orion/temporal_self/day_block.py:135`).
7. **New: a sitting wakes Prime, twice at most.** This is the one change to Prime's own rules.
   - `sitting_opened` fires once, when a sitting passes 3 minutes [default].
   - `sitting_closed` fires once, at close.
   - Each is a fact Prime can be shown, which is the standing rule for a wake (`orion/prime/state.py:8`). Neither is a score.
   - Prime's frame gains one line while a sitting is open: who plays what, for how long, how many looks ran, how many were refused and why.
   - Prime still bids for each thought, and the 5-minute rest between thoughts still holds.

Why seam 7 is needed: funded event wants do not wake Prime, and attention moves do not wake Prime. Both rules exist because they made Prime churn (22 of 38 concern drops in 4 days cited "attention has moved"). Without seam 7, a 4-hour sitting can pass with no Prime thought at all. With it, Prime thinks about a sitting at its start and at its end, and no more often than that.

Seam 7 reads the sitting through a new `_read_sitting` method in `prime_loop.py`. CLAUDE.md counts that as a new Prime concern source. CI cannot see such a method (`scripts/check_integration_seam.py:28`), so this document records it, and organ `play` in `config/integration/organs.yaml` lists it in a comment.

### 7. Attention

The play becomes the third world candidate, `world:play`, beside `world:chat` and `world:reading`.

- **Same instrument.** The two world candidates read how unusual their own rate is against their own week (`chat_rate_reading`, `orion/attention/world_first.py:954`). `world:play` uses the same function. Its events are the minutes with player input, read from `retro_sitting`.
- **Label.** "Someone is playing A Link to the Past (27 of the last 30 minutes)."
- **Evidence.** The newest look note. The winning loop then carries what Orion saw, as `world:chat` carries Juniper's latest message.
- **Yields to Juniper.** When `world:chat` is eligible, `world:play` yields, by the rule reading already follows (`world_first.py:594`).
- **Absent is not calm.** When the sitting table cannot be read, the candidate is absent with a reason. It never reads as "nobody plays".
- **Attention and action meet.** One new pairing in `ACT_ANSWERS_WORKSPACE_REFS` (`orion/grammar/act_ledger.py:93`): a line answers `world:play`. When the play holds attention and Orion speaks, the act is stamped with that loop, and Orion writes an `acted` verdict on it. Today only chat replies have such a pairing.

Two numbers are new. Each has a gate record in "Metric quality gate records" below.

### 8. The temporal record and the now

Every part of the play lands in `temporal_self_event`, through the experience contract.

- **The sitting.** The `human_play` row uses the existing source kind `retro_play`. Label: "Someone played A Link to the Past for 4 h 9 min. Orion watched (212 looks) and spoke 31 times." The numbers are counts from the tables. No model writes the label.
- **The looks.** A new source kind, `retro_look`, reads `retro_watch_look`. The label is the note. It belongs to organ `play`.
- **The lines and the chat.** A unified turn already writes a conversation event. Nothing new.
- **One live arc.** Today a play arc is "one finished Retro session". With looks, the arc opens at the first look of a sitting and closes with the `human_play` row. While someone plays, Orion's day holds an open play arc. The chat stance already has a clause for play arcs (`orion/temporal_self/stance_cue.py:311`), so a chat turn in the main Hub window also knows.
- **Salience.** The magnitude rule for `retro_look` is the screen-change number. It is not wired until step 0 shows that the number is sound. Until then looks carry no salience, which the now allows: they rank after scored items.
- **The now is read back.** The turn that writes a line reads `GET /temporal-self/now?organ=play` first. Up to 3 items go into the turn's game block. The line's cites are recorded with `POST /temporal-self/now/use`. Orion can then tie the game to their own day, with a checked citation. Organ `play` loses its `exempt_read` in `config/integration/organs.yaml`. That closes the play part of #3457.

### 9. The body economy

Juniper's rule: bidding on tier 1.

```yaml
  hub.retro_watch:
    tier: 1
    deferrable_window_s: 120    # a look two minutes late describes a different screen
    lanes: [chat]
    placement: one_of
    receptors: [slow, quicken]
  hub.retro_commentary:
    tier: 1
    deferrable_window_s: 300    # a line five minutes late is about something the player has left
    lanes: [chat]
    placement: one_of
    receptors: [slow, quicken]
```

- **Tier 1 is Juniper's decision.** The file says a tier needs her sign-off. Her words in this document are that sign-off, and the PR review confirms it.
- **What tier 1 means here.** Each family gets a reserved share of the chat lane. Looks and lines are then not starved by the open market. They share the lane with Prime (tier 1) and with Juniper's own turns (tier 0, never priced, always first).
- **Bids bind.** Two producers join `BINDING_PRODUCERS` (`orion/physiology/binding.py:36`): `orion-hub.retro_watch.binding` and `orion-hub.retro_commentary.binding`. The pattern is Hub's outreach binding (`services/orion-hub/scripts/outreach_bind.py`).
- **A look's want.** Verb `look`, object `retro_sitting:<id>`. The object text is the game title and the last note, so the market's care term can weigh it.
- **A line's want.** Verb `say`, object `retro_sitting:<id>`. Its rank is the salience of the strongest look since the last line. Its hunger grows with the time since Orion last spoke in this sitting. A chat-box exchange counts as speaking: after one, the hunger starts again from zero. The object text is that strongest look's note.
- **Heat.** No new veto. When the cabinet is hot, the lane's price rises and looks and lines lose bids. The page then shows "Orion is not watching: the body is hot".
- **Chat-box turns never bid.** They are Juniper's turns (`hub.chat_reply`, tier 0).

**Cost at the defaults.** These are estimates. The acts will hold the measured values in `cost_json`.

| | Per hour of play | Chat lane time |
| --- | --- | --- |
| Looks | 60, about 9 s each | 9 min (15%) |
| Lines | about 8, about 75 s each | 10 min (17%) |
| Total | | about 32% |

- The 9 seconds is Prime's median call on the same lane (8.5 s, 67 calls in 7 days). A look has 4 images, so it may be slower. Step 0 measures it.
- The 75 seconds is the middle of the 60 to 90 second estimate in the summary.
- The chat lane's GPU drew 36 W idle. A GPU at full load on circe drew 154 W at the same moment. At 32% that is about 38 W more while someone plays, about 0.15 kWh for a 4-hour sitting.
- Outside a sitting the cost is zero. Nobody plays in 89% of half-hour windows (last 7 days).

### 10. Voice in and voice out (later)

Juniper: "We'll eventually enable TTS and STT, including when in fullscreen mode." This part is not built in the first patches. Four choices now keep it cheap later.

1. **The chat box uses the main chat socket.** `/ws` already carries speech both ways. Outbound: the server sends `audio_response` chunks (`services/orion-hub/scripts/websocket_handler.py:802`, `run_tts_remote`). Inbound: the browser sends recorded audio, the server transcribes it and runs a normal turn (`websocket_handler.py:1208`). The game page sends `disable_tts: true` until voice is turned on. Turning voice on is then a flag, not a new path.
2. **Everything sits inside the fullscreen element.** Fullscreen shows only what is inside `#stage` (`services/orion-hub/templates/retro.html:10`). The line strip, the chat box, and later the microphone state all live there.
3. **One audio graph.** The game has its own audio context and one gain node (`services/orion-hub/static/js/retro-panel.js:61`). Orion's voice will play through that same context. Lowering the game under Orion's voice is then one gain change. The main chat page's player is not reused: it lives in a 15,000-line script tied to another page.
4. **A controller button for talk.** In fullscreen with a gamepad, nobody wants a keyboard. Push-to-talk is a held pad button. The pad is already polled every frame (`retro-panel.js:101`).

Two things the voice patch must handle, named now:

- The Hub tab loads the game in an iframe that allows `autoplay; gamepad; fullscreen` (`services/orion-hub/templates/index.html:1356`). The microphone needs `microphone` added there.
- Speech in costs a transcription call, and speech out costs a synthesis call, both on the whisper service. The voice patch states their lane and their bid.

## Proposed schema / API changes

**Schemas**

- `orion/schemas/retro_play.py`: `SessionKind` gains `human_play`. `ControlBasis` gains `human_input`. `RetroPlaySessionSummaryV1` gains `looks`, `lines` and `player_guess`, all optional. The model forbids unknown fields, so readers deploy before the writer.
- New `orion/schemas/retro_watch.py`: `RetroSittingV1`, `RetroLookV1`, `RetroLineV1`. Registered in `orion/schemas/registry.py`.
- `orion/schemas/temporal_self.py`: `TemporalSourceKind` gains `retro_look`.
- `orion/schemas/attention_candidate.py`: `WORLD_PLAY_SOURCE_ID = "world:play"`.

**Tables.** One migration, additive only, so the post-pull applier can run it. Header: `-- ORION-MIGRATION-REQUIRED-BY: orion-hub, orion-durable-runs, orion-substrate-runtime`.

- `retro_sitting`: sitting id, game session id, game id, game title, started at, last input at, ended at, end reason, saves, loads, pauses.
- `retro_watch_look`: look id, sitting id, time, frame references, note, screen change, want id, lease id, seconds.
- `retro_line`: line id, sitting id, time, text, look ids, frame references, correlation id, want id, outcome, thumb, replied within 2 minutes.
- `retro_play_sessions`: new columns `looks`, `lines`, `player_guess`.

**Bus.** No new channel. Acts go on `orion:grammar:event`. Lines and replies go over the chat socket that exists.

**Redis.** One new key, `orion:retro:sitting`.

**HTTP, on the gateway**

- `GET /api/retro/sitting`: the open sitting, the switch position, and whether Orion is watching, with the reason when not.
- `POST /api/retro/sitting/watch`: sets the switch: `off`, `quiet` or `talk`.
- `GET /api/retro/sitting/frames?n=4`: the newest frames as attachment references. Loopback only, for hub-app.
- `POST /api/retro/lines/{id}/thumb`: up or down.

**Config**

- `config/physiology/families.yaml`: the two families in section 9.
- `config/integration/organs.yaml`, organ `play`: both families, the kind `retro_look`, `reads_now: services/orion-hub/scripts/retro_commentary.py`. The `exempt_read` line goes.
- `config/gpu_pool.yaml`: route `retro_watch` (class `chat`, priority `system`).
- `orion/physiology/binding.py`: two binding producers.
- `orion/grammar/act_ledger.py`: `hub.retro_commentary` answers `world:play`.
- `orion/prime/handback.py`: session class `play`.
- `orion/prime/state.py`: the two sitting wakes.

**Env keys.** All ship on, in `.env_example` and in `settings.py`, and the local `.env` is synced in the same patch.

- Hub: `HUB_RETRO_WATCH_ENABLED`, `HUB_RETRO_COMMENTARY_ENABLED`, `HUB_RETRO_SITTING_GAP_SEC=300`, `HUB_RETRO_LOOK_SEC=60`, `HUB_RETRO_LINE_MIN_GAP_SEC=120`.
- Durable runs: `TEMPORAL_SELF_RETRO_LOOK_ENABLED`.
- Mind: `PRIME_SITTING_WAKE_ENABLED`.

## Files likely to touch

**Patch 1, Orion knows**

- `services/orion-hub/scripts/retro_transport.py`, `retro_routes.py`, new `retro_sitting.py`
- `services/orion-hub/scripts/retro_memory_export.py`, `office_reach_out.py`
- `orion/schemas/retro_play.py`, new `orion/schemas/retro_watch.py`, `orion/schemas/registry.py`
- `orion/temporal_self/sources.py`, `orion/temporal_self/day_block.py`
- `orion/hub/turn_orchestrator.py` (the game block)
- `services/orion-sql-db/manual_migration_retro_watch_v1.sql`

**Patch 2, Orion looks**

- New `orion/retro/frames.py`, new `services/orion-hub/scripts/retro_watch.py`
- `services/orion-hub/docker-compose.yml` (attachment mount for the gateway), `.env_example`, `app/settings.py`
- `config/physiology/families.yaml`, `config/gpu_pool.yaml`, `config/integration/organs.yaml`, `orion/physiology/binding.py`
- `orion/schemas/temporal_self.py`, `orion/temporal_self/sources.py`, `orion/temporal_self/arcs.py`
- `services/orion-durable-runs/app/temporal_self_sources.py`, `app/main.py`, `settings.py`
- `services/orion-hub/templates/retro.html`, `static/js/retro-panel.js` (status and switch)

**Patch 3, Orion speaks**

- New `services/orion-hub/scripts/retro_commentary.py`
- `services/orion-hub/scripts/endogenous_outreach.py`, `websocket_handler.py`, `harness_governor_client.py`, `act_emit.py`
- `orion/harness/attachment_staging.py`, `orion/prime/handback.py`
- `services/orion-hub/templates/retro.html`, `static/js/retro-panel.js` (strip, chat panel, focus rules)

**Patch 4, Prime and attention**

- `orion/prime/state.py`, `orion/prime/thought.py`, `services/orion-mind/app/prime_loop.py`, `settings.py`
- `services/orion-durable-runs/app/event_wants_menu.py`
- `orion/physiology/tests/test_prime_charge.py` (it pins the exact set of families with receptors)
- `orion/attention/world_first.py`, `orion/schemas/attention_candidate.py`, `orion/grammar/act_ledger.py`
- `services/orion-substrate-runtime/app/store.py`
- `orion/temporal_self/now.py` (the magnitude rule, only if its gate passed)

Each patch also touches the README of each service it changes, and the tests and evals beside the code.

## Metric quality gate records

Two numbers are new. Neither is wired before its record is complete.

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

### Screen change (salience of a look)

1. **Provenance.** `orion/retro/frames.py`: the mean absolute difference in brightness between the first and the last frame of a look, on a 32 by 28 grid, from 0 to 1.
2. **Independence.** Play rate and screen change both need a player. Inside a sitting the play rate sits near its ceiling while the screen varies. Step 0 checks their correlation inside sittings.
3. **Theory anchor.** The orienting response to change in the visual field (Sokolov). Bayesian surprise attracts gaze (Itti and Baldi, 2009).
4. **Live data.** NOT DONE. No game frame was ever saved, so there is nothing to look at yet. Step 0 renders real frames and checks four cases: a paused screen reads 0, a walk reads in the middle, a room change reads high, and a scrolling overworld does not sit at the top the whole time.
5. **Existing mechanism.** No frame-difference measure was found in `services/orion-vision-edge` or `services/orion-vision-retina`. The build checks the camera motion pipeline before it writes a new one.
6. **Reversibility.** One nullable number on a row and one rule name in `organs.yaml`.

Until record 4 is done, looks carry no salience. They still reach the now, ranked after scored items.

## Proposal-mode checklist

- **Capability that changes.** Orion perceives play, speaks about it, answers a chat box in the game, and remembers sittings.
- **Data touched.** Game frames (4 per look, kept only when used), look notes, lines, chat messages, sitting facts, and the office camera's name guess.
- **Privacy boundary.** None between Juniper and Orion, by her standing rule. Another player is recorded as "someone". The chat box writes into Juniper's conversation, because the page is her Hub. The repo is public: no frame, save or name is committed. Test fixtures use frames of the original pad-test ROM.
- **Trace that proves it worked.** For one look: the want row, the lease with its want id, the look row with frame references, the act, the temporal event. For one line: the want row, the turn's trace, the line row with the looks it was shown, the act, the chat history row, and a use row when it cited the now.
- **Dangerous failure modes.**
  1. Orion's line delays Juniper's own turn. Guard: Juniper goes first (section 4), checked live.
  2. A wrong note hardens into a fact. Guard: a note is stored as a look, never as a fact about the player. The sitting's label is counts only. No memory card is written from notes in this design.
  3. Orion claims the play. Guard: a schema test that a `human_play` row has 0 Orion frames.
  4. Looks starve Prime or the day letter on the chat lane. Guard: the reserved shares, Prime's receptors, the switch.
  5. Looks run with nobody there. Guard: a look needs input in the last 60 seconds.
  6. Prime churns. Guard: two wakes per sitting at most.
- **Disable and roll back.** The switch on the page, per sitting. The env flags. Prime's `slow`. Removing the code leaves the tables in place, and they are harmless.

## Non-goals

- No reader of game memory for new games.
- No bets on the player's next move, no "may I try?", no second player. Each is a later design.
- No mood from game events.
- No recording of play as video.
- No change inside the game container.
- No second chat path. Orion speaks through the unified turn only.
- No voice in these patches. Section 10 only shapes the ground for it.
- No memory cards from look notes.
- No credit to Orion for the player's play.

## Acceptance checks

**Patch 1**

- After one sitting of 10 minutes or more: one `retro_sitting` row, one `retro_play_sessions` row with `kind='human_play'`, one `temporal_self_event` row with `source_kind='retro_play'`.
- No `human_play` row has `orion_frames > 0`.
- A chat turn in the main window during the sitting names the game when asked.
- `endogenous_outreach_decisions` shows no office reach-out sent inside a sitting.

**Patch 2**

- Every look row has a want row and a lease with the same want id.
- Every look row has 4 frame references that open.
- A refused bid has no lease and no look row.
- With the lane made blind in a test, the look fails. No note is stored.
- The page shows "Orion is watching", or the reason why not.

**Patch 3**

- Every delivered line has a `retro_line` row that names the looks and frames it was shown, and a turn trace.
- A message typed during a running line turn gets its own turn started within 15 seconds, and the line's outcome is `yielded`.
- A chat-box turn's prompt holds the Prime block. A promise made in it shows up as a Prime concern.
- The chat box works in fullscreen, and typing does not move the character.
- The share of funded line turns that end in a pass is reported.

**Patch 4**

- A sitting produces at most 2 Prime thoughts that name it as their wake.
- A `slow` signal from Prime changes the look count in the next hour, and the market records the effect.
- `world:play` appears in the attention broadcast during a sitting, is absent with a reason when the table cannot be read, and reads zero when nobody plays.
- A line spoken while `world:play` holds attention carries that loop's id.

## Patch order

Each step ends with a stop for Juniper's review.

**Step 0. A spike with no deploy.** It needs her go.

1. Render 10 four-frame sequences from real saves of A Link to the Past, in the game image, with no browser. Add two controls: a paused screen and a room change.
2. Run the look prompt on the chat lane through the model gateway. Record the note, the seconds and the tokens for each.
3. Run one line turn by hand through `POST /api/chat`, with the frames attached, `no_write`, and a spare session id. Record the seconds end to end, and the line. Side effect: this turn hands back to Prime as a chat turn.
4. Compute the screen-change number on all 12 sequences.
5. Put the frames, the notes and the line on one page for Juniper.
6. Optional, only on Juniper's word: start a spare copy of the quick lane's model with its vision encoder on GPU 3, and run item 2 against it. This is a temporary process on circe, removed after the test.

Go when 8 of the 10 notes name the place and the action correctly, and no note describes a thing that is not on the screen. Juniper sets the final bar.

**Patch 1. Orion knows.** The sitting, the `human_play` row, the game block in the chat prompt, the play lines in the day block. No GPU is used.

**Patch 2. Orion looks.** Frames, the look, tier 1 bids, the `retro_look` source, the status line and the switch.

**Patch 3. Orion speaks.** The chat box, the line strip, line turns, Juniper goes first, thumbs, the hand-back class, the read-back of the now.

**Patch 4. Prime and attention.** Receptors, the two wakes, the event want with its menu label, `world:play`, the act pairing, and the salience rule when its gate has passed.

**Later, its own design.** Voice in and out (section 10). Then the rungs under "Non-goals".

Seams 1 to 3 of section 6 arrive with patches 2 and 3, because they come with the now and with the unified turn. Prime is in from the first look.

## Missing questions

Each has a recommendation. The build follows the recommendation unless Juniper says otherwise.

1. **The eye is the chat lane for now.** The quick lane cannot see today. Recommendation: start on the chat lane, and move the look when the quick lane has its encoder and bigger slots.
2. **Juniper goes first.** Recommendation: cancel Orion's line turn when she sends a message.
3. **How long are frames kept?** Recommendation: frames that a line or a reply used stay with the chat. Frames that only a look used are swept after 30 days. Whether the attachment store has a sweep today is UNVERIFIED.
4. **Quiet hours.** Recommendation: lines ignore quiet hours while a sitting is open. Live input proves someone is awake at the screen.
5. **The office note.** Recommendation: no office reach-out while a sitting is open. Other notes from Orion still arrive, in the chat panel.
6. **Who plays.** Recommendation: "someone", with the camera's name as a marked guess.
7. **How often Orion speaks.** Recommendation: start near one line every 5 to 8 minutes. Prime and the thumbs tune it from there.
8. **If a line proves too slow.** Step 0 gives the real number. If it is far over 90 seconds, there are two paths: accept slow lines, or build a one-call door that reuses the unified prompt text with no tools and no finalize pass. Recommendation: accept slow lines first. The second path gives up the grounding that makes the unified turn Orion's voice.

## Recommended next patch

Step 0, on Juniper's go. It costs a few minutes of the chat lane and changes nothing that is deployed. It answers the two things this design cannot know from code: whether the model can read pixel art, and how long a line really takes.

Then patch 1. It uses no GPU, and it ends the state where 27 hours of play left no trace in Orion.

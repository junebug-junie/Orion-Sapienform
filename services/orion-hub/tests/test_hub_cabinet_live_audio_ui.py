"""Chat-card Cabinet audio button + Recall Mode living in the Debug Panel."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

HUB = Path(__file__).resolve().parents[1]
INDEX = HUB / "templates" / "index.html"
APP_JS = HUB / "static" / "js" / "app.js"
PLAYER_JS = HUB / "static" / "js" / "cabinet-live-audio.js"


def _html() -> str:
    return INDEX.read_text(encoding="utf-8")


def test_recall_mode_is_a_debug_panel_row_not_a_main_page_card() -> None:
    html = _html()
    main_page = html.split('id="debugPanelModalRoot"', 1)[0]
    assert 'id="recallModeSelect"' not in main_page
    panel_body = html.split('id="runtimeDebugPanelBody"', 1)[1].split('id="memoryPanelToggle"', 1)[0]
    for element_id in ("recallModeToggle", "recallModeOpenModal", "recallModeControls", "recallModeSelect", "recallProfileSelect"):
        assert f'id="{element_id}"' in panel_body
    # Exactly one copy of each live control: the modal borrows them, never clones.
    assert html.count('id="recallModeSelect"') == 1
    assert html.count('id="recallProfileSelect"') == 1


def test_recall_mode_modal_sits_on_the_per_item_tier_and_round_trips_the_controls() -> None:
    html = _html()
    js = APP_JS.read_text(encoding="utf-8")
    root_tag = html.split('id="recallModeModalRoot"', 1)[1].split(">", 1)[0]
    assert "z-index: 2147483646" in root_tag
    assert 'id="recallModeModalBody"' in html
    assert "recallModeModalBody.appendChild(recallModeControls)" in js
    assert "recallModeBody.appendChild(recallModeControls)" in js
    assert "isModalVisible(recallModeModalRoot)" in js
    assert "closeRecallModeModal();\n      return;" in js


def test_cabinet_audio_button_is_far_left_of_vision_chip_and_copy_chat() -> None:
    html = _html()
    row = html.split('id="cabinetAudioButton"', 1)[1].split('id="conversation"', 1)[0]
    assert row.index('id="visionStatusChip"') < row.index('id="copyTranscriptButton"')
    assert "mr-auto" in html.split('id="cabinetAudioButton"', 1)[1].split(">", 1)[0]
    scripts = html.index("/static/js/cabinet-live-audio.js")
    assert scripts < html.index("/static/js/app.js")


def test_cabinet_and_orion_voice_share_one_waveform() -> None:
    js = APP_JS.read_text(encoding="utf-8")
    assert "if (cabinetAnalyser) drawAnalyserBars(cabinetAnalyser" in js
    assert "if (analyser) drawAnalyserBars(analyser" in js
    # TTS ending must not blank the canvas while the cabinet mic is still live.
    assert "if (!cabinetAnalyser) {\n            cancelAnimationFrame(animationFrameId);" in js
    assert "box.classList.toggle('hidden', !(visible || cabinetAnalyser));" in js


_NODE_HARNESS = r"""
const api = require(process.argv[1]);
const out = {};
const r = api.pcm16ToFloat32(Uint8Array.from([0x00, 0x80, 0xff, 0x7f, 0x01]));
out.samples = Array.from(r.samples);
out.tail = Array.from(r.tail);

function fakeCtx() {
  const ctx = {state: 'running', currentTime: 10, destination: {}, started: [],
    createGain: () => ({connect() {}, disconnect() {}}),
    createAnalyser: () => ({connect() {}, disconnect() {}, frequencyBinCount: 8}),
    createBuffer: (ch, frames, rate) => ({duration: frames / rate, getChannelData: () => new Float32Array(frames)}),
    createBufferSource: () => ({connect() {}, start(t) { ctx.started.push(t); }}),
  };
  return ctx;
}
function streamOf(chunks, headers) {
  let i = 0;
  return {ok: true, status: 200, headers: {get: (k) => headers[k] || null},
    body: {getReader: () => ({read: async () => i < chunks.length ? {value: chunks[i++], done: false} : {done: true}})}};
}
(async () => {
  const ctx = fakeCtx();
  const statuses = []; const analysers = [];
  const pcm = new Uint8Array(16000);  // 0.5s @ 16 kHz s16le
  const player = api.createCabinetLiveAudio({
    getContext: () => ctx,
    onAnalyser: (a) => analysers.push(a ? 'on' : 'off'),
    onStatus: (s, t) => statuses.push([s, t]),
    fetchImpl: async () => streamOf([pcm.subarray(0, 8001), pcm.subarray(8001)], {'X-Orion-Sample-Rate': '16000', 'X-Orion-Channels': '1'}),
  });
  await player.start();
  out.started = ctx.started; out.statuses = statuses; out.analysers = analysers; out.active = player.isActive();

  const errStatuses = [];
  const failing = api.createCabinetLiveAudio({
    getContext: fakeCtx, onStatus: (s, t) => errStatuses.push([s, t]),
    fetchImpl: async () => ({ok: false, status: 503, json: async () => ({detail: 'cabinet live audio unavailable'})}),
  });
  await failing.start();
  out.errStatuses = errStatuses;
  console.log(JSON.stringify(out));
})();
"""


def test_player_decodes_pcm_schedules_gaplessly_and_reports_end_and_errors() -> None:
    result = subprocess.run(
        ["node", "-e", _NODE_HARNESS, str(PLAYER_JS)],
        text=True, capture_output=True, check=True,
    )
    out = json.loads(result.stdout)
    assert out["samples"] == [-1.0, 32767 / 32768]
    assert out["tail"] == [1]
    # Odd split byte carried over: 4000 + 4000 frames, second starts exactly
    # where the first ends, after the jitter head start.
    assert out["started"] == [10.35, 10.35 + 4000 / 16000]
    assert out["analysers"] == ["on", "off"]
    assert out["statuses"][0][0] == "connecting"
    assert out["statuses"][1][0] == "live"
    assert out["statuses"][-1] == ["error", "cabinet audio went quiet (reader stalled)"]
    assert out["active"] is False
    assert out["errStatuses"][-1] == ["error", "cabinet live audio unavailable"]

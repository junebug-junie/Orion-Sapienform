"""The play panel is in the rendered template and wired in the linked asset (browser eval:
evals/retro_play_panel.py exercises the interaction against a fixture Retro)."""
from pathlib import Path

HUB = Path(__file__).resolve().parents[2]


def test_template_has_play_panel_and_current_asset():
    html = (HUB / 'templates/retro.html').read_text()
    for marker in ('id="play"', 'id="atlas"', 'id="leisure"', 'id="zgoal"', 'id="zprediction"',
                   'id="verdicts"', 'id="takeover"', '/static/js/retro-panel.js?v=11'):
        assert marker in html, marker


def test_asset_reads_leisure_atlas_and_shed_and_watches_without_a_controller():
    js = (HUB / 'static/js/retro-panel.js').read_text()
    for marker in ("api('leisure')", '/atlas`', "'gpu_pool_unavailable'", 'retry_after_sec',
                   'function watch()', "'Orion is waiting: '", "'zelda-navigator-v1'", 'last_finished'):
        assert marker in js, marker
    watch = js[js.index('function watch()'):js.index('let atlasAt')]
    assert '/input' not in watch and 'grant(' not in watch   # watching never takes a controller


def test_blur_keeps_a_handoff_but_a_hidden_tab_ends_it():
    # Retro cancels a grant whose sponsor socket closes, so a blur must not end Orion's turn
    # (2026-10-23: a co-play handoff died within a second of the window losing focus).
    js = (HUB / 'static/js/retro-panel.js').read_text()
    assert "addEventListener('blur',()=>{away=true;if(!scriptedMode)neutral();})" in js
    assert "if(document.hidden){away=true;neutral();}" in js   # nobody can watch a hidden tab
    # Once the handed-off grant is gone, the page leaves handoff mode.
    assert '!mine || mine.epoch!==handoffGrant.epoch' in js


def test_policy_poll_is_fast_while_orion_holds_the_controller():
    # 2026-10-24: a 1 s poll added up to 1 s to every decision while Link stood still.
    js = (HUB / 'static/js/retro-panel.js').read_text()
    assert "if(!(scriptedMode && cognitivePort!==null) && now-lastPolicyPoll<1000)return;" in js
    assert "finally{pollingPolicy=false;}},250);" in js


def test_gamepad_uses_the_nes_layout_and_skips_empty_or_nonstandard_pads():
    # 8BitDo Pro 2 (and the standard mapping) is positional: right face button = NES A,
    # bottom = NES B. The old mapping had them swapped (Xbox layout).
    import json, re, shutil, subprocess
    import pytest
    js = (HUB / 'static/js/retro-panel.js').read_text()
    html = (HUB / 'templates/retro.html').read_text()
    assert 'id="padInfo"' in html and "addEventListener('gamepadconnected',padStatus)" in js
    if not shutil.which('node'):
        pytest.skip('node not installed')
    body = re.search(r'function padMask\(pad\) \{.*?\n  \}', js, re.S).group(0)
    pick = re.search(r'function gamepad\(\) \{.*?\n  \}', js, re.S).group(0)
    probe = f'''{body}
{pick}
const pad=(pressed,axes=[0,0])=>({{id:'8BitDo Pro 2',connected:true,mapping:'standard',axes,
  buttons:Array.from({{length:16}},(_,i)=>({{pressed:pressed.includes(i)}}))}});
const el=()=>({{value:'0'}});
const out={{A:padMask(pad([1])),B:padMask(pad([0])),start:padMask(pad([9])),select:padMask(pad([8])),
  stickLeftUp:padMask(pad([],[-1,-1])),dpadRight:padMask(pad([15]))}};
Object.defineProperty(globalThis,'navigator',{{configurable:true,
  value:{{getGamepads:()=>[null,{{...pad([]),mapping:''}},pad([1])]}}}});   // Node 21+ has a read-only navigator
out.picked=padMask(gamepad());
console.log(JSON.stringify(out));'''
    run = subprocess.run(['node', '-e', probe], capture_output=True, text=True)
    assert run.returncode == 0, run.stderr
    got = json.loads(run.stdout)
    assert got == dict(A=256, B=1, start=8, select=4, stickLeftUp=64 | 16, dpadRight=128, picked=256)

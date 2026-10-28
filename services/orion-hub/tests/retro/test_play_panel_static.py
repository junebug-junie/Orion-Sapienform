"""The play panel is in the rendered template and wired in the linked asset (browser eval:
evals/retro_play_panel.py exercises the interaction against a fixture Retro)."""
from pathlib import Path

HUB = Path(__file__).resolve().parents[2]


def test_template_has_play_panel_and_current_asset():
    html = (HUB / 'templates/retro.html').read_text()
    for marker in ('id="play"', 'id="atlas"', 'id="leisure"', 'id="zgoal"', 'id="zprediction"',
                   'id="verdicts"', 'id="takeover"', '/static/js/retro-panel.js?v=13'):
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


def test_select_chords_save_and_load_and_never_leak_select_into_the_game():
    # Juniper 2026-10-27: Select + L2 saves, Select + R2 loads. Zelda's Select pauses, so a chord
    # must never reach the game; Select alone still does, as a tap on release.
    import json, re, shutil, subprocess
    import pytest
    js = (HUB / 'static/js/retro-panel.js').read_text()
    assert "fire(l2?'F7':'F10')" in js and 'function runHotkey(code)' in js
    if not shutil.which('node'):
        pytest.skip('node not installed')
    body = re.search(r'function selectChord\(st, pad, m, now, fire\) \{.*?\n  \}', js, re.S).group(0)
    reset = re.search(r'function resetChord\(st\) \{.*?\}', js).group(0)
    probe = f'''{body}
{reset}
const pad=(...down)=>({{buttons:Array.from({{length:16}},(_,i)=>({{pressed:down.includes(i)}}))}});
const st={{held:false,chorded:false,pulseUntil:0}}, fired=[], fire=c=>fired.push(c), seen=[];
const SEL=8,L2=6,R2=7, step=(t,...d)=>seen.push(selectChord(st,pad(...d),d.includes(SEL)?4:0,t,fire)&4);
step(0,SEL); step(50,SEL,L2); step(100,SEL,L2); step(150,SEL); step(200);      // save chord
step(400,SEL); step(450);                                                      // Select alone
const tap=[selectChord(st,pad(),0,500,fire)&4, selectChord(st,pad(),0,700,fire)&4];
step(1000,SEL); step(1050,SEL,R2); step(1100);                                  // load chord
step(2000,SEL); resetChord(st); step(2100,SEL); step(2150);                     // focus lost mid-hold
const after=[selectChord(st,pad(),0,2200,fire)&4];
console.log(JSON.stringify({{fired,seen,tap,after}}));'''
    run = subprocess.run(['node', '-e', probe], capture_output=True, text=True)
    assert run.returncode == 0, run.stderr
    got = json.loads(run.stdout)
    assert got['fired'] == ['F7', 'F10']                         # each chord fires once
    assert got['seen'][:5] == [0, 0, 0, 0, 0]                    # a save chord never sends Select
    assert got['seen'][5:7] == [0, 4] and got['tap'] == [4, 0]   # a lone Select taps on release, 150 ms
    assert got['seen'][7:10] == [0, 0, 0]
    assert got['seen'][10:] == [0, 0, 0] and got['after'] == [0]  # no stray tap after a lost hold


def test_let_orion_play_renews_until_taken_back():
    # Juniper 2026-10-28: a button that lets Orion play for real, not a 60 s experiment.
    # Behaviour is covered by evals/retro_play_panel.mjs (renewal and click take-back, mutation-checked).
    html = (HUB / 'templates/retro.html').read_text()
    js = (HUB / 'static/js/retro-panel.js').read_text()
    assert '<button id="orionplay"' in html
    assert 'const ORION_PLAY_SEC = 1800;' in js and "action('orionplay'" in js
    assert 'handoffGrant=null;renewOrion();}' in js           # an ended turn is renewed
    assert "if(takeBack())return;" in js                      # a click on the game takes it back
    assert 'function neutral() { orionPlay = false;' in js    # a hidden tab or lost socket ends it

"""The play panel is in the rendered template and wired in the linked asset (browser eval:
evals/retro_play_panel.py exercises the interaction against a fixture Retro)."""
from pathlib import Path

HUB = Path(__file__).resolve().parents[2]


def test_template_has_play_panel_and_current_asset():
    html = (HUB / 'templates/retro.html').read_text()
    for marker in ('id="play"', 'id="atlas"', 'id="leisure"', 'id="zgoal"', 'id="zprediction"',
                   'id="verdicts"', 'id="takeover"', '/static/js/retro-panel.js?v=21'):
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


def test_asset_tolerates_audio_only_packets_from_a_warming_core():
    # 2026-11-21: N64/Angrylion sends no picture for its first ~64 frames. draw() threw on the
    # resulting 0x0 image, neutral() printed "Paused. Click the game to keep playing.", and the
    # closed socket paused the session before the renderer ever warmed up — no N64 game loaded.
    js = (HUB / 'static/js/retro-panel.js').read_text()
    draw = js[js.index('async function draw('):js.index('async function load()')]
    assert 'const picture=h.width>0 && h.height>0;' in draw
    assert 'if(picture){\n      const image=ctx.createImageData' in draw   # 0x0 never reaches createImageData
    assert draw.index('const picture=') < draw.index('createImageData')
    assert 'if(picture){\n      if(screen.width!==h.width' in draw          # no frame credit without a picture


def test_policy_poll_is_fast_while_orion_holds_the_controller():
    # 2026-10-24: a 1 s poll added up to 1 s to every decision while Link stood still.
    js = (HUB / 'static/js/retro-panel.js').read_text()
    assert "if(!(scriptedMode && cognitivePort!==null) && now-lastPolicyPoll<1000)return;" in js
    assert "finally{pollingPolicy=false;}},250);" in js


def test_gamepad_uses_nintendo_layout_and_skips_empty_or_nonstandard_pads():
    # 8BitDo Pro 2 (and the standard mapping) is positional: right face button = NES A,
    # bottom = NES B. The old mapping had them swapped (Xbox layout).
    import json, re, shutil, subprocess
    import pytest
    js = (HUB / 'static/js/retro-panel.js').read_text()
    html = (HUB / 'templates/retro.html').read_text()
    assert 'id="padInfo"' in html and "addEventListener('gamepadconnected',padStatus)" in js
    if not shutil.which('node'):
        pytest.skip('node not installed')
    body = re.search(r'function padMask\(pad, n64=false\) \{.*?\n  \}', js, re.S).group(0)
    pick = re.search(r'function gamepad\(\) \{.*?\n  \}', js, re.S).group(0)
    probe = f'''{body}
{pick}
const pad=(pressed,axes=[0,0])=>({{id:'8BitDo Pro 2',connected:true,mapping:'standard',axes,
  buttons:Array.from({{length:16}},(_,i)=>({{pressed:pressed.includes(i)}}))}});
const el=()=>({{value:'0'}});
const out={{A:padMask(pad([1])),B:padMask(pad([0])),start:padMask(pad([9])),select:padMask(pad([8])),
  X:padMask(pad([3])),Y:padMask(pad([2])),L:padMask(pad([4])),R:padMask(pad([5])),
  triggers:padMask(pad([6,7])),
  stickLeftUp:padMask(pad([],[-1,-1])),dpadRight:padMask(pad([15]))}};
Object.defineProperty(globalThis,'navigator',{{configurable:true,
  value:{{getGamepads:()=>[null,{{...pad([]),mapping:''}},pad([1])]}}}});   // Node 21+ has a read-only navigator
out.picked=padMask(gamepad());
console.log(JSON.stringify(out));'''
    run = subprocess.run(['node', '-e', probe], capture_output=True, text=True)
    assert run.returncode == 0, run.stderr
    got = json.loads(run.stdout)
    assert got == dict(A=256, B=1, X=512, Y=2, L=1024, R=2048, triggers=0,
                       start=8, select=4, stickLeftUp=64 | 16, dpadRight=128, picked=256)


def test_snes_keyboard_and_pad_buttons_are_filtered_only_for_nes_sessions():
    import json, re, shutil, subprocess
    import pytest
    if not shutil.which('node'): pytest.skip('node not installed')
    js = (HUB / 'static/js/retro-panel.js').read_text()
    functions = '\n'.join(re.search(pattern, js, re.S).group(0) for pattern in (
        r'const bits = .*?;', r'function mask\(\) \{.*?\n  \}',
        r'function padMask\(pad, n64=false\) \{.*?\n  \}'))
    probe = functions + '''
let activeSystem='snes', pad=null;
const screen={}, stage={}, document={activeElement:screen}, keys=new Set(['KeyA','KeyS','KeyQ','KeyW']);
const gamepad=()=>pad, chord={}, resetChord=()=>{}, selectChord=(_st,_pad,m)=>m;
const keyboard=mask(); keys.clear();
pad={buttons:Array.from({length:16},(_,i)=>({pressed:[2,3,4,5].includes(i)}))};
const gamepadSnes=mask(); activeSystem='nes'; const gamepadNes=mask();
keys.add('KeyZ'); const nesA=mask();
console.log(JSON.stringify({keyboard,gamepadSnes,gamepadNes,nesA}));'''
    run = subprocess.run(['node', '-e', probe], capture_output=True, text=True, check=True)
    assert json.loads(run.stdout) == dict(keyboard=3586, gamepadSnes=3586, gamepadNes=0, nesA=256)


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
    assert 'if(orionPlay && ranOut)renewOrion();' in js       # only a turn that ran out its time is renewed
    assert "Someone else took the controller" in js           # never taken back from someone else
    assert "if(takeBack())return;" in js                      # a click on the game takes it back
    assert 'function neutral() { setOrion(false);' in js      # a hidden tab or lost socket ends it
    assert 'aria-pressed="false"' in html


def test_n64_dropdown_label_sort_buttons_and_sticks():
    # Juniper 2026-11-16: N64 games read "(N64)" in the dropdown, and the dropdown is sorted.
    import json, re, shutil, subprocess
    import pytest
    js = (HUB / 'static/js/retro-panel.js').read_text()
    assert "SYSTEM_LABEL[g.system]" in js and ".sort((x,y)=>titleKey(x).localeCompare(titleKey(y)))" in js
    if not shutil.which('node'): pytest.skip('node not installed')
    functions = '\n'.join(re.search(pattern, js, re.S).group(0) for pattern in (
        r'const bits = .*?;', r'const SYSTEM_LABEL = .*?;', r'const gameKey = .*?;\n', r'const titleKey = .*?;\n',
        r'function mask\(\) \{.*?\n  \}', r'function sticks\(\) \{.*?\n  \}',
        r'function padMask\(pad, n64=false\) \{.*?\n  \}'))
    probe = functions + '''
let activeSystem='n64', pad=null;
const screen={}, stage={}, document={activeElement:screen};
const gamepad=()=>pad, chord={}, resetChord=()=>{}, selectChord=(_st,_pad,m)=>m;
const keys=new Set(['ArrowUp','ArrowLeft','KeyX','KeyC','KeyL','ShiftLeft']);
const keyboard={mask:mask(), sticks:sticks()}; keys.clear();
pad={axes:[.5,0,0,-1],buttons:Array.from({length:16},(_,i)=>({pressed:[1,6].includes(i)}))};
const padded={mask:mask(), sticks:sticks()};
const n64Keys=['KeyC','KeyJ','ArrowUp','ShiftLeft'].map(gameKey);
activeSystem='snes'; keys.add('KeyC'); keys.add('KeyI'); pad=null; const snesStray=mask();
const snesKeys=['KeyC','KeyJ','ArrowUp','KeyA'].map(gameKey);
const titles=[{title:'Super Mario Kart',system:'snes'},{title:'The Legend of Zelda: Ocarina of Time',system:'n64'},
  {title:'GoldenEye 007',system:'n64'},{title:'Mario Kart 64',system:'n64'}]
  .sort((x,y)=>titleKey(x).localeCompare(titleKey(y))).map(g=>g.title+(SYSTEM_LABEL[g.system]||''));
console.log(JSON.stringify({keyboard,padded,snesStray,titles,n64Keys,snesKeys}));'''
    run = subprocess.run(['node', '-e', probe], capture_output=True, text=True, check=True)
    got = json.loads(run.stdout)
    # Arrows and I/J/K/L move sticks, never the D-pad; X = A (bit 0), C = Z (bit 12); no Select.
    assert got['keyboard'] == dict(mask=1 | 4096, sticks=[-32767, -32767, 32767, 0])
    # Pad: right face = libretro A (bit 8), L2 = Z; the left stick is analog, not D-pad bits.
    assert got['padded'] == dict(mask=256 | 4096, sticks=[16384, 0, 0, -32767])
    assert got['snesStray'] == 0  # Z and C-stick keys never reach a console without them
    # N64-only keys are not game keys elsewhere: they neither take the game back nor get swallowed.
    assert got['n64Keys'] == [True, True, True, False] and got['snesKeys'] == [False, False, True, True]
    assert got['titles'] == ['GoldenEye 007 (N64)', 'The Legend of Zelda: Ocarina of Time (N64)',
                             'Mario Kart 64 (N64)', 'Super Mario Kart (SNES)']


def test_who_plays_question_is_inside_the_fullscreen_stage_and_wired():
    # Game company patch 1: the page asks who plays, once per sitting; the server never guesses.
    html = (HUB / 'templates/retro.html').read_text()
    stage = html[html.index('id="stage"'):html.index('</canvas>')]
    for marker in ('id="whoPlays" hidden', 'id="whoJuniper"', 'id="whoGuest"', 'id="whoLater"'):
        assert marker in stage, marker                      # fullscreen shows only what is inside #stage
    js = (HUB / 'static/js/retro-panel.js').read_text()
    for marker in ("api('sitting')", "api('sitting/player',{player,sitting_id:asked})", "['whoJuniper','juniper']",
                   "['whoGuest','guest']",
                   # asked of the page that holds the controller, once per sitting, never after an answer
                   "!(s && lease && !s.player_stated && s.sitting_id!==sittingAsked)",
                   # "Not now" is an answer the page keeps, also across a reload
                   "action('whoLater',async()=>settle(el('whoPlays').dataset.sitting));",
                   "sessionStorage.setItem('retroSittingAsked',id)", "sessionStorage.getItem('retroSittingAsked')",
                   "screen.focus();"):
        assert marker in js, marker

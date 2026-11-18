"""The chat box on the game page (game company patch 3, spec section 5): static wiring checks.
The browser eval (evals/retro_play_panel.py) drives the real page: open by key and button in
fullscreen, typing never moves the game, the /ws contract frame, the line strip and thumbs."""
import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

HUB = Path(__file__).resolve().parents[2]
HTML = (HUB / 'templates/retro.html').read_text()
JS = (HUB / 'static/js/retro-panel.js').read_text()


def test_template_loads_the_asset_and_the_chat_box_is_inside_the_stage():
    assert '<script src="/static/js/retro-panel.js?v=20" defer></script>' in HTML
    start = HTML.index('id="stage"')
    stage = HTML[start:HTML.index('</canvas>', start)]   # fullscreen shows only #stage
    for marker in ('id="chatToggle"', 'id="chatPanel" hidden', 'id="chatStrip" hidden', 'id="chatStripText"',
                   'id="chatLog"', 'id="chatForm"', 'id="chatInput" type="text"', 'id="chatClose"'):
        assert marker in stage, marker


def test_chat_key_is_not_a_game_key_or_hotkey():
    key = re.search(r"const CHAT_KEY='(\w+)'", JS).group(1)
    bits = re.search(r'const bits = \{(.*?)\}', JS, re.S).group(1)
    assert key == 'KeyT' and key not in bits and key not in re.search(r'const HOTKEYS=\{.*?\};', JS).group(0)
    # Typing never toggles it: the key opens the panel only outside a text field.
    assert "if(e.code===CHAT_KEY && !chatOpen && !typing(e.target)" in JS


def test_every_focus_return_skips_while_the_chat_is_open():
    assert 'function focusGame(){ if(!chatOpen)screen.focus(); }' in JS
    assert JS.count('screen.focus()') == 1                    # only inside focusGame
    # The spec's sites: grant, resume, save, load, Start, fullscreen change, the who-plays answer, the watch switch.
    assert JS.count('focusGame()') >= 9


def test_opening_releases_held_buttons_and_pauses_only_without_a_controller():
    body = re.search(r'function openChat\(\)\{.*?\n  \}', JS, re.S).group(0)
    assert 'keys.clear();send();' in body                     # one neutral input
    assert 'if(lease && !scriptedMode && !gamepad())' in body and 'pauseGame()' in body
    assert "action('pause',pauseGame);" in JS                 # the page's existing pause path
    close = re.search(r'function closeChat\(\)\{.*?\n  \}', JS, re.S).group(0)
    assert 'chat.pausedGame' in close and 'connect()' in close
    # The gamepad keeps driving the game while the panel is open.
    assert JS.count('document.fullscreenElement===stage || chatOpen') == 2


def test_message_builder_and_thumb_url_follow_the_contract():
    if not shutil.which('node'):
        pytest.skip('node not installed')
    builder = re.search(r'function chatMessage\(text, sessionId, sittingId\)\{.*?\n  \}', JS, re.S).group(0)
    thumb = re.search(r'const thumbURL=.*?;\n', JS).group(0)
    probe = builder + '\n' + thumb + '''
console.log(JSON.stringify({msg:chatMessage('hi','retro-guest-s1','s1'), url:thumbURL('line/1')}));'''
    run = subprocess.run(['node', '-e', probe], capture_output=True, text=True, check=True)
    got = json.loads(run.stdout)
    assert got['msg'] == {'text_input': 'hi', 'mode': 'orion', 'disable_tts': True, 'session_id': 'retro-guest-s1',
                          'surface_context': {'surface': 'retro', 'input_modality': 'typed', 'retro_sitting_id': 's1'}}
    assert got['url'] == 'lines/line%2F1/thumb'                # api() prefixes /api/retro/
    assert "api(thumbURL(lineId),{thumb})" in JS


def test_sessions_and_routing_of_pushes():
    # Juniper stated -> her main thread (created like the main page: GET /api/session); else a guest id.
    assert "localStorage.getItem('orion_sid')" in JS and "fetch('/api/session'" in JS
    assert "'retro-guest-'+(chat.sitting" in JS
    assert "{type:'session_hello', session_id:sessionId}" in JS
    # Each message keeps the source it was sent with.
    assert 'const player=chat.player, sittingId=chat.sitting;' in JS
    # Only a line (source retro_commentary with a line_id) reaches the strip; other outreach the panel only.
    assert "if(d.source==='retro_commentary' && d.line_id)" in JS
    # Orion's other messages are to Juniper: shown only when Juniper is the stated player.
    assert "} else if(chat.player==='juniper') chatEntry('orion',text,{note:'Orion, unprompted'});" in JS
    assert "if(d.type==='final')" in JS
    assert 'Math.min(30000,1000*2**chat.retry++)' in JS     # reconnect with backoff


def test_watch_switch_has_three_positions_mapped_to_the_server_modes():
    # Spec section 5: Off / Watch quietly / Watch and talk = modes off / quiet / on (orion/retro/watch.py MODES).
    for marker in ('<select id="watchSwitch"', '<option value="off">Off</option>',
                   '<option value="quiet">Watch quietly</option>', '<option value="on" selected>Watch and talk</option>'):
        assert marker in HTML, marker
    assert "api('sitting/watch',{mode:sw.value})" in JS
    assert "if(poll && (sw.disabled || document.activeElement===sw))return;" in JS

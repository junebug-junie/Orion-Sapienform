"""The dojo's pure parts: which goals a decision offers and how states fold into archive cells."""
import importlib.util
from pathlib import Path
import sys

EVALS = Path(__file__).resolve().parents[1] / 'evals'
sys.path.insert(0, str(EVALS))
spec = importlib.util.spec_from_file_location('zelda_dojo', EVALS / 'zelda_dojo.py')
dojo = importlib.util.module_from_spec(spec); spec.loader.exec_module(dojo)


def obs(**over):
    o = dict(status='observed', mode='play', level=0, room=119, link=dict(x=120, y=141, tile=[15, 11]),
             hearts=dict(current=3, containers=3), items=dict(sword=1, rupees=4, bombs='unknown'),
             enemies=[], drops=[])
    o.update(over)
    return o


def test_options_follow_the_offer_only():
    assert dojo.options(dict(observation=obs(), open_sides=['N', 'W'])) == [
        {'kind': 'exit', 'dir': 'N'}, {'kind': 'exit', 'dir': 'W'}]
    busy = obs(enemies=[dict(slot=2, type=7, tile=[3, 4])], drops=[dict(tile=[31, 0])])
    goals = dojo.options(dict(observation=busy, open_sides=[]))
    assert goals == [{'kind': 'fight', 'target': 'nearest'}, {'kind': 'wait'},
                     {'kind': 'collect', 'tile': {'x': 30, 'y': 1}}]          # clamped onto the play grid
    unknown = dojo.options(dict(observation=obs(), open_sides=None))
    assert [g['dir'] for g in unknown] == list('NSWE') and {g['kind'] for g in unknown} == {'explore'}


def test_cells_fold_nearby_tiles_and_split_on_health_held_items_and_room():
    a = dojo.cell_key(obs())
    assert dojo.cell_key(obs(link=dict(x=0, y=0, tile=[13, 10]))) == a          # same coarse block
    assert dojo.cell_key(obs(link=dict(x=0, y=0, tile=[2, 10]))) != a
    assert dojo.cell_key(obs(hearts=dict(current=2, containers=3))) != a
    assert dojo.cell_key(obs(items=dict(sword=1, rupees=9))) == a              # more rupees: same cell
    assert dojo.cell_key(obs(items=dict(sword=0, rupees=4))) != a              # a new item held: new cell
    assert dojo.cell_key(obs(room=120)) != a


def test_the_dojo_and_its_shared_contract_import_without_numpy():
    # The live dojo runs in the retro image, which has no numpy (review 2026-10-28): only running
    # weights (orion-prediction) may need it.
    import subprocess
    root = Path(__file__).resolve().parents[3]
    probe = ("import sys; sys.modules['numpy'] = None\n"
             "from orion.embodiment import transition as t\n"
             "x = t.zelda_features(t.zelda_facts({'mode': 'play', 'level': 0, 'room': 119, 'link': {'tile': [15, 11]},"
             " 'hearts': {'current': 3, 'containers': 3}, 'items': {}, 'enemies': [], 'drops': []}), {'kind': 'exit', 'dir': 'N'}, ['N'])\n"
             "assert len(x) == t.N_FEATURES\n"
             "import importlib.util\n"
             f"spec = importlib.util.spec_from_file_location('d', {str(EVALS / 'zelda_dojo.py')!r})\n"
             "print('ok')\n")
    run = subprocess.run([sys.executable, '-c', probe], capture_output=True, text=True,
                         env={'PYTHONPATH': f'{root}:{root / "services/orion-retro"}', 'PATH': '/usr/bin:/bin'})
    assert run.returncode == 0 and run.stdout.strip() == 'ok', run.stderr

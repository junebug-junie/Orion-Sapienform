"""TransitionModel labels, features and the training gate's pieces (no dojo run needed)."""
import importlib.util
from pathlib import Path
import random

import torch

from app.transition import HEADS, N_FEATURES, TransitionModel, zelda_features, zelda_targets
from orion.embodiment.skills import SKILLS

SCRIPT = Path(__file__).resolve().parents[1] / 'scripts' / 'train_transition.py'
spec = importlib.util.spec_from_file_location('train_transition', SCRIPT)
train = importlib.util.module_from_spec(spec); spec.loader.exec_module(train)


def facts(**over):
    f = dict(mode='play', level=0, room=119, tile=[15, 11], hearts=dict(current=3, containers=3),
             items=dict(sword=1, rupees=4, bombs='unknown'), enemies=[], drops=[])
    f.update(over)
    return f


def row(before, after, **over):
    return {**dict(before=before, after=after, goal={'kind': 'exit', 'dir': 'N'}, frames=120, died=False), **over}


def test_labels_follow_the_hubs_settle_rules():
    # Review 2026-10-28: a death changed the (mode, room) tuple and read as "place changed"; a heart
    # pickup did not read as a gain. The labels now use the Hub's own rules (orion.embodiment.transition).
    i = HEADS.index
    died = zelda_targets(row(facts(), facts(mode='death', hearts=dict(current=0, containers=3)), died=True, outcome='death'))
    assert died[i('place_changed')] == 0.0 and died[i('died')] == 1.0 and died[i('health_lost')] == 1.0
    hit = zelda_targets(row(facts(), facts(room=120), outcome='interrupted:hit'))
    assert hit[i('place_changed')] == 0.0                                # traverse settles an interruption failure
    heart = zelda_targets(row(facts(hearts=dict(current=2, containers=3)), facts(hearts=dict(current=3, containers=3))))
    assert heart[i('item_gained')] == 1.0 and heart[i('health_lost')] == 0.0
    into_cave = zelda_targets(row(facts(), facts(mode='cave')))
    assert into_cave[i('place_changed')] == 0.0                          # same room, as the Hub settles it
    assert zelda_targets(row(facts(), facts(room=120)))[i('place_changed')] == 1.0


def test_labels_count_a_cave_exit_as_a_new_place_and_a_newly_read_item_as_a_gain():
    out_of_cave = zelda_targets(row(facts(mode='cave'), facts()))
    assert out_of_cave[HEADS.index('place_changed')] == 1.0            # same room id, different mode
    read_later = zelda_targets(row(facts(), facts(items=dict(sword=1, rupees=4, bombs=3))))
    assert read_later[HEADS.index('item_gained')] == 1.0                # the Hub's rule: a new item read counts
    rupee = zelda_targets(row(facts(), facts(items=dict(sword=1, rupees=5, bombs='unknown'))))
    assert rupee[HEADS.index('item_gained')] == 1.0
    hurt = zelda_targets(row(facts(), facts(hearts=dict(current=2.5, containers=3)), died=True))
    assert hurt[HEADS.index('health_lost')] == 1.0 and hurt[HEADS.index('died')] == 1.0
    assert zelda_targets(row(facts(), None)) is None                    # no after state: says nothing


def test_features_use_the_body_vocabulary_and_the_count_key_reads_them_back():
    x = zelda_features(facts(enemies=[dict(tile=[16, 11])], hearts=dict(current=1, containers=3)),
                       {'kind': 'fight', 'target': 'nearest'}, ['N', 'W'])
    assert len(x) == N_FEATURES and x[:len(SKILLS)] == [float(s == 'engage') for s in SKILLS]
    situation, skill, everything = train.count_keys(x)
    assert situation == (SKILLS.index('engage'), 0, 2, 1) and skill == (SKILLS.index('engage'),) and everything == ()
    exit_n = zelda_features(facts(), {'kind': 'exit', 'dir': 'N'}, ['N', 'W'])
    assert train.count_keys(exit_n)[0] == (SKILLS.index('traverse'), 1, 0, 0)   # its side is open


def test_count_baseline_backs_off_to_the_skill_then_everything():
    x_fight = zelda_features(facts(enemies=[dict(tile=[16, 11])]), {'kind': 'fight', 'target': 'nearest'}, [])
    x_exit = zelda_features(facts(), {'kind': 'exit', 'dir': 'N'}, ['N'])
    rows = [dict(x=x_exit, y=[1.0, 0.0, 0.0, 0.0, 0.1])] * 10
    base = train.count_baseline(rows)
    assert base(x_exit)[0] == (10 + 1) / (10 + 2)                       # its own situation
    assert base(x_fight)[0] == (10 + 1) / (10 + 2)                      # no engage rows: the global rate


def test_cluster_interval_is_wider_than_one_cluster_pretending_to_be_many():
    rng = random.Random(0)
    diffs = [0.1] * 50 + [-0.1] * 50
    lo, hi, n = train.cluster_ci(diffs, ['a'] * 50 + ['b'] * 50, rng)
    assert n == 2 and lo < 0 < hi                                       # two clusters: no claim either way


def test_model_outputs_one_probability_per_head():
    out = TransitionModel().predict(torch.zeros(3, N_FEATURES))
    assert set(out) == set(HEADS) | {'frames'} and all(v.shape == (3,) for v in out.values())
    assert all(0 <= float(p) <= 1 for h in HEADS for p in out[h])


def test_numpy_export_matches_the_torch_model(tmp_path):
    # orion-prediction has no torch: it runs the exported weights through
    # orion.embodiment.transition.TransitionWeights. Same input, same heads.
    from orion.embodiment.transition import TransitionWeights
    torch.manual_seed(3)
    model = TransitionModel()
    train.export(model, tmp_path / 'w.npz', 'tag-x')
    w = TransitionWeights.load(tmp_path / 'w.npz')
    assert w.tag == 'tag-x' and w.version == 'zelda-body-v1'
    x = zelda_features(facts(enemies=[dict(tile=[16, 11])]), {'kind': 'exit', 'dir': 'W'}, ['W', 'N'])
    with torch.no_grad():
        want = model.predict(torch.tensor([x]))
    got = w.heads(x)
    for h in HEADS:
        assert abs(got[h] - float(want[h][0])) < 1e-5, h

"""Import-boundary tests: orion-thought's reverie must not load orion.substrate or requests.

The thin orion-thought container ships neither the graph engine nor ``requests``. Importing the reverie
(and its seed module) and building a seed and a plan request must not drag them in, at import or at
call time. Run in a fresh interpreter so prior test imports cannot mask a regression.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def _run(code: str) -> subprocess.CompletedProcess:
    service_root = Path(__file__).resolve().parents[1]
    repo_root = Path(__file__).resolve().parents[3]
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([str(service_root), str(repo_root)])
    return subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env,
                          cwd=str(service_root))


def test_importing_and_using_the_reverie_does_not_load_substrate_or_requests() -> None:
    code = (
        "import sys\n"
        "import app.reverie as r, app.reverie_seeds as s\n"
        "rows=[{'faculty':'f','bin':1,'n':60,'mean_p':0.15,'rate':0.40}]\n"
        "seed=s.lost_bet_seeds(rows, spent_refs=set())[0]\n"
        "req=r.build_question_plan_request(seed, correlation_id='c1')\n"
        "assert req.plan.verb_name == 'reverie_question'\n"
        "out=r.check_question({'question':'q?','refs':seed.refs}, seed=seed)\n"
        "assert out['refused_reason'] is None\n"
        "sub=[m for m in sys.modules if m.startswith('orion.substrate')]\n"
        "assert not sub, sub\n"
        "assert 'requests' not in sys.modules\n"
        "print('ok')\n"
    )
    result = _run(code)
    assert result.returncode == 0, result.stderr
    assert "ok" in result.stdout

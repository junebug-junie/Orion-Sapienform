from pathlib import Path


def test_app_js_loads_cognitive_loops():
    js = Path("services/orion-hub/static/js/app.js").read_text()
    assert "/api/attention/loops" in js
    assert "loadCognitiveLoops" in js
    assert "resolve" in js and "dismiss" in js


def test_template_has_cognitive_loop_container():
    html = Path("services/orion-hub/templates/index.html").read_text()
    assert 'id="cognitiveLoopsList"' in html

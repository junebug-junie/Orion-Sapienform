"""Live geometry check for the Walkway stories pager (PR #4019, ticket #4016).

Loads the REAL Hub page (real Tailwind, real stories) at phone size in Chromium, swaps in this
checkout's walkway-stories.js and pager markup, and walks Next / Back from deep inside each story,
with scroll anchoring on and off (off = iPhone Safari's behaviour). Every non-GET request is answered
locally, so nothing is written to Hub. Pass: the pager stays at the top bar's bottom edge and each
new story starts right under it.

Run:  python services/orion-hub/evals/walkway_pager_live_geometry.py   (needs playwright + chromium)
"""
import json, re, sys, os
from playwright.sync_api import sync_playwright
W = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
JS = open(f"{W}/static/js/walkway-stories.js", "rb").read()
tpl = open(f"{W}/templates/index.html").read()
def snippet(pat):
    m = re.search(pat, tpl, re.S); assert m, pat; return m.group(0)
NEW_PAGER = snippet(r'<div id="walkwayStoriesPager".*?</div>')
NEW_STATUS = snippet(r'<div id="walkwayStoriesStatus"[^>]*>')
BASE = os.environ.get("HUB_URL", "http://localhost:8080")
posts = []

def patch_html(h):
    h = h.replace('<div class="fixed top-0 left-0 right-0 z-40', '<div id="hubTopBar" class="fixed top-0 left-0 right-0 z-40', 1)
    h = re.sub(r'<div id="walkwayStoriesPager".*?</div>', NEW_PAGER, h, count=1, flags=re.S)
    h = re.sub(r'<div id="walkwayStoriesStatus"[^>]*>', NEW_STATUS, h, count=1)
    assert 'id="hubTopBar"' in h and 'position: sticky' in h
    return h

def handle(route):
    req = route.request; url = req.url
    if req.method != "GET":
        posts.append(url)
        if url.endswith("/parts"):
            sent = json.loads(req.post_data or "{}")
            return route.fulfill(body=json.dumps({"ok": True, "story": None, "story_verdict": None}), content_type="application/json")
        if url.endswith("/verdict"):
            return route.fulfill(body=json.dumps({"ok": True, "tally": None}), content_type="application/json")
        return route.fulfill(body=json.dumps({"ok": True}), content_type="application/json")
    if url.split("?")[0] == BASE + "/":
        r = route.fetch(); return route.fulfill(response=r, body=patch_html(r.text()))
    if "/static/js/walkway-stories.js" in url:
        return route.fulfill(body=JS, content_type="application/javascript")
    return route.continue_()

with sync_playwright() as p:
    for anchor in (True, False):
        b = p.chromium.launch(); page = b.new_page(viewport={"width": 390, "height": 844}, device_scale_factor=3, is_mobile=True, has_touch=True)
        page.route("**/*", handle)
        page.goto(BASE + "/", wait_until="domcontentloaded")
        page.wait_for_function("() => !!window.OrionWalkwayStories && !!window.tailwind", timeout=60000)
        if not anchor: page.add_style_tag(content="*{overflow-anchor:none!important}")
        page.select_option("#visionSource", "walkway-live")
        page.wait_for_selector("#walkwayStoriesList [data-story-id]:not(.hidden)", timeout=60000)
        page.wait_for_timeout(3000)
        tag = "anchor" if anchor else "NOANCHOR"
        G = """() => { const bar=document.getElementById('hubTopBar').getBoundingClientRect(); const pg=document.getElementById('walkwayStoriesPager').getBoundingClientRect();
          const it=document.querySelector('#walkwayStoriesList [data-story-id]:not(.hidden)').getBoundingClientRect();
          return {bar: Math.round(bar.bottom), pager: Math.round(pg.top), pagerBottom: Math.round(pg.bottom), story: Math.round(it.top), storyH: Math.round(it.height), y: Math.round(scrollY), pos: document.getElementById('walkwayStoriesPos').textContent, pin: document.getElementById('walkwayStoriesPager').style.top}; }"""
        print(tag, "start", page.evaluate(G))
        n = page.evaluate("document.querySelectorAll('#walkwayStoriesList [data-story-id]').length")
        heights = page.evaluate("""() => [...document.querySelectorAll('#walkwayStoriesList [data-story-id]')].map(e => { e.classList.remove('hidden'); const h=Math.round(e.getBoundingClientRect().height); return h; })""")
        page.evaluate("() => window.dispatchEvent(new Event('resize'))")
        # restore the one-at-a-time view
        page.click("#walkwayStoriesNext"); page.wait_for_timeout(300); page.click("#walkwayStoriesBack"); page.wait_for_timeout(300)
        print(tag, "stories", n, "heights", heights)
        # scroll to the bottom of the current story, check pager stays under the bar
        page.evaluate("() => { const it=document.querySelector('#walkwayStoriesList [data-story-id]:not(.hidden)'); const r=it.getBoundingClientRect(); scrollBy(0, r.bottom - innerHeight + 10); }")
        page.wait_for_timeout(300)
        print(tag, "deep in story", page.evaluate(G))
        for i in range(min(n - 1, 6)):
            page.click("#walkwayStoriesNext"); page.wait_for_timeout(400)
            g = page.evaluate(G); print(tag, "Next", g)
            page.evaluate("() => { const it=document.querySelector('#walkwayStoriesList [data-story-id]:not(.hidden)'); const r=it.getBoundingClientRect(); scrollBy(0, Math.max(0, r.bottom - innerHeight + 10)); }")
            page.wait_for_timeout(200)
        for i in range(3):
            page.click("#walkwayStoriesBack"); page.wait_for_timeout(400)
            print(tag, "Back from bottom", page.evaluate(G))
            page.evaluate("() => { const it=document.querySelector('#walkwayStoriesList [data-story-id]:not(.hidden)'); const r=it.getBoundingClientRect(); scrollBy(0, Math.max(0, r.bottom - innerHeight + 10)); }")
            page.wait_for_timeout(200)
        b.close()
print("POSTS (faked, none reached Hub):", posts)

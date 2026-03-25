from __future__ import annotations

from pathlib import Path

import pytest


def test_schedule_panel_browser_smoke(tmp_path):
    pytest.importorskip("playwright.sync_api")
    from playwright.sync_api import sync_playwright
    schedule_js = Path(__file__).resolve().parents[1] / "static" / "js" / "workflow-schedule-ui.js"
    script = schedule_js.read_text()
    html = f"""
    <html><body>
      <div id='status'></div>
      <button id='refresh'>Refresh</button>
      <div id='toast'></div>
      <div id='scheduleInventoryList'></div>
      <div id='scheduleModal' style='display:none'><div id='modalHealth'></div><div id='modalAnalytics'></div></div>
      <div id='scheduleEditModal' style='display:none'><div id='editStatus'></div></div>
      <script>{script}</script>
      <script>
        const api = window.OrionWorkflowScheduleUI;
        let selected = null;
        let conflictMode = false;
        let rows = [
          api.normalizeSchedule({{
            schedule_id:'abc1234567', workflow_id:'journal_pass', workflow_display_name:'Journal Pass', state:'scheduled', notify_on:'completion',
            next_run_at:'2026-03-26T05:00:00Z', execution_policy:{{schedule:{{kind:'recurring', cadence:'daily', hour_local:22, minute_local:0}}}},
            analytics:{{health:'degraded', needs_attention:true, is_overdue:true, overdue_seconds:3600, recent_run_count:5, recent_success_count:2, recent_failure_count:3, recent_outcomes:['failed','failed','completed']}}
          }})
        ];
        const list = document.getElementById('scheduleInventoryList');
        const status = document.getElementById('status');
        const toast = document.getElementById('toast');

        function performAction() {{
          if (conflictMode) {{
            const err = new Error('Schedule revision conflict: expected 1, current 2.');
            err.code = 'schedule_revision_conflict';
            throw err;
          }}
          return {{ ok: true }};
        }}

        function render() {{
          list.innerHTML='';
          rows.forEach((r) => {{
            const d=document.createElement('div');
            d.innerHTML = `<span class='health'>${{r.analytics ? r.analytics.health : 'none'}}</span><span class='overdue'>${{r.analytics && r.analytics.is_overdue ? 'Overdue' : ''}}</span><button class='details'>Details</button><button class='edit'>Edit</button><button class='pause'>Pause</button><span class='state'>${{r.state}}</span>`;
            d.querySelector('.details').onclick=()=>{{
              document.getElementById('scheduleModal').style.display='block';
              document.getElementById('modalHealth').textContent=`health:${{r.analytics ? r.analytics.health : '--'}}`;
              document.getElementById('modalAnalytics').textContent=r.analytics && r.analytics.trend_text ? r.analytics.trend_text : 'No recent runs.';
            }};
            d.querySelector('.edit').onclick=()=>{{ selected = r; document.getElementById('scheduleEditModal').style.display='block'; document.getElementById('editStatus').textContent='Editing'; }};
            d.querySelector('.pause').onclick=()=>{{r.state='paused'; status.textContent='refreshed'; toast.textContent='Schedule pause succeeded.'; render();}};
            list.appendChild(d);
          }});
        }}

        document.getElementById('refresh').onclick = () => {{ status.textContent = 'refreshed'; render(); }};
        window.smokeSave = () => {{
          try {{
            performAction();
            document.getElementById('editStatus').textContent = 'Saved.';
            toast.textContent = 'Schedule update saved.';
          }} catch (err) {{
            const isConflict = String(err.code || '') === 'schedule_revision_conflict';
            document.getElementById('editStatus').textContent = isConflict ? `${{err.message}} Refresh and retry.` : err.message;
            toast.textContent = isConflict ? 'Schedule changed elsewhere. Refresh and retry.' : err.message;
          }}
        }};
        window.smokeSetConflict = (value) => {{ conflictMode = Boolean(value); }};
        render();
      </script>
    </body></html>
    """
    page_path = tmp_path / "schedule-smoke.html"
    page_path.write_text(html)

    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        page.goto(page_path.as_uri())

        assert page.locator('.health').first.inner_text() == 'degraded'
        assert page.locator('.overdue').first.inner_text() == 'Overdue'

        page.locator('.details').first.click()
        assert page.locator('#scheduleModal').evaluate('el => el.style.display') == 'block'
        assert 'health:degraded' in page.locator('#modalHealth').inner_text().lower()
        assert 'recent succeeded' in page.locator('#modalAnalytics').inner_text().lower()

        page.locator('.edit').first.click()
        page.evaluate('window.smokeSetConflict(false)')
        page.evaluate('window.smokeSave()')
        assert page.locator('#editStatus').inner_text() == 'Saved.'
        assert 'saved' in page.locator('#toast').inner_text().lower()

        page.evaluate('window.smokeSetConflict(true)')
        page.evaluate('window.smokeSave()')
        assert 'refresh and retry' in page.locator('#editStatus').inner_text().lower()
        assert 'refresh and retry' in page.locator('#toast').inner_text().lower()

        page.locator('#refresh').click()
        page.locator('.pause').first.click()
        assert page.locator('.state').first.inner_text() == 'paused'
        assert page.locator('#status').inner_text() == 'refreshed'
        browser.close()

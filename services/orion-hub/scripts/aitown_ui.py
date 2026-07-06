"""Server-side AI Town tab HTML fragments for Hub index render."""
from __future__ import annotations

import html
from typing import Any, Tuple


def render_aitown_tab_blocks(settings: Any) -> Tuple[str, str]:
    if not bool(getattr(settings, "HUB_AITOWN_ENABLED", False)):
        return "", ""

    aitown_ui_url = html.escape(str(getattr(settings, "HUB_AITOWN_UI_URL", "") or ""))
    nav = (
        '<a id="aiTownTabButton" href="#ai-town" data-hash-target="#ai-town" '
        'class="px-3 py-1.5 text-xs font-semibold rounded-full bg-gray-800 text-gray-200 '
        'border border-gray-700 hover:bg-gray-700" role="button">AI Town</a>'
    )
    panel = (
        '<section id="ai-town" data-panel="ai-town" class="hidden w-full bg-gray-900 rounded-2xl '
        'shadow-lg p-5 flex flex-col gap-4 min-h-[56rem]">'
        '<div class="flex items-center justify-between gap-3">'
        '<h2 class="text-xl font-bold text-white">AI Town</h2>'
        '<div class="flex items-center gap-2 text-xs">'
        f'<a href="{aitown_ui_url}" target="_blank" rel="noopener" '
        'class="text-indigo-300 hover:text-indigo-200 underline">Open standalone</a>'
        '<button id="aitownRefreshBtn" type="button" '
        'class="px-2 py-1 rounded bg-gray-800 border border-gray-700 text-gray-200">Refresh</button>'
        "</div></div>"
        '<div id="aitownStatusStrip" class="text-xs text-gray-400">Loading AI Town status…</div>'
        '<iframe id="aitownFrame" title="AI Town" class="w-full flex-1 min-h-[48rem] rounded-xl '
        'border border-gray-700 bg-black" loading="lazy"></iframe>'
        "</section>"
    )
    return nav, panel

from app.executor import _skip_journal_pageindex_for_automated_trigger


def test_skips_scheduler_daily():
    ctx = {
        "metadata": {
            "journal_trigger": {"trigger_kind": "daily_summary", "source_kind": "scheduler"},
        }
    }
    assert _skip_journal_pageindex_for_automated_trigger(ctx) is True


def test_skips_metacog_digest():
    ctx = {
        "metadata": {
            "journal_trigger": {"trigger_kind": "metacog_digest", "source_kind": "metacog"},
        }
    }
    assert _skip_journal_pageindex_for_automated_trigger(ctx) is True


def test_skips_notify_summary():
    ctx = {
        "metadata": {
            "journal_trigger": {"trigger_kind": "notify_summary", "source_kind": "notify"},
        }
    }
    assert _skip_journal_pageindex_for_automated_trigger(ctx) is True


def test_manual_collapse_not_skipped():
    ctx = {
        "metadata": {
            "journal_trigger": {"trigger_kind": "something_else", "source_kind": "manual"},
        }
    }
    assert _skip_journal_pageindex_for_automated_trigger(ctx) is False

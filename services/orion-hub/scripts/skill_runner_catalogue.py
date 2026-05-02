"""Hub Skill Runner: exact catalogue prompt -> concrete skill verb (direct exec, no chat_quick).

Keys MUST match option ``value=`` strings in ``services/orion-hub/templates/index.html``.
See ``docs/operator_skill_prompt_catalogue.md`` (kept in sync with the Hub Skill Runner dropdown).
"""

from __future__ import annotations

SKILL_RUNNER_CATALOGUE_VERBS: dict[str, str] = {
    "What time is it right now?": "skills.system.time_now.v1",
    "Show NVIDIA GPU status on this node.": "skills.gpu.nvidia_smi_snapshot.v1",
    "Show Docker container status on this node.": "skills.docker.ps_status.v1",
    "Dry-run cleanup of stopped containers.": "skills.runtime.docker_prune_stopped_containers.v1",
    "Prune stopped containers.": "skills.runtime.docker_prune_stopped_containers.v1",
    "Show the current biometrics snapshot.": "skills.biometrics.snapshot.v1",
    "Show the 10 most recent biometrics readings.": "skills.biometrics.raw_recent.v1",
    "Show the most recent biometrics readings for athena.": "skills.biometrics.raw_recent.v1",
    "Show the landing pad metrics snapshot.": "skills.landing_pad.metrics_snapshot.v1",
    "Show the last 10 landing pad events.": "skills.landing_pad.last_events.v1",
    "Show the last 10 landing pad events with salience above 0.7.": "skills.landing_pad.last_events.v1",
    "Which nodes are up right now?": "skills.mesh.tailscale_mesh_status.v1",
    "Run an active mesh probe.": "skills.mesh.tailscale_mesh_status.v1",
    "Check disk health across active nodes.": "skills.storage.disk_health_snapshot.v1",
    "Summarize recent PR changes.": "skills.repo.github_recent_prs.v1",
    "Run a mesh ops round.": "skills.mesh.mesh_ops_round.v1",
    "Run a mesh ops round with PR digest and disk health.": "skills.mesh.mesh_ops_round.v1",
    "Run a mesh ops round including docker housekeeping preview.": "skills.mesh.mesh_ops_round.v1",
    'Send a notification to operators saying "test alert from Orion".': "skills.system.notify_chat_message.v1",
    (
        "Run skills.chat.discussion_window.v1 on chat_history_log with lookback_seconds 3600 and max_turns 30 "
        "(optional filters: current user_id and hub source)."
    ): "skills.chat.discussion_window.v1",
    (
        "Run skills.chat.discussion_window.v1 on chat_history_log with lookback_seconds 86400 and max_turns 30 "
        "(optional filters: current user_id and hub source)."
    ): "skills.chat.discussion_window.v1",
}


def resolve_skill_runner_catalogue_verb(*, prompt: str, skill_runner_origin: bool) -> str | None:
    if not skill_runner_origin:
        return None
    key = str(prompt or "").strip()
    return SKILL_RUNNER_CATALOGUE_VERBS.get(key)

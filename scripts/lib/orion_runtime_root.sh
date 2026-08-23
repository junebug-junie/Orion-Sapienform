# shellcheck shell=bash
# Resolve the Orion checkout that owns venv/systemd runtime (main), not a linked worktree.
orion_resolve_runtime_root() {
    local repo_root="${1:?repo root required}"
    local main_root="$repo_root"

    if git -C "$repo_root" rev-parse --git-common-dir >/dev/null 2>&1; then
        main_root="$(cd "$(git -C "$repo_root" rev-parse --path-format=absolute --git-common-dir)/.." && pwd)"
    fi

    if [[ -x "${main_root}/venv/bin/python" ]]; then
        printf '%s\n' "$main_root"
    elif [[ -x "${repo_root}/venv/bin/python" ]]; then
        printf '%s\n' "$repo_root"
    elif [[ -x "${main_root}/orion_dev/bin/python" ]]; then
        printf '%s\n' "$main_root"
    elif [[ -x "${repo_root}/orion_dev/bin/python" ]]; then
        printf '%s\n' "$repo_root"
    else
        printf '%s\n' "$main_root"
    fi
}

orion_resolve_runtime_python() {
    local runtime_root
    runtime_root="$(orion_resolve_runtime_root "$1")"
    if [[ -x "${runtime_root}/venv/bin/python" ]]; then
        printf '%s\n' "${runtime_root}/venv/bin/python"
    elif [[ -x "${runtime_root}/orion_dev/bin/python" ]]; then
        printf '%s\n' "${runtime_root}/orion_dev/bin/python"
    else
        printf '%s\n' "python3"
    fi
}

def create_default_planner(base_dir: Path) -> SemanticPlanner:
    verbs_dir = base_dir / "verbs"
    prompts_dir = base_dir / "prompts"
    return SemanticPlanner(verbs_dir=verbs_dir, prompts_dir=prompts_dir)

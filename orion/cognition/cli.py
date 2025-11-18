import argparse
from pathlib import Path

from orion_cognition.packs_loader import PackManager
from orion_cognition.planner import SemanticPlanner, SystemState
from orion_cognition.planner.rdf_sync import generate_turtle_for_all

def main():
    parser = argparse.ArgumentParser(description="Orion Cognition CLI")
    sub = parser.add_subparsers(dest="cmd")

    sub.add_parser("list-packs")
    sub.add_parser("generate-rdf")

    sp = sub.add_parser("show-pack")
    sp.add_argument("pack")

    vp = sub.add_parser("verify-pack")
    vp.add_argument("pack")

    pl = sub.add_parser("plan")
    pl.add_argument("verb")

    args = parser.parse_args()
    base = Path(__file__).resolve().parent

    pm = PackManager(base)

    if args.cmd == "list-packs":
        pm.load_packs()
        print("\n".join(pm.list_packs()))

    elif args.cmd == "show-pack":
        pm.load_packs()
        pack = pm.get_pack(args.pack)
        print(pack.label)
        print("Verbs:", ", ".join(pack.verbs))

    elif args.cmd == "verify-pack":
        pm.load_packs()
        print(pm.verify_pack(args.pack))

    elif args.cmd == "generate-rdf":
        print(generate_turtle_for_all(base))

    elif args.cmd == "plan":
        planner = SemanticPlanner(
            verbs_dir=base/"verbs",
            prompts_dir=base/"prompts"
        )
        plan = planner.build_plan(args.verb, system_state=SystemState(name="Idle"))
        print(plan)

if __name__ == "__main__":
    main()

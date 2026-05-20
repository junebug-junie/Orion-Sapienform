from __future__ import annotations

import argparse
import sys
from pathlib import Path

from orion.knowledge_forge.compile import compile_context_pack_markdown
from orion.knowledge_forge.lint import lint_corpus
from orion.knowledge_forge.paths import resolve_corpus_root
from orion.knowledge_forge.probes import probe_source_coverage
from orion.knowledge_forge.review import apply_pending_patch, list_pending_patches
from orion.knowledge_forge.store import KnowledgeStore
from orion.knowledge_forge.yaml_doc import save_yaml_doc


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="orion.knowledge_forge")
    sub = parser.add_subparsers(dest="cmd", required=True)

    lint_p = sub.add_parser("lint")
    lint_p.add_argument("--report-out", default="")

    review = sub.add_parser("review")
    review_sub = review.add_subparsers(dest="review_cmd", required=True)
    review_sub.add_parser("list")
    apply_p = review_sub.add_parser("apply")
    apply_p.add_argument("patch_id")

    compile_p = sub.add_parser("compile")
    compile_sub = compile_p.add_subparsers(dest="compile_cmd", required=True)
    ctx = compile_sub.add_parser("context-pack")
    ctx.add_argument("--spec", required=True)
    ctx.add_argument("--task", required=True)
    ctx.add_argument("--out", required=True)

    probe_p = sub.add_parser("probe")
    probe_sub = probe_p.add_subparsers(dest="probe_cmd", required=True)
    src = probe_sub.add_parser("source")
    src.add_argument("--source-id", required=True)
    src.add_argument("--path", required=True)
    src.add_argument("--keyword", required=True)

    args = parser.parse_args(argv)
    root = resolve_corpus_root()

    if args.cmd == "lint":
        store = KnowledgeStore(root)
        store.load()
        report = lint_corpus(store)
        if args.report_out:
            out = Path(args.report_out)
            out.parent.mkdir(parents=True, exist_ok=True)
            lines = [f"{i.code}\t{i.doc_id}\t{i.message}" for i in report.issues]
            out.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        for issue in report.issues:
            print(f"{issue.code}\t{issue.doc_id}\t{issue.message}")
        return 0 if report.ok else 1

    if args.cmd == "review" and args.review_cmd == "list":
        for patch in list_pending_patches(root):
            print(f"{patch.patch_id}\t{patch.target}\t{patch.action}")
        return 0

    if args.cmd == "review" and args.review_cmd == "apply":
        target = apply_pending_patch(root, args.patch_id)
        print(f"applied\t{target}")
        return 0

    if args.cmd == "compile" and args.compile_cmd == "context-pack":
        store = KnowledgeStore(root)
        store.load()
        md = compile_context_pack_markdown(store, spec_id=args.spec, task=args.task)
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(md, encoding="utf-8")
        pack_id = args.spec.replace("spec:", "ctx:", 1)
        meta_path = out.with_suffix(".yaml")
        save_yaml_doc(
            meta_path,
            {
                "type": "context_pack",
                "id": pack_id,
                "target": "cursor",
                "task": args.task,
                "included_specs": [args.spec],
                "allowed_sources": [],
                "excluded_context": [],
            },
        )
        print(f"wrote\t{out}")
        return 0

    if args.cmd == "probe" and args.probe_cmd == "source":
        store = KnowledgeStore(root)
        store.load()
        report = probe_source_coverage(
            store,
            source_path=Path(args.path),
            source_id=args.source_id,
            min_keyword=args.keyword,
        )
        for issue in report.issues:
            print(f"{issue.code}\t{issue.message}")
        return 0 if report.ok else 1

    parser.error("unknown command")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())

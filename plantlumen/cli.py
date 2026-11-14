"""Command line entry point: python -m plantlumen <command> ..."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from . import verify


def default_bench() -> Path:
    """Locate the shipped artefacts without the user having to pass --bench.

    Works from a source checkout (artefacts sit next to the package) and from a
    working directory that already contains an ``artefacts`` folder. Anything
    else is an error: guessing elsewhere risks verifying against someone else's
    bench tree, which would silently compare different numbers.
    """
    here = Path(__file__).resolve().parent
    for candidate in (here.parent / "artefacts", Path.cwd() / "artefacts"):
        if (candidate / "results").is_dir():
            return candidate
    raise SystemExit(
        "no artefacts/ directory next to the package or in the working directory; "
        "pass --bench DIR pointing at a directory that contains results/<cohort>/")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="plantlumen", description=__doc__)
    sub = ap.add_subparsers(dest="command", required=True)

    v = sub.add_parser("verify", help="reproduce published scores from shipped artefacts")
    v.add_argument("--bench", default=None,
                   help="directory holding results/<cohort>/; defaults to the "
                        "artefacts shipped next to this package")
    v.add_argument("--cohort", default="grain")
    v.add_argument("--report", default=None)

    r = sub.add_parser("run", help="run the frozen chain on a cohort bundle")
    r.add_argument("--bundle", required=True, help="directory produced by make_bundle")
    r.add_argument("--protocol", default="section", choices=("section", "stage"))
    r.add_argument("--out", required=True)
    r.add_argument("--force", nargs="*", default=None,
                   help="gene IDs to force into every fold's panel")

    args = ap.parse_args(argv)
    if args.command == "verify":
        bench = str(args.bench) if args.bench else str(default_bench())
        return verify.main(["--bench", bench, "--cohort", args.cohort]
                           + (["--report", args.report] if args.report else []))
    if args.command == "run":
        from . import data, run
        sections, config = data.load_bundle(args.bundle)
        rows = run.run_cohort(sections, config, protocol=args.protocol, force=args.force)
        run.save_folds(rows, args.out)
        summary = {row["section"]: round(row["r"], 4) for row in rows}
        print(json.dumps(summary, indent=1))
        print(f"mean r = {sum(row['r'] for row in rows) / len(rows):.4f}")
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())

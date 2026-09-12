"""CLI: riproduce una stima da un CSV e un file .dag, senza scrivere codice."""

from __future__ import annotations

import argparse
import sys

import pandas as pd

from .api import estimate_ate
from .graph import DAG


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="ateflow", description=__doc__)
    p.add_argument("--data", required=True, help="percorso del CSV")
    p.add_argument("--dag", required=True, help="percorso del file .dag")
    p.add_argument("--treatment", required=True)
    p.add_argument("--outcome", required=True)
    p.add_argument("--method", default="g-computation",
                   choices=["g-computation", "stratification"])
    p.add_argument("--adjust", nargs="*", default=None,
                   help="forza l'insieme di aggiustamento (viene validato)")
    p.add_argument("--boot", type=int, default=500, help="ricampionamenti bootstrap, 0 per saltare")
    p.add_argument("--no-refute", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    data = pd.read_csv(args.data)
    dag = DAG.from_file(args.dag)
    try:
        result = estimate_ate(
            data,
            dag,
            treatment=args.treatment,
            outcome=args.outcome,
            method=args.method,
            adjustment_set=args.adjust,
            n_boot=args.boot,
            refute=not args.no_refute,
            seed=args.seed,
        )
    except ValueError as exc:
        print(f"errore: {exc}", file=sys.stderr)
        return 2
    print(result.report())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

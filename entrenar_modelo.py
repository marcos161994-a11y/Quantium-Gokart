"""Entrena Quantium NHL desde cero y deja el modelo, el backtest y los picks."""

from __future__ import annotations

import argparse

from picks_nhl import generar_picks
from reentrenar_nhl import reentrenar


def main() -> None:
    parser = argparse.ArgumentParser(description="Backtest walk-forward y primer modelo")
    parser.add_argument("--min-entrenamiento", type=int, default=500)
    parser.add_argument("--sin-picks", action="store_true")
    parser.add_argument("--sin-red", action="store_true", help="Picks sin consultar porteros en vivo")
    args = parser.parse_args()
    reentrenar(min_entrenamiento=args.min_entrenamiento)
    if not args.sin_picks:
        generar_picks(en_vivo=not args.sin_red)


if __name__ == "__main__":
    main()

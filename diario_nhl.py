"""Tarea diaria: resultados, calificación, reentreno y picks.

En CI se llama con --commit para dejar data/ en git. No usa /var/data.
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

from comun_nhl import iniciar_sentry

ARCHIVOS = [
    "data/games.csv",
    "data/picks.json",
    "data/picks_archivo.json",
    "data/modelo.joblib",
    "data/reentrenos.json",
    "data/backtest.json",
    "data/porteros_referencia.json",
]


def _git(*args: str, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], check=check, text=True, capture_output=True)


def confirmar_data() -> bool:
    """Hace commit de los artefactos si cambiaron. No hace push."""
    email = _git("config", "--get", "user.email", check=False).stdout.strip()
    nombre = _git("config", "--get", "user.name", check=False).stdout.strip()
    if not email:
        _git("config", "user.email", "quantium-nhl@users.noreply.github.com")
    if not nombre:
        _git("config", "user.name", "quantium-nhl")
    presentes = [ruta for ruta in ARCHIVOS if Path(ruta).exists()]
    if not presentes:
        print("no hay artefactos para commitear", flush=True)
        return False
    _git("add", "--", *presentes)
    diff = _git("diff", "--cached", "--quiet", check=False)
    if diff.returncode == 0:
        print("data sin cambios", flush=True)
        return False
    _git("commit", "-m", "datos: actualiza resultados, picks y modelo de la NHL")
    print("commit de data/ listo", flush=True)
    return True


def confirmar_picks() -> bool:
    """Commit solo de data/picks.json. El recálculo de porteros no toca el modelo."""
    email = _git("config", "--get", "user.email", check=False).stdout.strip()
    nombre = _git("config", "--get", "user.name", check=False).stdout.strip()
    if not email:
        _git("config", "user.email", "quantium-nhl@users.noreply.github.com")
    if not nombre:
        _git("config", "user.name", "quantium-nhl")
    rutas = [ruta for ruta in ("data/picks.json", "data/picks_archivo.json") if Path(ruta).exists()]
    if "data/picks.json" not in rutas:
        print("no hay data/picks.json", flush=True)
        return False
    _git("add", "--", *rutas)
    diff = _git("diff", "--cached", "--quiet", check=False)
    if diff.returncode == 0:
        print("picks sin cambios", flush=True)
        return False
    _git("commit", "-m", "datos: recalcula picks con los porteros confirmados")
    print("commit de data/picks.json listo", flush=True)
    return True


def correr(completo: bool = False, commit: bool = False, sin_red_picks: bool = False) -> None:
    iniciar_sentry()
    from datos_nhl import actualizar_datos
    from picks_nhl import generar_picks
    from reentrenar_nhl import reentrenar

    try:
        actualizar_datos(completo=completo)
    except Exception as exc:
        print(f"no se pudo refrescar la cartelera: {exc}", flush=True)
        if not Path("data/games.csv").exists():
            raise
    reentrenar()
    generar_picks(en_vivo=not sin_red_picks)
    if commit:
        confirmar_data()


def main() -> None:
    parser = argparse.ArgumentParser(description="Ciclo diario de Quantium NHL")
    parser.add_argument("--completo", action="store_true", help="Vuelve a bajar el histórico")
    parser.add_argument("--commit", action="store_true", help="Commit de data/ si hubo cambios")
    parser.add_argument("--sin-red", action="store_true")
    args = parser.parse_args()
    correr(completo=args.completo, commit=args.commit, sin_red_picks=args.sin_red)


if __name__ == "__main__":
    main()

"""Reentreno diario.

Se entrena un candidato sin los últimos partidos de examen y se compara
su log loss con el del modelo que ya está guardado, en esos mismos
partidos. Solo si mejora se promueve un ajuste entrenado con todo el
histórico. El backtest walk-forward se vuelve a escribir siempre: es el
informe, no el modelo de producción.
"""

from __future__ import annotations

import math
from datetime import datetime, timezone

import joblib
import numpy as np

from comun_nhl import directorio_datos, guardar_json, leer_json
from datos_nhl import cargar_juegos
from modelo_nhl import (
    correr_backtest,
    entrenar_bundle,
    filas_entrenables,
    log_loss_bin,
    probabilidades_local,
    _entrenar,
)

N_EXAMEN = 400


def conviene_promover(log_loss_nuevo: float, log_loss_actual: float | None) -> bool:
    if log_loss_actual is None or (isinstance(log_loss_actual, float) and math.isnan(log_loss_actual)):
        return True
    if log_loss_nuevo is None or (isinstance(log_loss_nuevo, float) and math.isnan(log_loss_nuevo)):
        return False
    return float(log_loss_nuevo) + 1e-5 < float(log_loss_actual)


def _compactar(valor):
    if isinstance(valor, (np.floating, float)):
        numero = float(valor)
        if math.isnan(numero) or math.isinf(numero):
            return None
        return round(numero, 6)
    if isinstance(valor, (np.integer, int)) and not isinstance(valor, bool):
        return int(valor)
    if isinstance(valor, dict):
        return {str(k): _compactar(v) for k, v in valor.items()}
    if isinstance(valor, (list, tuple)):
        return [_compactar(v) for v in valor]
    if valor is None or isinstance(valor, str):
        return valor
    return valor


def _cargar_bundle():
    ruta = directorio_datos() / "modelo.joblib"
    if not ruta.exists():
        return None
    try:
        return joblib.load(ruta)
    except Exception:
        return None


def _log_loss_en_examen(filas, g0, bundle_previo) -> dict:
    entrenables = filas_entrenables(filas)
    if len(entrenables) < 30 or entrenables["gano_local"].nunique() < 2:
        raise RuntimeError("No hay partidos suficientes para reentrenar.")
    n_examen = min(N_EXAMEN, max(20, len(entrenables) // 5))
    if len(entrenables) - n_examen < 40:
        n_examen = max(10, len(entrenables) // 5)
    base = entrenables.iloc[:-n_examen]
    examen = entrenables.iloc[-n_examen:]
    if base["gano_local"].nunique() < 2:
        raise RuntimeError("El tramo de entrenamiento tiene una sola clase.")
    pipe = _entrenar(base)
    candidato = {"pipeline": pipe, "g0": g0}
    ll_nuevo = log_loss_bin(examen["gano_local"], probabilidades_local(candidato, examen))
    ll_viejo = None
    if bundle_previo is not None:
        try:
            ll_viejo = log_loss_bin(examen["gano_local"], probabilidades_local(bundle_previo, examen))
        except Exception as exc:
            print(f"el modelo anterior no se pudo puntuar: {exc}", flush=True)
            ll_viejo = None
    return {
        "log_loss_nuevo": ll_nuevo,
        "log_loss_anterior": ll_viejo,
        "n_examen": int(len(examen)),
        "n_base": int(len(base)),
    }


def reentrenar(min_entrenamiento: int = 500) -> dict:
    juegos = cargar_juegos()
    referencia = leer_json(directorio_datos() / "porteros_referencia.json", {})
    informe = correr_backtest(juegos, referencia, min_entrenamiento=min_entrenamiento)
    resumen = informe.get("resumen") or {}
    guardar_json(directorio_datos() / "backtest.json", _compactar(resumen))
    g0 = resumen.get("g0")
    umbral = float(resumen.get("umbral_alta") or 0.62)
    filas = informe.get("filas")
    if filas is None or filas.empty:
        raise RuntimeError("El backtest no produjo filas entrenables.")
    previo = _cargar_bundle()
    medicion = _log_loss_en_examen(filas, g0, previo)
    promovido = conviene_promover(medicion["log_loss_nuevo"], medicion["log_loss_anterior"])
    if promovido:
        bundle = entrenar_bundle(filas, g0, umbral, medicion["log_loss_nuevo"])
        joblib.dump(bundle, directorio_datos() / "modelo.joblib")
    registro = leer_json(directorio_datos() / "reentrenos.json", {"reentrenos": []})
    entrada = {
        "fecha": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "promovido": promovido,
        "log_loss_nuevo": round(float(medicion["log_loss_nuevo"]), 6),
        "log_loss_anterior": None
        if medicion["log_loss_anterior"] is None
        else round(float(medicion["log_loss_anterior"]), 6),
        "n_examen": medicion["n_examen"],
        "g0": g0,
        "umbral_alta": umbral,
        "log_loss_backtest": None if resumen.get("log_loss_modelo") is None else round(float(resumen["log_loss_modelo"]), 6),
        "n_backtest": resumen.get("n"),
    }
    registro.setdefault("reentrenos", []).append(entrada)
    guardar_json(directorio_datos() / "reentrenos.json", registro)
    print(
        f"reentreno promovido={promovido} ll_nuevo={entrada['log_loss_nuevo']} "
        f"ll_anterior={entrada['log_loss_anterior']} umbral={umbral} g0={g0}",
        flush=True,
    )
    return entrada


def main() -> None:
    reentrenar()


if __name__ == "__main__":
    main()

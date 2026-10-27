"""Elo + regresión logística.

La etiqueta es ganar el partido, prórroga y shootout incluidos: es el
mismo suceso que paga el moneyline. Las features se arman antes de
conocer el resultado. El encogimiento hacia 50% al inicio de temporada
solo se aplica si la validación interna lo prefiere frente a no encoger.
"""

from __future__ import annotations

from datetime import date, datetime, timezone

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from elo_nhl import EloNHL
from porteros_nhl import referencia_de

COLUMNAS = [
    "diff_elo",
    "local",
    "b2b_local",
    "b2b_visita",
    "descanso_local",
    "descanso_visita",
    "diff_sv",
    "diff_gsax",
]

GRILLA_G0 = (None, 8, 12, 16, 24, 40)


def _fecha(valor) -> date | None:
    if valor is None or (isinstance(valor, float) and np.isnan(valor)):
        return None
    if isinstance(valor, date) and not isinstance(valor, datetime):
        return valor
    texto = str(valor)[:10]
    if not texto or texto in {"NaT", "nan", "None"}:
        return None
    return date.fromisoformat(texto)


def _descanso(ultima: date | None, fecha: date) -> tuple[float, int]:
    if ultima is None:
        return 4.0, 0
    dias = (fecha - ultima).days
    b2b = 1 if dias <= 1 else 0
    return float(min(max(dias, 1), 6)), b2b


def aplicar_encogimiento(probabilidad: float, n_muestra: float, g0) -> float:
    """Acerca la probabilidad a 50% cuando los equipos apenas han jugado.

    Con g0 vacío no hay encogimiento. En pretemporada n_muestra es 0,
    así que un g0 activo deja el partido en 50%.
    """
    if not g0:
        return float(probabilidad)
    peso = max(0.0, min(1.0, float(n_muestra) / float(g0)))
    return 0.5 + (float(probabilidad) - 0.5) * peso


def log_loss_bin(y, p) -> float:
    y_arr = np.asarray(y, dtype=float)
    p_arr = np.clip(np.asarray(p, dtype=float), 1e-6, 1.0 - 1e-6)
    if len(y_arr) == 0:
        return float("nan")
    return float(-(y_arr * np.log(p_arr) + (1.0 - y_arr) * np.log(1.0 - p_arr)).mean())


def brier_bin(y, p) -> float:
    y_arr = np.asarray(y, dtype=float)
    p_arr = np.asarray(p, dtype=float)
    if len(y_arr) == 0:
        return float("nan")
    return float(np.mean((p_arr - y_arr) ** 2))


def nuevo_pipeline() -> Pipeline:
    return Pipeline(
        [
            ("escala", StandardScaler()),
            (
                "logit",
                LogisticRegression(C=1.0, max_iter=800, solver="lbfgs", random_state=0),
            ),
        ]
    )


def construir_filas(juegos: pd.DataFrame, referencia: dict | None = None) -> pd.DataFrame:
    """Una fila por partido, con Elo y descanso calculados solo con el pasado."""
    referencia = referencia or {}
    if juegos.empty:
        return pd.DataFrame()
    orden = juegos.sort_values(["inicio_utc", "id"], kind="mergesort")
    elo = EloNHL()
    ultima: dict[str, date] = {}
    jugados: dict[tuple[int, str], int] = {}
    filas = []
    for partido in orden.to_dict(orient="records"):
        tipo = int(partido.get("tipo") or 0)
        if tipo not in (1, 2, 3):
            continue
        fecha = _fecha(partido.get("fecha"))
        if fecha is None:
            continue
        local = str(partido["local"])
        visita = str(partido["visita"])
        temporada = int(partido.get("temporada") or 0)
        if tipo in (2, 3):
            elo.abrir_temporada(temporada)
        neutral = bool(int(partido.get("neutral") or 0))
        descanso_local, b2b_local = _descanso(ultima.get(local), fecha)
        descanso_visita, b2b_visita = _descanso(ultima.get(visita), fecha)
        ref_local, liga_local = referencia_de(referencia, temporada, local)
        ref_visita, liga_visita = referencia_de(referencia, temporada, visita)
        liga = liga_local if ref_local or not ref_visita else liga_visita
        sv_local = None if ref_local is None else float(ref_local["sv"])
        sv_visita = None if ref_visita is None else float(ref_visita["sv"])
        gsax_local = 0.0 if ref_local is None else float(ref_local["gsax"])
        gsax_visita = 0.0 if ref_visita is None else float(ref_visita["gsax"])
        base_local = liga if sv_local is None else sv_local
        base_visita = liga if sv_visita is None else sv_visita
        if tipo == 3:
            n_muestra = 82
        elif tipo == 2:
            n_local = jugados.get((temporada, local), 0)
            n_visita = jugados.get((temporada, visita), 0)
            n_muestra = int(min(n_local, n_visita))
        else:
            n_muestra = 0
        gano = partido.get("gano_local")
        final = gano is not None and not (isinstance(gano, float) and np.isnan(gano))
        filas.append(
            {
                "id": int(partido["id"]),
                "temporada": temporada,
                "tipo": tipo,
                "fecha": fecha.isoformat(),
                "inicio_utc": partido.get("inicio_utc") or "",
                "equipo_local": local,
                "equipo_visita": visita,
                "neutral": int(neutral),
                "final": int(final),
                "periodo": partido.get("periodo") or "",
                "gano_local": int(gano) if final else np.nan,
                "diff_elo": elo.obtener(local) - elo.obtener(visita),
                "local": 0 if neutral else 1,
                "b2b_local": b2b_local,
                "b2b_visita": b2b_visita,
                "descanso_local": descanso_local,
                "descanso_visita": descanso_visita,
                "diff_sv": base_local - base_visita,
                "diff_gsax": gsax_local - gsax_visita,
                "sv_ref_local": sv_local if sv_local is not None else np.nan,
                "sv_ref_visita": sv_visita if sv_visita is not None else np.nan,
                "n_muestra": n_muestra,
                "prob_mercado_local": partido.get("prob_mercado_local"),
                "cuota_local": partido.get("cuota_local"),
                "cuota_visita": partido.get("cuota_visita"),
                "linea": partido.get("linea") or "",
                "casa": partido.get("casa") or "",
                "estado": partido.get("estado") or "",
            }
        )
        if final:
            ultima[local] = fecha
            ultima[visita] = fecha
            if tipo in (2, 3):
                elo.actualizar(
                    local,
                    visita,
                    int(partido["goles_local"]),
                    int(partido["goles_visita"]),
                    str(partido.get("periodo") or "REG"),
                    neutral=neutral,
                    pretemporada=False,
                )
            if tipo == 2:
                jugados[(temporada, local)] = jugados.get((temporada, local), 0) + 1
                jugados[(temporada, visita)] = jugados.get((temporada, visita), 0) + 1
    return pd.DataFrame(filas)


def filas_entrenables(filas: pd.DataFrame) -> pd.DataFrame:
    if filas.empty:
        return filas
    mascara = filas["final"].eq(1) & filas["tipo"].isin([2, 3]) & filas["gano_local"].notna()
    return filas.loc[mascara].sort_values(["inicio_utc", "id"], kind="mergesort").reset_index(drop=True)


def _matriz(filas: pd.DataFrame) -> np.ndarray:
    return filas[COLUMNAS].fillna(0.0).to_numpy(dtype=float)


def _entrenar(filas: pd.DataFrame) -> Pipeline:
    pipe = nuevo_pipeline()
    pipe.fit(_matriz(filas), filas["gano_local"].to_numpy(dtype=int))
    return pipe


def _predecir_crudo(pipe: Pipeline, filas: pd.DataFrame) -> np.ndarray:
    return pipe.predict_proba(_matriz(filas))[:, 1]


def elegir_g0(probabilidades, y, n_muestra) -> tuple:
    """Elige el encogimiento con mejor log loss. None significa no encoger."""
    mejor_g0 = None
    mejor_ll = float("inf")
    for g0 in GRILLA_G0:
        ajustadas = [
            aplicar_encogimiento(p, n, g0) for p, n in zip(probabilidades, n_muestra, strict=True)
        ]
        ll = log_loss_bin(y, ajustadas)
        if ll < mejor_ll:
            mejor_ll = ll
            mejor_g0 = g0
    return mejor_g0, mejor_ll


def correr_backtest(juegos: pd.DataFrame, referencia: dict | None = None, min_entrenamiento: int = 500) -> dict:
    """Walk-forward por temporada. Cada pliegue elige su encogimiento en el tramo previo."""
    filas = filas_entrenables(construir_filas(juegos, referencia))
    if filas.empty:
        return {"oos": filas, "resumen": {}}
    temporadas = [int(t) for t in filas["temporada"].drop_duplicates().tolist()]
    pedazos = []
    decisiones = []
    for temporada in temporadas:
        entrenamiento = filas[filas["temporada"] < temporada]
        examen = filas[filas["temporada"] == temporada]
        if len(entrenamiento) < min_entrenamiento or examen.empty:
            continue
        if entrenamiento["gano_local"].nunique() < 2:
            continue
        corte = max(40, int(len(entrenamiento) * 0.25))
        corte = min(corte, len(entrenamiento) // 3)
        g0 = None
        if corte >= 30 and len(entrenamiento) - corte >= 40:
            base = entrenamiento.iloc[:-corte]
            val = entrenamiento.iloc[-corte:]
            if base["gano_local"].nunique() >= 2:
                pipe_val = _entrenar(base)
                crudo = _predecir_crudo(pipe_val, val)
                g0, _ = elegir_g0(crudo, val["gano_local"].to_numpy(), val["n_muestra"].to_numpy())
        pipe = _entrenar(entrenamiento)
        crudo = _predecir_crudo(pipe, examen)
        ajustadas = np.array(
            [aplicar_encogimiento(p, n, g0) for p, n in zip(crudo, examen["n_muestra"].to_numpy(), strict=True)]
        )
        bloque = examen.copy()
        bloque["p_modelo"] = ajustadas
        bloque["g0_pliegue"] = g0 if g0 is not None else 0
        pedazos.append(bloque)
        decisiones.append(
            {
                "temporada": int(temporada),
                "n_entrenamiento": int(len(entrenamiento)),
                "n_examen": int(len(examen)),
                "g0": g0,
                "log_loss": log_loss_bin(examen["gano_local"], ajustadas),
                "log_loss_sin_encoger": log_loss_bin(examen["gano_local"], crudo),
                "brier": brier_bin(examen["gano_local"], ajustadas),
            }
        )
    if not pedazos:
        return {"oos": filas.iloc[0:0].copy(), "resumen": {}, "pliegues": decisiones}
    oos = pd.concat(pedazos, ignore_index=True)
    resumen = _resumen_oos(oos, decisiones)
    return {"oos": oos, "resumen": resumen, "pliegues": decisiones, "filas": filas}


def elegir_umbral(prob_ganador, aciertos, minimo: int = 80) -> float:
    """Umbral de confianza alta. Si ninguno tiene muestra, se queda en 62%."""
    mejor = (0.62, -1.0, 0)
    prob = np.asarray(prob_ganador, dtype=float)
    ok = np.asarray(aciertos, dtype=float)
    for umbral in (0.58, 0.60, 0.62, 0.64, 0.66, 0.68, 0.70):
        mascara = prob >= umbral - 1e-12
        n = int(mascara.sum())
        if n < minimo:
            continue
        tasa = float(ok[mascara].mean()) if n else 0.0
        if tasa > mejor[1] + 1e-12 or (abs(tasa - mejor[1]) <= 1e-12 and n > mejor[2]):
            mejor = (umbral, tasa, n)
    return float(mejor[0])


def nivel_de(probabilidad: float, umbral: float) -> str:
    piso = min(0.55, umbral - 0.01)
    if probabilidad + 1e-12 >= umbral:
        return "alta"
    if probabilidad + 1e-12 >= piso:
        return "media"
    return "baja"


def _cubeta(prob_ganador, aciertos, umbral: float) -> dict:
    prob = np.asarray(prob_ganador, dtype=float)
    ok = np.asarray(aciertos, dtype=float)
    salida = {}
    for nombre in ("alta", "media", "baja"):
        mascara = np.array([nivel_de(p, umbral) == nombre for p in prob])
        total = int(mascara.sum())
        acierto = int(ok[mascara].sum()) if total else 0
        salida[nombre] = {
            "aciertos": acierto,
            "total": total,
            "pct": round(acierto / total, 4) if total else None,
        }
    return salida


def _mercado(oos: pd.DataFrame) -> dict:
    sub = oos[pd.to_numeric(oos["prob_mercado_local"], errors="coerce").notna()].copy()
    if sub.empty:
        return {"n": 0}
    y = sub["gano_local"].to_numpy()
    p_mod = sub["p_modelo"].to_numpy()
    p_mkt = pd.to_numeric(sub["prob_mercado_local"], errors="coerce").to_numpy()
    dk = sub["casa"].astype(str).str.lower().str.replace(" ", "", regex=False).str.contains("draftking")
    resultado = {
        "n": int(len(sub)),
        "log_loss_modelo": log_loss_bin(y, p_mod),
        "log_loss_mercado": log_loss_bin(y, p_mkt),
        "brier_modelo": brier_bin(y, p_mod),
        "brier_mercado": brier_bin(y, p_mkt),
    }
    if int(dk.sum()) >= 30:
        resultado["n_draftkings"] = int(dk.sum())
        resultado["log_loss_modelo_dk"] = log_loss_bin(y[dk.to_numpy()], p_mod[dk.to_numpy()])
        resultado["log_loss_mercado_dk"] = log_loss_bin(y[dk.to_numpy()], p_mkt[dk.to_numpy()])
        resultado["brier_modelo_dk"] = brier_bin(y[dk.to_numpy()], p_mod[dk.to_numpy()])
        resultado["brier_mercado_dk"] = brier_bin(y[dk.to_numpy()], p_mkt[dk.to_numpy()])
    return resultado


def _resumen_oos(oos: pd.DataFrame, pliegues: list[dict]) -> dict:
    y = oos["gano_local"].to_numpy()
    p = oos["p_modelo"].to_numpy()
    prob_ganador = np.maximum(p, 1.0 - p)
    acierto = ((p >= 0.5) == (y == 1)).astype(float)
    conteo = oos.groupby("temporada").size().to_dict()
    completas = sorted(int(t) for t, n in conteo.items() if n >= 900)
    examen = completas[-1] if completas else None
    if examen is None:
        previas = oos
        examen_df = oos.iloc[0:0]
    else:
        previas = oos[oos["temporada"] < examen]
        examen_df = oos[oos["temporada"] == examen]
    minimo = 40 if len(oos) < 500 else 80
    if len(previas) >= minimo:
        umbral = elegir_umbral(
            np.maximum(previas["p_modelo"], 1 - previas["p_modelo"]),
            ((previas["p_modelo"] >= 0.5) == (previas["gano_local"] == 1)).astype(float),
            minimo=minimo,
        )
    else:
        umbral = elegir_umbral(prob_ganador, acierto, minimo=minimo)
    # g0 de producción: el del pliegue más reciente con entrenamiento decente.
    # Corresponde a la validación previa a la temporada que estamos prediciendo.
    g0 = None
    for pliegue in pliegues:
        g0 = pliegue["g0"]
    en_curso = None
    if examen is not None:
        posteriores = [p for p in pliegues if int(p["temporada"]) > examen]
        if posteriores:
            en_curso = posteriores[-1]["temporada"]
            g0 = posteriores[-1]["g0"]
    return {
        "n": int(len(oos)),
        "log_loss_modelo": log_loss_bin(y, p),
        "brier_modelo": brier_bin(y, p),
        "acierto_modelo": float(acierto.mean()),
        "umbral_alta": umbral,
        "temporada_examen": examen,
        "g0": g0,
        "mercado": _mercado(oos),
        "por_nivel_oos": _cubeta(prob_ganador, acierto, umbral),
        "por_nivel_examen": _cubeta(
            np.maximum(examen_df["p_modelo"], 1 - examen_df["p_modelo"]) if len(examen_df) else [],
            ((examen_df["p_modelo"] >= 0.5) == (examen_df["gano_local"] == 1)).astype(float) if len(examen_df) else [],
            umbral,
        )
        if len(examen_df)
        else {},
        "temporada_en_curso": en_curso,
        "pliegues": pliegues,
    }


def probabilidades_local(bundle: dict, filas: pd.DataFrame) -> np.ndarray:
    crudo = _predecir_crudo(bundle["pipeline"], filas)
    g0 = bundle.get("g0")
    return np.array(
        [aplicar_encogimiento(p, n, g0) for p, n in zip(crudo, filas["n_muestra"].to_numpy(), strict=True)]
    )


def entrenar_bundle(filas_todas: pd.DataFrame, g0, umbral: float, log_loss_examen: float | None) -> dict:
    entrenables = filas_entrenables(filas_todas) if "final" in filas_todas.columns else filas_todas
    pipe = _entrenar(entrenables)
    return {
        "pipeline": pipe,
        "columnas": list(COLUMNAS),
        "g0": g0,
        "umbral_alta": float(umbral),
        "log_loss_examen": log_loss_examen,
        "n_entrenamiento": int(len(entrenables)),
        "entrenado": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
    }


def examen_log_loss(filas: pd.DataFrame, bundle: dict, n_examen: int = 400) -> tuple[float, int]:
    """Log loss del bundle en los últimos partidos, sin reentrenar."""
    entrenables = filas_entrenables(filas) if "final" in filas.columns else filas
    n_examen = min(n_examen, max(1, len(entrenables) // 5))
    examen = entrenables.iloc[-n_examen:]
    probs = probabilidades_local(bundle, examen)
    return log_loss_bin(examen["gano_local"], probs), int(len(examen))

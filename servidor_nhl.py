"""Panel de Quantium NHL.

/api/health responde enseguida.
/api/panel devuelve la jornada vigente en America/Puerto_Rico.
"""

from __future__ import annotations

import os
from datetime import timedelta

from fastapi import FastAPI
from fastapi.responses import FileResponse

from comun_nhl import RAIZ, directorio_datos, en_puerto_rico, fecha_de_hoy, iniciar_sentry, leer_json

iniciar_sentry()

app = FastAPI(title="Quantium NHL", version="1.0.0")
HTML = RAIZ / "QuantiumNHL.html"


def _compactar_backtest(back: dict) -> dict:
    if not back:
        return {}
    mercado = back.get("mercado") or {}
    return {
        "n": back.get("n"),
        "log_loss_modelo": back.get("log_loss_modelo"),
        "brier_modelo": back.get("brier_modelo"),
        "umbral_alta": back.get("umbral_alta"),
        "temporada_examen": back.get("temporada_examen"),
        "g0": back.get("g0"),
        "acierto_modelo": back.get("acierto_modelo"),
        "mercado": {
            "n": mercado.get("n"),
            "log_loss_modelo": mercado.get("log_loss_modelo"),
            "log_loss_mercado": mercado.get("log_loss_mercado"),
            "brier_modelo": mercado.get("brier_modelo"),
            "brier_mercado": mercado.get("brier_mercado"),
            "n_draftkings": mercado.get("n_draftkings"),
            "log_loss_modelo_dk": mercado.get("log_loss_modelo_dk"),
            "log_loss_mercado_dk": mercado.get("log_loss_mercado_dk"),
        },
        "por_nivel_examen": back.get("por_nivel_examen") or {},
        "por_nivel_oos": back.get("por_nivel_oos") or {},
    }


@app.get("/api/health")
def health() -> dict:
    return {"ok": True, "servicio": "quantium-nhl"}


@app.get("/api/panel")
def panel() -> dict:
    """Picks de la jornada vigente. De madrugada sigue siendo el día anterior."""
    hoy = fecha_de_hoy()
    fecha = hoy.isoformat()
    madrugada = en_puerto_rico().hour < 6
    documento = leer_json(directorio_datos() / "picks.json", {})
    por_fecha = documento.get("por_fecha") or {}
    if not por_fecha and documento.get("picks"):
        por_fecha = {documento.get("fecha") or fecha: documento.get("picks")}
    proxima = (hoy + timedelta(days=1)).isoformat()
    back = _compactar_backtest(leer_json(directorio_datos() / "backtest.json", {}))
    return {
        "servicio": "quantium-nhl",
        "fecha": fecha,
        "zona": "America/Puerto_Rico",
        "en_madrugada": madrugada,
        "umbral_alta": documento.get("umbral_alta") or back.get("umbral_alta") or 0.62,
        "g0": documento.get("g0", back.get("g0")),
        "picks": por_fecha.get(fecha) or [],
        "proxima_fecha": proxima if madrugada else None,
        "proxima": por_fecha.get(proxima) if madrugada else None,
        "record": documento.get("record") or {},
        "backtest": back,
        "actualizado": documento.get("actualizado"),
    }


@app.get("/")
def inicio():
    if not HTML.exists():
        return {"detalle": "Falta QuantiumNHL.html"}
    return FileResponse(HTML)


def main() -> None:
    import uvicorn

    puerto = int(os.environ.get("PORT", "8741"))
    uvicorn.run("servidor_nhl:app", host="0.0.0.0", port=puerto, reload=False)


if __name__ == "__main__":
    main()

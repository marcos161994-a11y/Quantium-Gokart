"""Picks de ganador.

Cada pick es un equipo y una probabilidad (prórroga y shootout incluidos).
La línea del mercado se guarda al lado para compararla, no como segundo pick.
La confianza alta usa el umbral que mejor funcionó en el backtest; si no
hay muestra suficiente, 62%.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone

import pandas as pd

from comun_nhl import (
    _como_utc,
    directorio_datos,
    en_puerto_rico,
    fecha_de_hoy,
    guardar_json,
    hora_en_puerto_rico,
    leer_json,
    nombre_equipo,
)
from datos_nhl import ESPN_SCOREBOARD, cargar_juegos, evento_desde_espn, obtener_json
from modelo_nhl import construir_filas, nivel_de, probabilidades_local
from porteros_nhl import _sin_acentos, ajustar_probabilidad, emparejar_portero, gsax_de

NIVELES = ("alta", "media", "baja")


def _vacio() -> dict:
    return {nivel: {"aciertos": 0, "total": 0, "pct": None} for nivel in NIVELES}


def cubeta(picks: list[dict]) -> dict:
    grupos = _vacio()
    for pick in picks:
        if pick.get("acierto") is None:
            continue
        nivel = pick.get("nivel") or "baja"
        if nivel not in grupos:
            continue
        grupos[nivel]["total"] += 1
        grupos[nivel]["aciertos"] += int(bool(pick["acierto"]))
    for grupo in grupos.values():
        if grupo["total"]:
            grupo["pct"] = round(grupo["aciertos"] / grupo["total"], 4)
    return grupos


def _doc() -> dict:
    return leer_json(
        directorio_datos() / "picks.json",
        {"por_fecha": {}, "vivo": [], "umbral_alta": 0.62, "record": {}},
    )


def _resultados(juegos: pd.DataFrame) -> dict[int, dict]:
    if juegos.empty:
        return {}
    salida = {}
    for fila in juegos.to_dict(orient="records"):
        salida[int(fila["id"])] = fila
    return salida


def calificar_lista(picks: list[dict], resultados: dict[int, dict]) -> list[dict]:
    """Marca acierto cuando el partido ya tiene ganador. Se puede correr dos veces."""
    for pick in picks:
        juego = resultados.get(int(pick["id"]))
        if not juego:
            continue
        gano = juego.get("gano_local")
        if gano is None or (isinstance(gano, float) and pd.isna(gano)):
            continue
        ganador_real = juego["local"] if int(gano) == 1 else juego["visita"]
        pick["ganador_real"] = ganador_real
        pick["acierto"] = int(pick.get("ganador") == ganador_real)
        pick["estado"] = "final"
        pick["periodo"] = juego.get("periodo") or pick.get("periodo") or ""
    return picks


def _porteros_espn(fecha: str) -> dict[tuple[str, str], dict]:
    compacta = fecha.replace("-", "")
    try:
        payload = obtener_json(
            ESPN_SCOREBOARD.format(fecha=compacta),
            cache_nombre=f"scoreboard/{compacta}.json",
            refrescar=True,
        )
    except Exception as exc:
        print(f"sin probables de {fecha}: {exc}", flush=True)
        return {}
    indice = {}
    for evento in payload.get("events") or []:
        parsed = evento_desde_espn(evento)
        if parsed is None or parsed["fecha"] != fecha:
            continue
        indice[(parsed["visita"], parsed["local"])] = parsed
    return indice


def _ficha_portero(probable: dict | None, stats: dict | None, liga: float) -> dict | None:
    if not probable and not stats:
        return None
    nombre = (probable or {}).get("nombre") or ""
    if stats is not None:
        bruto = stats.get("name")
        if isinstance(bruto, dict) and not nombre:
            nombre = bruto.get("default") or ""
        sv = stats.get("savePctg")
        if sv is None and stats.get("shotsAgainst"):
            tiros = float(stats.get("shotsAgainst") or 0)
            paradas = float(stats.get("saves") or 0)
            sv = (paradas / tiros) if tiros else None
        tiros = stats.get("shotsAgainst")
        paradas = stats.get("saves")
        return {
            "nombre": nombre,
            "estado": (probable or {}).get("estado") or "",
            "sv": None if sv is None else round(float(sv), 4),
            "tiros": None if tiros is None else int(tiros),
            "gsax": None if gsax_de(paradas, tiros, liga) is None else round(gsax_de(paradas, tiros, liga), 2),
            "partidos": stats.get("gamesPlayed"),
        }
    return {"nombre": nombre, "estado": (probable or {}).get("estado") or "", "sv": None, "tiros": None, "gsax": None}


def _stats_del_partido(juego_id: int) -> list[dict]:
    try:
        payload = obtener_json(
            f"https://api-web.nhle.com/v1/gamecenter/{juego_id}/landing",
            cache_nombre=f"landing/{juego_id}.json",
            refrescar=True,
        )
    except Exception as exc:
        print(f"landing {juego_id}: {exc}", flush=True)
        return []
    return (((payload.get("matchup") or {}).get("goalieSeasonStats") or {}).get("goalies")) or []


def _armar_pick(fila: dict, p_local: float, umbral: float, evento: dict | None, liga: float) -> dict:
    probable_local = (evento or {}).get("portero_local")
    probable_visita = (evento or {}).get("portero_visita")
    stats = []
    if probable_local or probable_visita:
        stats = _stats_del_partido(int(fila["id"]))
    ficha_local = None
    ficha_visita = None
    if probable_local or probable_visita or stats:
        crudo_local = emparejar_portero((probable_local or {}).get("apellido") or "", stats)
        crudo_visita = emparejar_portero((probable_visita or {}).get("apellido") or "", stats)
        ficha_local = _ficha_portero(probable_local, crudo_local, liga)
        ficha_visita = _ficha_portero(probable_visita, crudo_visita, liga)
        p_local = ajustar_probabilidad(
            p_local,
            None if not ficha_local else ficha_local.get("sv"),
            None if not ficha_local else ficha_local.get("tiros"),
            None if not ficha_local else ficha_local.get("estado"),
            None if pd.isna(fila.get("sv_ref_local")) else float(fila.get("sv_ref_local")),
            None if not ficha_visita else ficha_visita.get("sv"),
            None if not ficha_visita else ficha_visita.get("tiros"),
            None if not ficha_visita else ficha_visita.get("estado"),
            None if pd.isna(fila.get("sv_ref_visita")) else float(fila.get("sv_ref_visita")),
            liga=liga,
        )
    if p_local >= 0.5:
        ganador = fila["equipo_local"]
        probabilidad = float(p_local)
    else:
        ganador = fila["equipo_visita"]
        probabilidad = float(1.0 - p_local)
    mercado = fila.get("prob_mercado_local")
    mercado_f = None if mercado is None or (isinstance(mercado, float) and pd.isna(mercado)) else float(mercado)
    return {
        "id": str(int(fila["id"])),
        "fecha": fila["fecha"],
        "inicio_utc": fila.get("inicio_utc") or "",
        "hora": hora_en_puerto_rico(fila["inicio_utc"]) if fila.get("inicio_utc") else "",
        "equipo_local": fila["equipo_local"],
        "equipo_visita": fila["equipo_visita"],
        "nombre_local": nombre_equipo(fila["equipo_local"]),
        "nombre_visita": nombre_equipo(fila["equipo_visita"]),
        "ganador": ganador,
        "nombre_ganador": nombre_equipo(ganador),
        "probabilidad": round(probabilidad, 4),
        "probabilidad_local": round(float(p_local), 4),
        "nivel": nivel_de(probabilidad, umbral),
        "linea": fila.get("linea") or "",
        "casa": fila.get("casa") or "",
        "cuota_local": None if pd.isna(fila.get("cuota_local")) else int(fila.get("cuota_local")),
        "cuota_visita": None if pd.isna(fila.get("cuota_visita")) else int(fila.get("cuota_visita")),
        "prob_mercado_local": None if mercado_f is None else round(mercado_f, 4),
        "portero_local": ficha_local,
        "portero_visita": ficha_visita,
        "estado": "final" if int(fila.get("final") or 0) == 1 else "programado",
        "periodo": fila.get("periodo") or "",
        "acierto": None,
    }


def _nombre_portero(ficha: dict | None) -> str:
    if not ficha:
        return ""
    return str(ficha.get("nombre") or "").strip()


def _mismo_portero(antes: str, ahora: str) -> bool:
    """El mismo titular aunque cambie el acento o solo llegue el apellido."""
    uno = _sin_acentos(antes).strip()
    otro = _sin_acentos(ahora).strip()
    if not uno or not otro:
        return True
    if uno == otro:
        return True
    return uno.split()[-1] == otro.split()[-1]


def ya_empezo(pick: dict, ahora: datetime) -> bool:
    """El saque ya pasó: el pick queda congelado para calificarlo."""
    inicio = pick.get("inicio_utc") or ""
    if not inicio:
        return str(pick.get("estado") or "") not in {"", "programado"}
    try:
        momento = _como_utc(str(inicio))
    except ValueError:
        return False
    return momento <= _como_utc(ahora)


def _descarga_incompleta(nuevo: dict, previo: dict) -> bool:
    """ESPN no trajo un titular que ya teníamos. No se borra ni se recalcula."""
    for lado in ("portero_local", "portero_visita"):
        if _nombre_portero(previo.get(lado)) and not _nombre_portero(nuevo.get(lado)):
            return True
    return False


def conservar_portero(nuevo: dict, previo: dict | None, ahora: datetime) -> dict:
    """Recalcula solo si el partido no empezó. Un titular distinto deja la probabilidad anterior."""
    if previo is None:
        return nuevo
    if ya_empezo(previo, ahora):
        return previo
    if _descarga_incompleta(nuevo, previo):
        return previo
    cambio = False
    for lado in ("portero_local", "portero_visita"):
        antes = _nombre_portero(previo.get(lado))
        actual = _nombre_portero(nuevo.get(lado))
        if antes and actual and not _mismo_portero(antes, actual):
            cambio = True
    if cambio:
        nuevo["portero_cambiado"] = True
        if previo.get("portero_cambiado") and previo.get("probabilidad_anterior") is not None:
            nuevo["probabilidad_anterior"] = previo.get("probabilidad_anterior")
            nuevo["portero_local_anterior"] = previo.get("portero_local_anterior") or ""
            nuevo["portero_visita_anterior"] = previo.get("portero_visita_anterior") or ""
        else:
            nuevo["probabilidad_anterior"] = previo.get("probabilidad")
            nuevo["portero_local_anterior"] = _nombre_portero(previo.get("portero_local"))
            nuevo["portero_visita_anterior"] = _nombre_portero(previo.get("portero_visita"))
        return nuevo
    if previo.get("portero_cambiado"):
        nuevo["portero_cambiado"] = True
        nuevo["probabilidad_anterior"] = previo.get("probabilidad_anterior")
        nuevo["portero_local_anterior"] = previo.get("portero_local_anterior") or ""
        nuevo["portero_visita_anterior"] = previo.get("portero_visita_anterior") or ""
    return nuevo


def generar_picks(en_vivo: bool = True, ahora: datetime | None = None, eventos: dict | None = None) -> dict:
    """Escribe data/picks.json para la jornada vigente y la siguiente."""
    import joblib

    from porteros_nhl import SV_LIGA_DEFECTO

    ruta_modelo = directorio_datos() / "modelo.joblib"
    if not ruta_modelo.exists():
        raise FileNotFoundError("No hay data/modelo.joblib. Corre entrenar_modelo.py primero.")
    bundle = joblib.load(ruta_modelo)
    juegos = cargar_juegos()
    referencia = leer_json(directorio_datos() / "porteros_referencia.json", {})
    filas = construir_filas(juegos, referencia)
    umbral = float(bundle.get("umbral_alta") or 0.62)
    ahora = ahora or datetime.now(timezone.utc)
    hoy = fecha_de_hoy(ahora)
    fechas = [hoy.isoformat(), (hoy + timedelta(days=1)).isoformat()]
    previo = _doc()
    previos = {
        str(pick["id"]): pick
        for lista in (previo.get("por_fecha") or {}).values()
        for pick in lista
    }
    por_fecha: dict[str, list] = {}
    for fecha in fechas:
        slate = filas[(filas["fecha"] == fecha) & (filas["tipo"].isin([2, 3]))].copy()
        picks = []
        vistos: set[str] = set()
        if not slate.empty:
            probs = probabilidades_local(bundle, slate)
            if eventos is None:
                cartelera = _porteros_espn(fecha) if en_vivo else {}
            else:
                cartelera = eventos.get(fecha) or {}
            for indice, fila in enumerate(slate.to_dict(orient="records")):
                evento = cartelera.get((fila["equipo_visita"], fila["equipo_local"]))
                nuevo = _armar_pick(fila, float(probs[indice]), umbral, evento, SV_LIGA_DEFECTO)
                anterior = previos.get(nuevo["id"])
                picks.append(conservar_portero(nuevo, anterior, ahora))
                vistos.add(nuevo["id"])
        for pick in previo.get("por_fecha", {}).get(fecha) or []:
            if str(pick["id"]) in vistos:
                continue
            if ya_empezo(pick, ahora):
                picks.append(pick)
        picks.sort(key=lambda pick: pick.get("inicio_utc") or "")
        por_fecha[fecha] = picks
    resultados = _resultados(juegos)
    vivos = calificar_lista(list(previo.get("vivo") or []), resultados)
    ya = {str(p["id"]) for p in vivos}
    for lista in (previo.get("por_fecha") or {}).values():
        for pick in calificar_lista(list(lista), resultados):
            if pick.get("acierto") is None or str(pick["id"]) in ya:
                continue
            if pick["fecha"] >= hoy.isoformat():
                continue
            vivos.append(pick)
            ya.add(str(pick["id"]))
    for fecha, lista in por_fecha.items():
        calificar_lista(lista, resultados)

    backtest = leer_json(directorio_datos() / "backtest.json", {})
    record = {
        "examen": {
            "temporada": backtest.get("temporada_examen"),
            **(backtest.get("por_nivel_examen") or _vacio()),
        },
        "oos": backtest.get("por_nivel_oos") or _vacio(),
        "vivo": cubeta(vivos),
    }
    # El examen trae las cubetas en la raíz si así se guardó el backtest.
    if "alta" not in record["examen"] and backtest.get("por_nivel_examen"):
        record["examen"].update(backtest["por_nivel_examen"])

    documento = {
        "actualizado": en_puerto_rico(ahora).isoformat(),
        "zona": "America/Puerto_Rico",
        "fecha": hoy.isoformat(),
        "umbral_alta": umbral,
        "g0": bundle.get("g0"),
        "por_fecha": por_fecha,
        "picks": por_fecha.get(hoy.isoformat(), []),
        "vivo": vivos,
        "record": record,
    }
    guardar_json(directorio_datos() / "picks.json", documento)
    n = len(documento["picks"])
    print(f"picks {hoy.isoformat()}: {n}", flush=True)
    return documento


def main() -> None:
    parser = argparse.ArgumentParser(description="Genera los picks de la jornada")
    parser.add_argument("--sin-red", action="store_true", help="No consulta porteros probables")
    parser.add_argument(
        "--commit",
        action="store_true",
        help="Commit solo de data/picks.json si el recálculo cambió algo",
    )
    args = parser.parse_args()
    generar_picks(en_vivo=not args.sin_red)
    if args.commit:
        from diario_nhl import confirmar_picks

        confirmar_picks()


if __name__ == "__main__":
    main()

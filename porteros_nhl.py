"""Porteros titulares.

En el histórico el modelo ve al titular de la temporada anterior (el que
más partidos inició). Eso no se entera del resultado de hoy, así que no
hay fuga. En la cartelera del día, si ESPN trae un titular confirmado o
probable, se corrige la probabilidad solo por la sorpresa: cuánto se
aleja ese portero del titular de referencia que el modelo ya conocía.

GSAx se aproxima con paradas por encima del porcentaje de la liga
(paradas - sv_liga * tiros). La NHL no publica GSAx; si hay paradas y
tiros, el número se muestra y entra como diferencia de la temporada previa.
"""

from __future__ import annotations

import math
import unicodedata

from comun_nhl import a_nhl, aliases_franquicia, temporada_anterior

PRIOR_TIROS = 350.0
SV_LIGA_DEFECTO = 0.903
PESO_SV = 7.0  # logit por cada 1.0 de diferencia de sv% (0.01 ≈ 1.7 puntos)


def _sin_acentos(texto: str) -> str:
    base = unicodedata.normalize("NFKD", texto or "")
    return "".join(c for c in base if not unicodedata.combining(c)).lower()


def construir_referencia(resumenes: dict[int, list[dict]]) -> dict:
    """Arma el titular de referencia de cada equipo y temporada."""
    referencia: dict[str, dict] = {}
    for temporada, filas in resumenes.items():
        paradas = sum(int(f.get("saves") or 0) for f in filas)
        tiros = sum(int(f.get("shotsAgainst") or 0) for f in filas)
        liga = (paradas / tiros) if tiros else SV_LIGA_DEFECTO
        mejores: dict[str, dict] = {}
        for fila in filas:
            equipos = [a_nhl(p.strip()) for p in str(fila.get("teamAbbrevs") or "").split(",") if p.strip()]
            if not equipos:
                continue
            exclusivo = len(equipos) == 1
            inicios = int(fila.get("gamesStarted") or 0)
            tiros_p = int(fila.get("shotsAgainst") or 0)
            paradas_p = int(fila.get("saves") or 0)
            sv = fila.get("savePct")
            if sv is None and tiros_p:
                sv = paradas_p / tiros_p
            if sv is None:
                continue
            sv = float(sv)
            gsax = paradas_p - liga * tiros_p
            for equipo in equipos:
                candidato = {
                    "id": fila.get("playerId"),
                    "nombre": fila.get("goalieFullName") or "",
                    "sv": sv,
                    "gsax": gsax,
                    "inicios": inicios,
                    "tiros": tiros_p,
                    "saves": paradas_p,
                    "_clave": (1 if exclusivo else 0, inicios),
                }
                previo = mejores.get(equipo)
                if previo is None or candidato["_clave"] > previo["_clave"]:
                    mejores[equipo] = candidato
        limpio = {}
        for equipo, datos in mejores.items():
            datos = dict(datos)
            datos.pop("_clave", None)
            limpio[equipo] = datos
        limpio["_liga_sv"] = liga
        referencia[str(temporada)] = limpio
    return referencia


def referencia_de(referencia: dict, temporada: int, equipo: str) -> tuple[dict | None, float]:
    bloque = referencia.get(str(temporada_anterior(int(temporada)))) or {}
    liga = float(bloque.get("_liga_sv") or SV_LIGA_DEFECTO)
    for alias in aliases_franquicia(equipo):
        if alias in bloque:
            return bloque[alias], liga
    return None, liga


def sv_encogido(sv: float | None, tiros: float | None, liga: float = SV_LIGA_DEFECTO) -> float:
    """Encoge el sv% hacia la liga. Con pocos tiros casi no nos movemos."""
    if sv is None or (isinstance(sv, float) and math.isnan(sv)):
        return liga
    tiros_f = float(tiros or 0)
    paradas = float(sv) * tiros_f
    return (paradas + liga * PRIOR_TIROS) / (tiros_f + PRIOR_TIROS)


def factor_estado(estado: str | None) -> float:
    texto = (estado or "").strip().lower()
    if texto in {"confirmed", "confirmado", "starting", "starter"}:
        return 1.0
    if texto in {"expected", "probable", "esperado", "probable"}:
        return 0.7
    if not texto:
        return 0.0
    return 0.45


def ajustar_probabilidad(
    probabilidad_local: float,
    sv_real_local: float | None,
    tiros_local: float | None,
    estado_local: str | None,
    sv_ref_local: float | None,
    sv_real_visita: float | None,
    tiros_visita: float | None,
    estado_visita: str | None,
    sv_ref_visita: float | None,
    liga: float = SV_LIGA_DEFECTO,
) -> float:
    """Corrige solo la sorpresa del titular respecto a la referencia.

    Si no hay portero probable en algún lado, la probabilidad no se toca.
    """
    if sv_real_local is None or sv_real_visita is None:
        return float(probabilidad_local)
    peso = min(factor_estado(estado_local), factor_estado(estado_visita))
    if peso <= 0:
        return float(probabilidad_local)
    base_local = liga if sv_ref_local is None else float(sv_ref_local)
    base_visita = liga if sv_ref_visita is None else float(sv_ref_visita)
    sorpresa_local = sv_encogido(sv_real_local, tiros_local, liga) - base_local
    sorpresa_visita = sv_encogido(sv_real_visita, tiros_visita, liga) - base_visita
    delta = PESO_SV * peso * (sorpresa_local - sorpresa_visita)
    p = min(max(float(probabilidad_local), 1e-6), 1.0 - 1e-6)
    logit = math.log(p / (1.0 - p)) + delta
    return 1.0 / (1.0 + math.exp(-logit))


def emparejar_portero(apellido: str, porteros: list[dict]) -> dict | None:
    buscado = _sin_acentos(apellido)
    if not buscado:
        return None
    for portero in porteros:
        nombre = ""
        bruto = portero.get("name")
        if isinstance(bruto, dict):
            nombre = bruto.get("default") or ""
        elif isinstance(bruto, str):
            nombre = bruto
        apellido_api = ""
        last = portero.get("lastName")
        if isinstance(last, dict):
            apellido_api = last.get("default") or ""
        texto = _sin_acentos(f"{nombre} {apellido_api}")
        if buscado in texto.split() or buscado in texto:
            return portero
    return None


def gsax_de(paradas: float | None, tiros: float | None, liga: float) -> float | None:
    if paradas is None or tiros is None:
        return None
    return float(paradas) - float(liga) * float(tiros)

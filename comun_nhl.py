"""Rutas, zona horaria y piezas compartidas de Quantium NHL.

La cartelera no cambia a medianoche. Un partido que empieza de madrugada
en Puerto Rico pertenece a la misma jornada que la noche anterior.
"""

from __future__ import annotations

import json
import os
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

ZONA = ZoneInfo("America/Puerto_Rico")
CORTE_MADRUGADA = 6  # antes de las 06:00 AST sigue siendo la jornada previa

RAIZ = Path(__file__).resolve().parent

# ESPN no usa siempre la abreviatura de la NHL.
ESPN_A_NHL = {
    "ANA": "ANA",
    "ARI": "ARI",
    "PHX": "ARI",
    "BOS": "BOS",
    "BUF": "BUF",
    "CAR": "CAR",
    "CBJ": "CBJ",
    "CGY": "CGY",
    "CHI": "CHI",
    "COL": "COL",
    "DAL": "DAL",
    "DET": "DET",
    "EDM": "EDM",
    "FLA": "FLA",
    "LA": "LAK",
    "LAK": "LAK",
    "MIN": "MIN",
    "MTL": "MTL",
    "MON": "MTL",
    "NJ": "NJD",
    "NJD": "NJD",
    "NSH": "NSH",
    "NAS": "NSH",
    "NYI": "NYI",
    "NYR": "NYR",
    "OTT": "OTT",
    "PHI": "PHI",
    "PIT": "PIT",
    "SEA": "SEA",
    "SJ": "SJS",
    "SJS": "SJS",
    "STL": "STL",
    "TB": "TBL",
    "TBL": "TBL",
    "TOR": "TOR",
    "UTA": "UTA",
    "UTAH": "UTA",
    "VAN": "VAN",
    "VGK": "VGK",
    "VEG": "VGK",
    "WPG": "WPG",
    "WSH": "WSH",
    "WAS": "WSH",
}

# Arizona se mudó a Utah. El Elo sigue a la franquicia, no al nombre.
FRANQUICIA = {"ARI": "UTA"}

NOMBRES = {
    "ANA": "Ducks de Anaheim",
    "ARI": "Coyotes de Arizona",
    "BOS": "Bruins de Boston",
    "BUF": "Sabres de Buffalo",
    "CAR": "Hurricanes de Carolina",
    "CBJ": "Blue Jackets de Columbus",
    "CGY": "Flames de Calgary",
    "CHI": "Blackhawks de Chicago",
    "COL": "Avalanche de Colorado",
    "DAL": "Stars de Dallas",
    "DET": "Red Wings de Detroit",
    "EDM": "Oilers de Edmonton",
    "FLA": "Panthers de Florida",
    "LAK": "Kings de Los Ángeles",
    "MIN": "Wild de Minnesota",
    "MTL": "Canadiens de Montreal",
    "NJD": "Devils de New Jersey",
    "NSH": "Predators de Nashville",
    "NYI": "Islanders de Nueva York",
    "NYR": "Rangers de Nueva York",
    "OTT": "Senators de Ottawa",
    "PHI": "Flyers de Filadelfia",
    "PIT": "Penguins de Pittsburgh",
    "SEA": "Kraken de Seattle",
    "SJS": "Sharks de San José",
    "STL": "Blues de San Luis",
    "TBL": "Lightning de Tampa Bay",
    "TOR": "Maple Leafs de Toronto",
    "UTA": "Mammoth de Utah",
    "VAN": "Canucks de Vancouver",
    "VGK": "Golden Knights de Vegas",
    "WPG": "Jets de Winnipeg",
    "WSH": "Capitals de Washington",
}


def directorio_datos() -> Path:
    """Artefactos que viajan en git: partidos, modelo, picks."""
    override = os.environ.get("QUANTIUM_DATOS", "").strip()
    ruta = Path(override) if override else RAIZ / "data"
    ruta.mkdir(parents=True, exist_ok=True)
    return ruta


def directorio_cache() -> Path:
    """Caché de la API. Nunca /var/data: vive dentro del proyecto.

    En Render la variable DATA_DIR apunta a data/cache del propio servicio.
    """
    override = os.environ.get("DATA_DIR", "").strip()
    ruta = Path(override) if override else directorio_datos() / "cache"
    ruta.mkdir(parents=True, exist_ok=True)
    return ruta


def a_nhl(abreviatura: str) -> str:
    codigo = (abreviatura or "").upper().strip()
    return ESPN_A_NHL.get(codigo, codigo)


def franquicia(equipo: str) -> str:
    return FRANQUICIA.get(equipo, equipo)


def aliases_franquicia(equipo: str) -> list[str]:
    if equipo in ("UTA", "ARI"):
        return ["UTA", "ARI"]
    return [equipo]


def nombre_equipo(equipo: str) -> str:
    return NOMBRES.get(equipo, equipo)


def _como_utc(momento: datetime | str) -> datetime:
    if isinstance(momento, str):
        texto = momento.replace("Z", "+00:00")
        momento = datetime.fromisoformat(texto)
    if momento.tzinfo is None:
        momento = momento.replace(tzinfo=timezone.utc)
    return momento.astimezone(timezone.utc)


def en_puerto_rico(momento: datetime | str | None = None) -> datetime:
    if momento is None:
        base = datetime.now(timezone.utc)
    else:
        base = _como_utc(momento)
    return base.astimezone(ZONA)


def fecha_de_cartelera(inicio: datetime | str, corte: int = CORTE_MADRUGADA) -> date:
    """Fecha de jornada para un partido, en America/Puerto_Rico.

    Si el saque es antes del corte (madrugada), cuenta para el día anterior.
    """
    local = en_puerto_rico(inicio)
    if local.hour < corte:
        return local.date() - timedelta(days=1)
    return local.date()


def fecha_de_hoy(momento: datetime | str | None = None, corte: int = CORTE_MADRUGADA) -> date:
    """Jornada vigente. De madrugada todavía es el día de la cartelera en curso."""
    local = en_puerto_rico(momento)
    if local.hour < corte:
        return local.date() - timedelta(days=1)
    return local.date()


def hora_en_puerto_rico(inicio: datetime | str) -> str:
    local = en_puerto_rico(inicio)
    hora = local.hour % 12 or 12
    sufijo = "am" if local.hour < 12 else "pm"
    return f"{hora}:{local.minute:02d} {sufijo}"


def temporada_anterior(temporada: int) -> int:
    """20242025 -> 20232024."""
    inicio = int(temporada) // 10000
    return (inicio - 1) * 10000 + inicio


def leer_json(ruta: Path, default):
    if not ruta.exists():
        return default
    return json.loads(ruta.read_text(encoding="utf-8"))


def guardar_json(ruta: Path, payload) -> None:
    ruta.parent.mkdir(parents=True, exist_ok=True)
    temporal = ruta.with_suffix(ruta.suffix + ".tmp")
    temporal.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporal.replace(ruta)


def iniciar_sentry() -> bool:
    """Activa Sentry solo si existe SENTRY_DSN. Sin DSN no hace nada."""
    dsn = os.environ.get("SENTRY_DSN", "").strip()
    if not dsn:
        return False
    try:
        import sentry_sdk

        sentry_sdk.init(
            dsn=dsn,
            environment=os.environ.get("SENTRY_ENVIRONMENT", "desarrollo"),
            traces_sample_rate=0.0,
            send_default_pii=False,
        )
        return True
    except Exception:
        return False

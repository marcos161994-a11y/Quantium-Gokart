"""Cartelera, resultados y moneyline.

Resultados: API pública de la NHL (api-web.nhle.com).
Cuotas: moneyline de DraftKings en la API pública de ESPN. Si el cierre
de DraftKings no quedó archivado, se usa el cierre de ESPN BET y se
guarda el nombre de la casa. La probabilidad del mercado se guarda sin
vigorish.

El histórico vive en data/games.csv. La caché de las respuestas crudas
va a data/cache (o a DATA_DIR), nunca a /var/data.
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, timedelta
from urllib.parse import quote

import pandas as pd

from comun_nhl import a_nhl, directorio_cache, directorio_datos, fecha_de_cartelera, fecha_de_hoy
from porteros_nhl import construir_referencia

NHL_SCHEDULE = "https://api-web.nhle.com/v1/schedule/{fecha}"
ESPN_SCOREBOARD = "https://site.api.espn.com/apis/site/v2/sports/hockey/nhl/scoreboard?dates={fecha}&limit=200"
ESPN_ODDS = "https://sports.core.api.espn.com/v2/sports/hockey/leagues/nhl/events/{eid}/competitions/{eid}/odds?limit=40"
NHL_PORTEROS = "https://api.nhle.com/stats/rest/en/goalie/summary?cayenneExp={exp}&limit=500"

COLUMNAS = [
    "id",
    "espn_id",
    "temporada",
    "tipo",
    "fecha",
    "inicio_utc",
    "estado",
    "visita",
    "local",
    "neutral",
    "goles_visita",
    "goles_local",
    "periodo",
    "gano_local",
    "cuota_local",
    "cuota_visita",
    "casa",
    "linea",
    "prob_mercado_local",
]

CAMPOS_CUOTA = {"cuota_local", "cuota_visita", "casa", "linea", "prob_mercado_local", "espn_id"}
ESTADOS_FINAL = {"OFF", "FINAL", "OVER", "OFFICIAL"}
TEMPORADAS_PORTEROS = [
    20202021,
    20212022,
    20222023,
    20232024,
    20242025,
    20252026,
    20262027,
]


def americana(valor) -> int | None:
    """Convierte una cuota americana ('-185', {'american': '+124'}, -130) a entero."""
    if valor is None or isinstance(valor, bool):
        return None
    if isinstance(valor, dict):
        for clave in ("american", "alternateDisplayValue", "odds"):
            if valor.get(clave) not in (None, ""):
                return americana(valor.get(clave))
        return None
    if isinstance(valor, (int, float)):
        numero = int(round(float(valor)))
    else:
        texto = str(valor).strip().replace("−", "-").replace("–", "-")
        if not texto or texto.lower() in {"even", "pk", "pick"}:
            return 100
        if texto.startswith("+"):
            texto = texto[1:]
        try:
            numero = int(round(float(texto)))
        except ValueError:
            return None
    if numero <= -100 or numero >= 100:
        return numero
    return None


def probabilidad_implicita(cuota: int) -> float:
    if cuota < 0:
        return (-cuota) / ((-cuota) + 100.0)
    return 100.0 / (cuota + 100.0)


def sin_vig(prob_local: float, prob_visita: float) -> float:
    total = prob_local + prob_visita
    if total <= 0:
        return 0.5
    return prob_local / total


def texto_linea(local: str, visita: str, cuota_local: int, cuota_visita: int) -> str:
    if cuota_local <= cuota_visita:
        lado, cuota = local, cuota_local
    else:
        lado, cuota = visita, cuota_visita
    signo = "+" if cuota > 0 else ""
    return f"{lado} {signo}{cuota}"


def _puntaje_casa(nombre: str) -> int:
    limpio = (nombre or "").lower().replace(" ", "")
    if "live" in (nombre or "").lower():
        return 0
    if "draftking" in limpio:
        return 3
    if "espnbet" in limpio:
        return 2
    return 1


def _linea_de_lado(bloque: dict | None) -> int | None:
    if not isinstance(bloque, dict):
        return None
    cierre = bloque.get("close") or {}
    if isinstance(cierre, dict):
        for clave in ("moneyLine", "odds"):
            numero = americana(cierre.get(clave))
            if numero is not None:
                return numero
    numero = americana(bloque.get("moneyLine"))
    if numero is not None:
        return numero
    apertura = bloque.get("open") or {}
    if isinstance(apertura, dict):
        for clave in ("moneyLine", "odds"):
            numero = americana(apertura.get(clave))
            if numero is not None:
                return numero
    return None


def extraer_moneylines(item: dict) -> tuple[int, int] | None:
    """Devuelve (cuota_local, cuota_visita) desde el scoreboard o el core de ESPN."""
    moneyline = item.get("moneyline")
    if isinstance(moneyline, dict) and ("home" in moneyline or "away" in moneyline):
        local = _linea_de_lado(moneyline.get("home"))
        visita = _linea_de_lado(moneyline.get("away"))
        if local is not None and visita is not None:
            return local, visita
    local = _linea_de_lado(item.get("homeTeamOdds"))
    visita = _linea_de_lado(item.get("awayTeamOdds"))
    if local is None or visita is None:
        return None
    return local, visita


def elegir_proveedor(items: list[dict]) -> tuple[str, int, int] | None:
    """Prefiere el cierre de DraftKings. Descarta las cuotas en vivo."""
    candidatos = []
    for item in items or []:
        nombre = ((item.get("provider") or {}).get("name")) or ""
        if "live" in nombre.lower():
            continue
        par = extraer_moneylines(item)
        if par is None:
            continue
        candidatos.append((_puntaje_casa(nombre), nombre, par[0], par[1]))
    if not candidatos:
        return None
    candidatos.sort(key=lambda fila: fila[0], reverse=True)
    _, nombre, local, visita = candidatos[0]
    return nombre, local, visita


def aplicar_cuota(juego: dict, eleccion: tuple[str, int, int]) -> None:
    nombre, cuota_local, cuota_visita = eleccion
    juego["cuota_local"] = int(cuota_local)
    juego["cuota_visita"] = int(cuota_visita)
    juego["casa"] = nombre
    juego["linea"] = texto_linea(juego["local"], juego["visita"], cuota_local, cuota_visita)
    juego["prob_mercado_local"] = round(
        sin_vig(probabilidad_implicita(cuota_local), probabilidad_implicita(cuota_visita)),
        6,
    )


def juego_desde_nhl(partido: dict) -> dict | None:
    try:
        visita = a_nhl(partido["awayTeam"]["abbrev"])
        local = a_nhl(partido["homeTeam"]["abbrev"])
    except (KeyError, TypeError):
        return None
    inicio = partido.get("startTimeUTC") or ""
    fecha = fecha_de_cartelera(inicio).isoformat() if inicio else ""
    estado = partido.get("gameState") or ""
    goles_visita = partido.get("awayTeam", {}).get("score")
    goles_local = partido.get("homeTeam", {}).get("score")
    final = estado in ESTADOS_FINAL and goles_visita is not None and goles_local is not None
    periodo = ""
    gano = None
    if final:
        periodo = (
            (partido.get("gameOutcome") or {}).get("lastPeriodType")
            or (partido.get("periodDescriptor") or {}).get("periodType")
            or "REG"
        ).upper()
        if goles_local > goles_visita:
            gano = 1
        elif goles_visita > goles_local:
            gano = 0
        else:
            final = False
            periodo = ""
    return {
        "id": int(partido["id"]),
        "espn_id": None,
        "temporada": int(partido.get("season") or 0),
        "tipo": int(partido.get("gameType") or 0),
        "fecha": fecha,
        "inicio_utc": inicio,
        "estado": estado if final or estado else estado,
        "visita": visita,
        "local": local,
        "neutral": 1 if partido.get("neutralSite") else 0,
        "goles_visita": int(goles_visita) if final else None,
        "goles_local": int(goles_local) if final else None,
        "periodo": periodo if final else "",
        "gano_local": gano if final else None,
        "cuota_local": None,
        "cuota_visita": None,
        "casa": "",
        "linea": "",
        "prob_mercado_local": None,
    }


def mezclar_juego(previo: dict | None, nuevo: dict) -> dict:
    if previo is None:
        return dict(nuevo)
    salida = dict(previo)
    nuevo_es_final = nuevo.get("gano_local") is not None
    previo_es_final = previo.get("gano_local") is not None
    for clave, valor in nuevo.items():
        if clave in CAMPOS_CUOTA:
            continue
        if valor in (None, ""):
            continue
        if previo_es_final and not nuevo_es_final and clave in {
            "goles_local",
            "goles_visita",
            "gano_local",
            "periodo",
            "estado",
        }:
            continue
        salida[clave] = valor
    return salida


def portero_desde_competidor(competidor: dict) -> dict | None:
    for probable in competidor.get("probables") or []:
        nombre = probable.get("name") or ""
        if "goalie" not in nombre.lower() and nombre != "probableStartingGoalie":
            continue
        atleta = probable.get("athlete") or {}
        completo = atleta.get("fullName") or atleta.get("displayName") or ""
        apellido = completo.split()[-1] if completo else ""
        estado = probable.get("status") or {}
        return {
            "nombre": completo,
            "apellido": apellido,
            "estado": (estado.get("type") or estado.get("name") or ""),
            "espn_id": atleta.get("id") or "",
        }
    return None


def evento_desde_espn(evento: dict) -> dict | None:
    competencias = evento.get("competitions") or []
    if not competencias:
        return None
    comp = competencias[0]
    inicio = comp.get("date") or evento.get("date") or ""
    if not inicio:
        return None
    equipos: dict[str, str] = {}
    porteros: dict[str, dict | None] = {}
    for competidor in comp.get("competitors") or []:
        lado = competidor.get("homeAway")
        abrev = a_nhl(((competidor.get("team") or {}).get("abbreviation")) or "")
        if lado and abrev:
            equipos[lado] = abrev
            porteros[lado] = portero_desde_competidor(competidor)
    if "home" not in equipos or "away" not in equipos:
        return None
    return {
        "espn_id": str(evento.get("id")),
        "fecha": fecha_de_cartelera(inicio).isoformat(),
        "inicio": inicio,
        "local": equipos["home"],
        "visita": equipos["away"],
        "odds": comp.get("odds") or [],
        "portero_local": porteros.get("home"),
        "portero_visita": porteros.get("away"),
    }


def obtener_json(url: str, cache_nombre: str | None = None, refrescar: bool = False, timeout: int = 30):
    ruta = None
    if cache_nombre:
        ruta = directorio_cache() / cache_nombre
        if ruta.exists() and not refrescar:
            return json.loads(ruta.read_text(encoding="utf-8"))
    peticion = urllib.request.Request(
        url,
        headers={"User-Agent": "QuantiumNHL/1.0", "Accept": "application/json"},
    )
    ultimo: Exception | None = None
    datos = None
    for intento in range(4):
        try:
            with urllib.request.urlopen(peticion, timeout=timeout) as respuesta:
                datos = json.loads(respuesta.read().decode())
            break
        except urllib.error.HTTPError as exc:
            ultimo = exc
            if exc.code in (429, 500, 502, 503, 504):
                time.sleep(0.7 * (intento + 1))
                continue
            raise
        except Exception as exc:  # red inestable: reintentar
            ultimo = exc
            time.sleep(0.5 * (intento + 1))
    if datos is None:
        raise RuntimeError(f"No se pudo leer {url}") from ultimo
    if ruta is not None:
        ruta.parent.mkdir(parents=True, exist_ok=True)
        ruta.write_text(json.dumps(datos), encoding="utf-8")
    return datos


def _fechas_cada(inicio: date, fin: date, paso: int) -> list[str]:
    fechas = []
    cursor = inicio
    while cursor <= fin:
        fechas.append(cursor.isoformat())
        cursor += timedelta(days=paso)
    return fechas


def descargar_calendario(fechas: list[str], refrescar_desde: str) -> list[dict]:
    def una(fecha: str):
        refrescar = fecha >= refrescar_desde
        return obtener_json(
            NHL_SCHEDULE.format(fecha=fecha),
            cache_nombre=f"schedule/{fecha}.json",
            refrescar=refrescar,
        )

    partidos: dict[int, dict] = {}
    errores = 0
    with ThreadPoolExecutor(max_workers=8) as pool:
        futuros = {pool.submit(una, fecha): fecha for fecha in fechas}
        hechos = 0
        for futuro in as_completed(futuros):
            hechos += 1
            try:
                payload = futuro.result()
            except Exception as exc:
                errores += 1
                print(f"calendario {futuros[futuro]} falló: {exc}", flush=True)
                continue
            for dia in payload.get("gameWeek") or []:
                for partido in dia.get("games") or []:
                    juego = juego_desde_nhl(partido)
                    if juego is None:
                        continue
                    partidos[juego["id"]] = mezclar_juego(partidos.get(juego["id"]), juego)
            if hechos % 40 == 0:
                print(f"calendario {hechos}/{len(fechas)}", flush=True)
    print(f"partidos en calendario: {len(partidos)} (errores {errores})", flush=True)
    return list(partidos.values())


def _necesita_cuota(juego: dict, hoy: str) -> bool:
    if int(juego.get("tipo") or 0) not in (2, 3):
        return False
    abierta = juego.get("gano_local") is None
    if abierta:
        return str(juego.get("fecha") or "") >= hoy
    return juego.get("cuota_local") in (None, "")


def completar_cuotas(juegos: dict[int, dict], hoy: str) -> None:
    pendientes = [j for j in juegos.values() if _necesita_cuota(j, hoy)]
    fechas = sorted({j["fecha"] for j in pendientes if j.get("fecha")})
    if not fechas:
        print("cuotas al día", flush=True)
        return
    print(f"buscando cuotas en {len(fechas)} jornadas ({len(pendientes)} partidos)", flush=True)

    def scoreboard(fecha: str):
        compacta = fecha.replace("-", "")
        refrescar = fecha >= hoy
        return fecha, obtener_json(
            ESPN_SCOREBOARD.format(fecha=compacta),
            cache_nombre=f"scoreboard/{compacta}.json",
            refrescar=refrescar,
        )

    indice: dict[tuple[str, str, str], dict] = {}
    with ThreadPoolExecutor(max_workers=8) as pool:
        futuros = [pool.submit(scoreboard, fecha) for fecha in fechas]
        for futuro in as_completed(futuros):
            try:
                fecha, payload = futuro.result()
            except Exception as exc:
                print(f"scoreboard falló: {exc}", flush=True)
                continue
            for evento in payload.get("events") or []:
                parsed = evento_desde_espn(evento)
                if parsed is None:
                    continue
                indice[(parsed["fecha"], parsed["visita"], parsed["local"])] = parsed

    faltan_core: list[tuple[dict, dict]] = []
    for juego in pendientes:
        clave = (juego["fecha"], juego["visita"], juego["local"])
        evento = indice.get(clave)
        if evento is None:
            continue
        juego["espn_id"] = evento["espn_id"]
        eleccion = elegir_proveedor(evento.get("odds") or [])
        if eleccion is not None and (juego.get("gano_local") is None or juego.get("cuota_local") in (None, "")):
            aplicar_cuota(juego, eleccion)
        if juego.get("cuota_local") in (None, "") or juego.get("gano_local") is None:
            faltan_core.append((juego, evento))

    def odds_de(par: tuple[dict, dict]):
        juego, evento = par
        refrescar = juego.get("gano_local") is None
        payload = obtener_json(
            ESPN_ODDS.format(eid=evento["espn_id"]),
            cache_nombre=f"odds/{evento['espn_id']}.json",
            refrescar=refrescar,
        )
        return juego["id"], payload

    print(f"cierres por partido: {len(faltan_core)}", flush=True)
    with ThreadPoolExecutor(max_workers=10) as pool:
        futuros = {pool.submit(odds_de, par): par[0]["id"] for par in faltan_core}
        hechos = 0
        for futuro in as_completed(futuros):
            hechos += 1
            try:
                juego_id, payload = futuro.result()
            except Exception as exc:
                if hechos % 100 == 0:
                    print(f"odds {hechos} error {exc}", flush=True)
                continue
            eleccion = elegir_proveedor(payload.get("items") or [])
            if eleccion is not None:
                aplicar_cuota(juegos[juego_id], eleccion)
            if hechos % 200 == 0:
                print(f"odds {hechos}/{len(faltan_core)}", flush=True)


def descargar_resumenes_porteros() -> dict[int, list[dict]]:
    resumenes: dict[int, list[dict]] = {}
    for temporada in TEMPORADAS_PORTEROS:
        exp = quote(f"seasonId={temporada} and gameTypeId=2")
        try:
            payload = obtener_json(
                NHL_PORTEROS.format(exp=exp),
                cache_nombre=f"porteros/{temporada}.json",
                refrescar=temporada >= 20252026,
            )
        except Exception as exc:
            print(f"porteros {temporada} no disponibles: {exc}", flush=True)
            continue
        resumenes[temporada] = payload.get("data") or []
        print(f"porteros {temporada}: {len(resumenes[temporada])}", flush=True)
    return resumenes


def guardar_juegos(filas: list[dict]) -> None:
    ordenadas = sorted(filas, key=lambda fila: (fila.get("inicio_utc") or "", int(fila["id"])))
    marco = pd.DataFrame(ordenadas)
    for columna in COLUMNAS:
        if columna not in marco.columns:
            marco[columna] = None
    marco = marco[COLUMNAS]
    destino = directorio_datos() / "games.csv"
    temporal = destino.with_suffix(".csv.tmp")
    marco.to_csv(temporal, index=False)
    temporal.replace(destino)


def cargar_juegos() -> pd.DataFrame:
    ruta = directorio_datos() / "games.csv"
    if not ruta.exists():
        return pd.DataFrame(columns=COLUMNAS)
    marco = pd.read_csv(ruta)
    for columna in ("id", "temporada", "tipo", "neutral"):
        if columna in marco.columns:
            marco[columna] = pd.to_numeric(marco[columna], errors="coerce").fillna(0).astype(int)
    for columna in ("goles_visita", "goles_local", "gano_local", "cuota_local", "cuota_visita", "prob_mercado_local"):
        if columna in marco.columns:
            marco[columna] = pd.to_numeric(marco[columna], errors="coerce")
    for columna in ("periodo", "casa", "linea", "fecha", "inicio_utc", "estado", "visita", "local", "espn_id"):
        if columna in marco.columns:
            marco[columna] = marco[columna].map(lambda valor: "" if pd.isna(valor) else str(valor))
            marco[columna] = marco[columna].replace({"nan": "", "None": ""})
    return marco


def _cargar_existentes() -> dict[int, dict]:
    marco = cargar_juegos()
    juegos: dict[int, dict] = {}
    for fila in marco.to_dict(orient="records"):
        if not fila.get("id"):
            continue
        limpio = {}
        for clave in COLUMNAS:
            valor = fila.get(clave)
            if clave in {"cuota_local", "cuota_visita", "prob_mercado_local", "goles_local", "goles_visita", "gano_local"}:
                if valor is None or (isinstance(valor, float) and pd.isna(valor)):
                    valor = None
                elif clave != "prob_mercado_local":
                    valor = int(valor) if clave != "gano_local" else int(valor)
            elif clave == "espn_id" and valor in ("", None):
                valor = None
            limpio[clave] = valor
        juegos[int(limpio["id"])] = limpio
    return juegos


def actualizar_datos(completo: bool = False) -> pd.DataFrame:
    """Descarga el calendario y completa cuotas. `completo` recorre varias temporadas."""
    hoy = fecha_de_hoy()
    if completo or not (directorio_datos() / "games.csv").exists():
        inicio = date(2021, 10, 1)
        fin = hoy + timedelta(days=8)
        paso = 4
    else:
        inicio = hoy - timedelta(days=21)
        fin = hoy + timedelta(days=8)
        paso = 2
    fechas = _fechas_cada(inicio, fin, paso)
    refrescar_desde = (hoy - timedelta(days=3)).isoformat()
    print(f"calendario {fechas[0]} → {fechas[-1]} ({len(fechas)} consultas)", flush=True)
    nuevos = descargar_calendario(fechas, refrescar_desde=refrescar_desde)
    juegos = _cargar_existentes()
    for juego in nuevos:
        juegos[juego["id"]] = mezclar_juego(juegos.get(juego["id"]), juego)
    completar_cuotas(juegos, hoy=hoy.isoformat())
    guardar_juegos(list(juegos.values()))
    resumenes = descargar_resumenes_porteros()
    referencia = construir_referencia(resumenes)
    from comun_nhl import guardar_json

    guardar_json(directorio_datos() / "porteros_referencia.json", referencia)
    finales = [j for j in juegos.values() if j.get("gano_local") is not None and int(j.get("tipo") or 0) in (2, 3)]
    con_cuota = [j for j in finales if j.get("cuota_local") not in (None, "")]
    print(f"games.csv: {len(juegos)} partidos; finales {len(finales)}; con cuota {len(con_cuota)}", flush=True)
    return cargar_juegos()


def main() -> None:
    parser = argparse.ArgumentParser(description="Descarga resultados NHL y moneyline de ESPN")
    parser.add_argument("--completo", action="store_true", help="Varias temporadas, no solo la ventana reciente")
    args = parser.parse_args()
    actualizar_datos(completo=args.completo)


if __name__ == "__main__":
    main()

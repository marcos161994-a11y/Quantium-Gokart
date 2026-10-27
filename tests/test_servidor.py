"""El panel responde la jornada vigente sin salir a la red."""

import hashlib
import json
from datetime import datetime, time, timezone

import pandas as pd
from fastapi.testclient import TestClient

from comun_nhl import ZONA, fecha_de_hoy, guardar_json, leer_json
from servidor_nhl import app


def test_health_y_panel_con_los_picks_de_la_jornada(tmp_path, monkeypatch):
    monkeypatch.setenv("QUANTIUM_DATOS", str(tmp_path))
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "cache"))
    fecha = fecha_de_hoy().isoformat()
    (tmp_path / "picks.json").write_text(
        json.dumps(
            {
                "fecha": fecha,
                "umbral_alta": 0.62,
                "por_fecha": {
                    fecha: [
                        {
                            "id": "99",
                            "fecha": fecha,
                            "ganador": "BOS",
                            "nombre_ganador": "Bruins de Boston",
                            "probabilidad": 0.64,
                            "probabilidad_local": 0.64,
                            "nivel": "alta",
                            "equipo_local": "BOS",
                            "equipo_visita": "MTL",
                            "nombre_local": "Bruins de Boston",
                            "nombre_visita": "Canadiens de Montreal",
                            "linea": "BOS -135",
                            "casa": "DraftKings",
                            "prob_mercado_local": 0.57,
                            "hora": "7:00 pm",
                        }
                    ]
                },
                "record": {"examen": {"temporada": 20252026, "alta": {"aciertos": 3, "total": 5, "pct": 0.6}}},
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "backtest.json").write_text(
        json.dumps({"n": 10, "log_loss_modelo": 0.68, "brier_modelo": 0.24, "umbral_alta": 0.62, "mercado": {"n": 8}}),
        encoding="utf-8",
    )
    cliente = TestClient(app)
    salud = cliente.get("/api/health")
    assert salud.status_code == 200
    assert salud.json()["ok"] is True
    panel = cliente.get("/api/panel")
    assert panel.status_code == 200
    cuerpo = panel.json()
    assert cuerpo["fecha"] == fecha
    assert cuerpo["zona"] == "America/Puerto_Rico"
    assert cuerpo["picks"][0]["ganador"] == "BOS"
    assert cuerpo["picks"][0]["linea"] == "BOS -135"
    assert cuerpo["backtest"]["log_loss_modelo"] == 0.68


def test_la_portada_destaca_los_picks_de_confianza_alta():
    html = TestClient(app).get("/").text
    assert 'id="picks-dia"' in html
    assert "Picks del día" in html
    assert "Hoy no hay picks de confianza alta." in html
    assert "Portero cambiado" in html
    assert "Récord desde" in html


def test_panel_publica_el_elo_de_los_partidos_jugados(tmp_path, monkeypatch):
    monkeypatch.setenv("QUANTIUM_DATOS", str(tmp_path))
    pd.DataFrame(
        [
            {
                "id": 1,
                "temporada": 20252026,
                "tipo": 2,
                "fecha": "2025-10-01",
                "inicio_utc": "2025-10-01T23:00:00Z",
                "visita": "MTL",
                "local": "BOS",
                "neutral": 0,
                "goles_visita": 1,
                "goles_local": 5,
                "periodo": "REG",
                "gano_local": 1,
            }
        ]
    ).to_csv(tmp_path / "games.csv", index=False)
    ranking = {fila["abbr"]: fila["elo"] for fila in TestClient(app).get("/api/panel").json()["ranking"]}
    assert ranking["BOS"] > 1500 > ranking["MTL"]
    assert ranking["BOS"] > ranking["MTL"]


def test_panel_vacio_no_revienta(tmp_path, monkeypatch):
    monkeypatch.setenv("QUANTIUM_DATOS", str(tmp_path))
    cliente = TestClient(app)
    respuesta = cliente.get("/api/panel")
    assert respuesta.status_code == 200
    assert respuesta.json()["picks"] == []


def test_reentreno_en_directorio_aislado_genera_pick(tmp_path, monkeypatch):
    monkeypatch.setenv("QUANTIUM_DATOS", str(tmp_path))
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "cache"))
    hoy = fecha_de_hoy()
    inicio = datetime.combine(hoy, time(19, 0), tzinfo=ZONA).astimezone(timezone.utc)
    filas = []
    equipos = ["BOS", "TOR", "MTL", "BUF", "OTT", "DET", "FLA", "TBL"]
    for temporada, anio in ((20222023, 2022), (20232024, 2023)):
        for i in range(90):
            local = equipos[i % 8]
            visita = equipos[(i + 3) % 8]
            gano = 1 if i % 3 else 0
            filas.append(
                {
                    "id": temporada * 1000 + i,
                    "espn_id": "",
                    "temporada": temporada,
                    "tipo": 2,
                    "fecha": f"{anio}-10-{(i % 27) + 1:02d}",
                    "inicio_utc": f"{anio}-10-{(i % 27) + 1:02d}T23:00:00Z",
                    "estado": "OFF",
                    "visita": visita,
                    "local": local,
                    "neutral": 0,
                    "goles_visita": 1 if gano else 4,
                    "goles_local": 4 if gano else 1,
                    "periodo": "REG",
                    "gano_local": gano,
                    "cuota_local": -130 if gano else 120,
                    "cuota_visita": 110 if gano else -140,
                    "casa": "DraftKings",
                    "linea": f"{local} -130" if gano else f"{visita} -140",
                    "prob_mercado_local": 0.56 if gano else 0.44,
                }
            )
    filas.append(
        {
            "id": 999001,
            "espn_id": "",
            "temporada": 20262027,
            "tipo": 2,
            "fecha": hoy.isoformat(),
            "inicio_utc": inicio.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "estado": "FUT",
            "visita": "MTL",
            "local": "BOS",
            "neutral": 0,
            "goles_visita": "",
            "goles_local": "",
            "periodo": "",
            "gano_local": "",
            "cuota_local": -140,
            "cuota_visita": 120,
            "casa": "DraftKings",
            "linea": "BOS -140",
            "prob_mercado_local": 0.58,
        }
    )
    pd.DataFrame(filas).to_csv(tmp_path / "games.csv", index=False)
    from reentrenar_nhl import reentrenar
    from picks_nhl import generar_picks

    entrada = reentrenar(min_entrenamiento=40)
    assert "promovido" in entrada
    assert (tmp_path / "modelo.joblib").exists()
    assert (tmp_path / "backtest.json").exists()
    documento = generar_picks(en_vivo=False)
    assert documento["picks"], "la jornada de hoy debería tener un pick"
    assert documento["picks"][0]["ganador"] in {"BOS", "MTL"}
    assert 0.5 <= documento["picks"][0]["probabilidad"] <= 1


def _sha(ruta) -> str:
    return hashlib.sha256(ruta.read_bytes()).hexdigest()


def test_el_record_empieza_hoy_y_archiva_los_picks_anteriores(tmp_path, monkeypatch):
    monkeypatch.setenv("QUANTIUM_DATOS", str(tmp_path))
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "cache"))
    desde = "2026-10-27"
    guardar_json(tmp_path / "record.json", {"desde": desde, "zona": "America/Puerto_Rico"})
    protegidos = {
        "modelo.joblib": b"modelo-intacto",
        "games.csv": (
            b"id,temporada,tipo,fecha,inicio_utc,visita,local,neutral,"
            b"goles_visita,goles_local,periodo,gano_local\n"
            b"1,20252026,2,2025-10-01,2025-10-01T23:00:00Z,MTL,BOS,0,1,5,REG,1\n"
        ),
        "backtest.json": json.dumps(
            {
                "n": 5779,
                "log_loss_modelo": 0.673,
                "umbral_alta": 0.68,
                "por_nivel_examen": {"alta": {"aciertos": 77, "total": 118, "pct": 0.6525}},
                "por_nivel_oos": {"alta": {"aciertos": 817, "total": 1187, "pct": 0.6883}},
            }
        ).encode(),
    }
    for nombre, contenido in protegidos.items():
        (tmp_path / nombre).write_bytes(contenido)
    huellas = {nombre: _sha(tmp_path / nombre) for nombre in protegidos}
    hoy = desde
    pendiente = {
        "id": "hoy-pendiente",
        "fecha": hoy,
        "inicio_utc": "2026-10-28T02:00:00Z",
        "estado": "programado",
        "nivel": "alta",
        "acierto": None,
        "ganador": "BOS",
        "probabilidad": 0.7,
        "equipo_local": "BOS",
        "equipo_visita": "MTL",
    }
    calificado = {
        "id": "hoy-cerrado",
        "fecha": hoy,
        "inicio_utc": "2026-10-27T23:00:00Z",
        "estado": "final",
        "nivel": "media",
        "acierto": 0,
        "ganador": "NYR",
        "probabilidad": 0.6,
        "equipo_local": "NYR",
        "equipo_visita": "TOR",
    }
    viejo = {
        "id": "ayer",
        "fecha": "2026-10-26",
        "inicio_utc": "2026-10-26T23:00:00Z",
        "estado": "final",
        "nivel": "alta",
        "acierto": 1,
        "ganador": "EDM",
        "probabilidad": 0.8,
    }
    guardar_json(
        tmp_path / "picks.json",
        {
            "fecha": hoy,
            "umbral_alta": 0.68,
            "vivo": [viejo],
            "por_fecha": {hoy: [pendiente, calificado]},
            "picks": [pendiente, calificado],
            "record": {
                "examen": {"alta": {"aciertos": 77, "total": 118, "pct": 0.6525}},
                "oos": {"alta": {"aciertos": 817, "total": 1187, "pct": 0.6883}},
                "vivo": {"alta": {"aciertos": 1, "total": 1, "pct": 1}},
            },
        },
    )
    from picks_nhl import archivar_picks

    documento = leer_json(tmp_path / "picks.json", {})
    documento, archivados = archivar_picks(documento, desde)
    guardar_json(tmp_path / "picks.json", documento)
    assert archivados == 1
    assert [pick["id"] for pick in documento["vivo"]] == []
    assert [pick["id"] for pick in documento["por_fecha"][hoy]] == ["hoy-pendiente", "hoy-cerrado"]
    assert documento["record"]["desde"] == desde
    assert documento["record"]["alta"]["total"] == 0
    assert documento["record"]["media"] == {"aciertos": 0, "total": 1, "pct": 0.0}
    assert "examen" not in documento["record"]

    otra_vez, repetidos = archivar_picks(leer_json(tmp_path / "picks.json", {}), desde)
    assert repetidos == 0
    archivo = leer_json(tmp_path / "picks_archivo.json", {})
    assert archivo["n"] == 1
    assert archivo["picks"][0]["id"] == "ayer"
    assert [pick["id"] for pick in otra_vez["por_fecha"][hoy]] == ["hoy-pendiente", "hoy-cerrado"]

    panel = TestClient(app).get("/api/panel").json()
    assert panel["record"]["desde"] == desde
    assert panel["record"]["alta"]["total"] == 0
    assert panel["record"]["media"]["total"] == 1
    assert panel["record"]["media"]["aciertos"] == 0
    assert "examen" not in panel["record"]
    assert {pick["id"] for pick in panel["picks"]} == {"hoy-pendiente", "hoy-cerrado"}
    assert panel["backtest"]["por_nivel_examen"]["alta"]["aciertos"] == 77
    assert panel["backtest"]["n"] == 5779
    assert {nombre: _sha(tmp_path / nombre) for nombre in protegidos} == huellas

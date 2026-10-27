"""Reglas del modelo: Elo, cuotas sin vig, jornada de Puerto Rico, encogimiento."""

from datetime import date, datetime

import numpy as np
import pandas as pd

from comun_nhl import ZONA, a_nhl, fecha_de_cartelera, fecha_de_hoy
from datos_nhl import americana, elegir_proveedor, probabilidad_implicita, sin_vig, texto_linea
from elo_nhl import EloNHL
from modelo_nhl import _g0_justificado, aplicar_encogimiento, construir_filas, elegir_umbral, nivel_de
from porteros_nhl import ajustar_probabilidad
from reentrenar_nhl import conviene_promover


def _delta(periodo, goles_local, goles_visita):
    elo = EloNHL()
    return elo.actualizar("BOS", "MTL", goles_local, goles_visita, periodo, neutral=True)


def test_shootout_y_prorroga_mueven_menos_que_el_reglamento():
    paliza = _delta("REG", 6, 1)
    corto = _delta("REG", 2, 1)
    prorroga = _delta("OT", 3, 2)
    shootout = _delta("SO", 3, 2)
    assert paliza > corto > prorroga > shootout > 0


def test_local_tiene_ventaja_y_la_franquicia_de_arizona_sigue_en_utah():
    elo = EloNHL()
    assert elo.expectativa_local("BOS", "MTL", neutral=False) > 0.5
    assert abs(elo.expectativa_local("BOS", "MTL", neutral=True) - 0.5) < 1e-9
    elo.actualizar("ARI", "BOS", 4, 1, "REG", neutral=True)
    assert elo.obtener("UTA") > 1500
    assert elo.obtener("ARI") == elo.obtener("UTA")


def test_regresion_al_cambiar_de_temporada():
    elo = EloNHL()
    elo.rating["BOS"] = 1700
    elo.temporada = 20222023
    elo.abrir_temporada(20232024)
    assert abs(elo.obtener("BOS") - (1500 * 0.3 + 1700 * 0.7)) < 1e-9
    elo.abrir_temporada(20232024)
    assert abs(elo.obtener("BOS") - (1500 * 0.3 + 1700 * 0.7)) < 1e-9


def test_madrugada_en_puerto_rico_pertenece_a_la_jornada_anterior():
    assert fecha_de_cartelera("2026-10-28T05:30:00Z") == date(2026, 10, 27)
    assert fecha_de_cartelera("2026-10-27T23:00:00Z") == date(2026, 10, 27)
    assert fecha_de_cartelera("2026-10-28T15:00:00Z") == date(2026, 10, 28)
    assert fecha_de_hoy(datetime(2026, 10, 28, 4, 30, tzinfo=ZONA)) == date(2026, 10, 27)
    assert fecha_de_hoy(datetime(2026, 10, 28, 7, 0, tzinfo=ZONA)) == date(2026, 10, 28)


def test_abreviaturas_de_espn():
    assert a_nhl("LA") == "LAK"
    assert a_nhl("TB") == "TBL"
    assert a_nhl("SJ") == "SJS"
    assert a_nhl("NJ") == "NJD"
    assert a_nhl("UTAH") == "UTA"


def test_cuota_americana_y_mercado_sin_vig():
    assert americana("-185") == -185
    assert americana("+124") == 124
    assert americana({"american": "-130"}) == -130
    assert americana("even") == 100
    assert americana(50) is None
    assert abs(sin_vig(probabilidad_implicita(-110), probabilidad_implicita(-110)) - 0.5) < 1e-9
    local = probabilidad_implicita(-150)
    visita = probabilidad_implicita(130)
    limpia = sin_vig(local, visita)
    assert limpia < local
    assert 0.5 < limpia < local
    assert texto_linea("PHI", "CBJ", 100, -120) == "CBJ -120"


def test_prefiere_draftkings_y_descarta_la_linea_en_vivo():
    items = [
        {
            "provider": {"name": "ESPN Bet - Live Odds"},
            "homeTeamOdds": {"moneyLine": -3000},
            "awayTeamOdds": {"moneyLine": 800},
        },
        {
            "provider": {"name": "ESPN BET"},
            "homeTeamOdds": {"close": {"moneyLine": {"american": "+155"}}},
            "awayTeamOdds": {"close": {"moneyLine": {"american": "-185"}}},
        },
        {
            "provider": {"name": "DraftKings"},
            "moneyline": {
                "home": {"close": {"odds": "+140"}},
                "away": {"close": {"odds": "-160"}},
            },
        },
    ]
    nombre, local, visita = elegir_proveedor(items)
    assert nombre == "DraftKings"
    assert local == 140
    assert visita == -160


def test_encogimiento_hacia_el_empate_y_umbral():
    assert aplicar_encogimiento(0.8, 0, 20) == 0.5
    assert abs(aplicar_encogimiento(0.8, 20, 20) - 0.8) < 1e-9
    assert abs(aplicar_encogimiento(0.8, 10, 20) - 0.65) < 1e-9
    assert aplicar_encogimiento(0.8, 0, None) == 0.8
    assert nivel_de(0.64, 0.62) == "alta"
    assert nivel_de(0.57, 0.62) == "media"
    assert nivel_de(0.52, 0.62) == "baja"
    rng = np.random.default_rng(0)
    prob = rng.uniform(0.5, 0.8, size=200)
    acierto = (rng.random(200) < (prob - 0.45)).astype(float)
    assert elegir_umbral(prob, acierto, minimo=30) >= 0.58


def test_el_elo_de_un_partido_no_usa_su_propio_resultado():
    juegos = pd.DataFrame(
        [
            {
                "id": 1,
                "temporada": 20222023,
                "tipo": 2,
                "fecha": "2022-10-15",
                "inicio_utc": "2022-10-15T23:00:00Z",
                "local": "BOS",
                "visita": "MTL",
                "neutral": 0,
                "goles_local": 6,
                "goles_visita": 1,
                "periodo": "REG",
                "gano_local": 1,
            },
            {
                "id": 2,
                "temporada": 20222023,
                "tipo": 2,
                "fecha": "2022-10-17",
                "inicio_utc": "2022-10-17T23:00:00Z",
                "local": "BOS",
                "visita": "TOR",
                "neutral": 0,
                "goles_local": 2,
                "goles_visita": 3,
                "periodo": "OT",
                "gano_local": 0,
            },
        ]
    )
    filas = construir_filas(juegos, {})
    assert abs(filas.iloc[0]["diff_elo"]) < 1e-6
    assert filas.iloc[1]["diff_elo"] > 0
    assert int(filas.iloc[1]["b2b_local"]) == 0
    assert filas.iloc[1]["descanso_local"] == 2


def test_portero_peor_que_la_referencia_baja_la_probabilidad():
    igual = ajustar_probabilidad(0.61, None, None, None, 0.91, None, None, None, 0.90)
    assert igual == 0.61
    peor = ajustar_probabilidad(
        0.61,
        0.880,
        900,
        "confirmed",
        0.915,
        0.905,
        900,
        "confirmed",
        0.905,
    )
    assert peor < 0.61


def test_el_encogimiento_solo_sigue_si_mejora_temporadas_cerradas():
    g0, delta = _g0_justificado(
        [
            {
                "n_examen": 1400,
                "g0": 40,
                "log_loss_encogido": 0.671,
                "log_loss_sin_encoger": 0.667,
            }
        ]
    )
    assert g0 is None
    assert delta > 0
    g0, delta = _g0_justificado(
        [
            {
                "n_examen": 1400,
                "g0": 16,
                "log_loss_encogido": 0.650,
                "log_loss_sin_encoger": 0.670,
            }
        ]
    )
    assert g0 == 16
    assert delta < 0


def test_solo_se_promueve_si_mejora_el_log_loss():
    assert conviene_promover(0.66, 0.68) is True
    assert conviene_promover(0.70, 0.68) is False
    assert conviene_promover(0.66, None) is True
    assert conviene_promover(float("nan"), 0.6) is False

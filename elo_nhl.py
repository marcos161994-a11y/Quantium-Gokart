"""Elo de la NHL.

Una victoria en tiempo reglamentario mueve la clasificación de verdad.
La prórroga y, más todavía, el shootout se anotan cerca del empate: el
marcador observado es 0.60 / 0.40 en OT y 0.55 / 0.45 en shootout, sin
multiplicador por diferencia de goles. Así un desempate no se parece a
una paliza de cinco goles.
"""

from __future__ import annotations

import math

from comun_nhl import franquicia

BASE = 1500.0
K = 10.0
VENTAJA_LOCAL = 40.0
REGRESION = 0.30


class EloNHL:
    def __init__(
        self,
        k: float = K,
        ventaja_local: float = VENTAJA_LOCAL,
        base: float = BASE,
        regresion: float = REGRESION,
    ):
        self.k = k
        self.ventaja_local = ventaja_local
        self.base = base
        self.regresion = regresion
        self.rating: dict[str, float] = {}
        self.temporada: int | None = None

    def obtener(self, equipo: str) -> float:
        return self.rating.get(franquicia(equipo), self.base)

    def expectativa_local(self, local: str, visita: str, neutral: bool = False) -> float:
        ventaja = 0.0 if neutral else self.ventaja_local
        diff = self.obtener(local) + ventaja - self.obtener(visita)
        return 1.0 / (1.0 + 10 ** (-diff / 400.0))

    def abrir_temporada(self, temporada: int | None) -> None:
        """Al cambiar de temporada encoge cada Elo hacia 1500.

        Es el mismo espíritu que encoger la probabilidad al inicio: en
        octubre no creemos del todo el cierre de junio.
        """
        if temporada is None:
            return
        if self.temporada is None:
            self.temporada = int(temporada)
            return
        if int(temporada) == self.temporada:
            return
        self.temporada = int(temporada)
        for equipo, valor in list(self.rating.items()):
            self.rating[equipo] = self.base * self.regresion + valor * (1.0 - self.regresion)

    @staticmethod
    def observado(gano_local: bool, periodo: str) -> float:
        periodo = (periodo or "REG").upper()
        if periodo == "REG":
            return 1.0 if gano_local else 0.0
        if periodo == "OT":
            return 0.60 if gano_local else 0.40
        # SO y cualquier otro desempate: todavía más cerca de 50/50.
        return 0.55 if gano_local else 0.45

    def actualizar(
        self,
        local: str,
        visita: str,
        goles_local: int,
        goles_visita: int,
        periodo: str,
        neutral: bool = False,
        pretemporada: bool = False,
    ) -> float:
        expectativa = self.expectativa_local(local, visita, neutral)
        real = self.observado(goles_local > goles_visita, periodo)
        mov = 1.0
        if (periodo or "REG").upper() == "REG":
            diff_goles = abs(int(goles_local) - int(goles_visita))
            ventaja = 0.0 if neutral else self.ventaja_local
            elo_diff = abs(self.obtener(local) + ventaja - self.obtener(visita))
            mov = math.log(diff_goles + 1) * (2.2 / (elo_diff * 0.001 + 2.2))
        k = self.k * (0.5 if pretemporada else 1.0) * mov
        delta = k * (real - expectativa)
        self.rating[franquicia(local)] = self.obtener(local) + delta
        self.rating[franquicia(visita)] = self.obtener(visita) - delta
        return delta

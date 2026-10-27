# Quantium NHL

Modelo de predicción de la NHL. Cada pick es un ganador y una probabilidad de llevarse el partido, prórroga y shootout incluidos. No hay spread ni total.

La probabilidad sale de un Elo por franquicia más una regresión logística. Se compara siempre con la probabilidad implícita del moneyline **sin vigorish**. La fuente de la línea es DraftKings, vía la API pública de ESPN; si el cierre de DraftKings no quedó archivado, se usa el cierre de ESPN BET y se guarda el nombre de la casa.

## Cómo está armado

| Pieza | Qué hace |
| --- | --- |
| `datos_nhl.py` | Calendario y resultados (`api-web.nhle.com`) y moneyline (ESPN). Histórico en `data/games.csv`. |
| `elo_nhl.py` | Elo por franquicia. Arizona y Utah son el mismo club. |
| `porteros_nhl.py` | Titular de referencia (temporada anterior) y ajuste si el probable de hoy no es ese titular. |
| `modelo_nhl.py` | Features, logística, encogimiento y backtest walk-forward. |
| `entrenar_modelo.py` | Backtest, primer modelo y picks. |
| `reentrenar_nhl.py` | Reentreno. Solo promueve `data/modelo.joblib` si baja el log loss de examen. |
| `picks_nhl.py` | Picks del día en `data/picks.json`, con récord por confianza. |
| `diario_nhl.py` | Ciclo diario: refrescar, reentrenar, picks. `--commit` deja `data/` en git. |
| `servidor_nhl.py` | FastAPI: `/api/health`, `/api/panel` y `QuantiumNHL.html`. |

La caché de las APIs va a `data/cache`, o a la ruta de `DATA_DIR`. No se usa `/var/data`. El modelo, los picks y el histórico sí viajan en git.

## Elo, prórroga y shootout

Partimos de 1500. Ventaja de local: 40 puntos. K = 10.

- Tiempo reglamentario: el resultado observado es 1 o 0, con un multiplicador por diferencia de goles.
- Prórroga: 0.60 para el ganador y 0.40 para el perdedor, sin multiplicador.
- Shootout: 0.55 / 0.45.

Al abrir una temporada nueva, cada Elo se encoge un 30% hacia 1500.

## Features

La etiqueta es «ganó el local», contando prórroga y shootout.

- Diferencia de Elo (sin meter ahí la ventaja de local).
- `local`: 0 si la pista es neutral.
- Back-to-back y días de descanso de cada equipo, con tope de 6.
- Diferencia de sv% y de GSAx del titular de la temporada anterior. El GSAx es `paradas − sv_liga × tiros`: la NHL no publica GSAx.

Esas features se calculan antes del partido. La pretemporada no mueve el Elo ni entra al ajuste.

## Encogimiento

En cada pliegue, la validación interna prueba acercar la probabilidad a 50% durante los primeros partidos (horizonte `g0` de 8, 12, 16, 24 o 40). Eso solo se adopta si, en temporadas ya cerradas, el log loss de examen bajó. En la corrida actual no bajó: donde la validación pidió `g0 = 40` (2024-25), el log loss pasó de 0.667 a 0.671. El modelo que se sirve no encoge. El Elo sí se acerca un 30% a 1500 al cambiar de temporada, y eso no se apaga.

## Picks y confianza

Solo se publica el ganador y su probabilidad. Alta, media y baja salen de un umbral elegido en las temporadas anteriores a la de examen (candidatos del 58% al 70%; si no hay muestra, 62%). Media es desde 55% hasta ese umbral. Baja es el resto.

El récord que enseña el panel es la temporada de examen, fuera de muestra. El umbral no se eligió mirando esa temporada.

## Día de Puerto Rico

La jornada no cambia a medianoche. Antes de las 06:00 en `America/Puerto_Rico`, un partido que empieza de madrugada sigue perteneciendo al día anterior, y «hoy» sigue siendo esa jornada.

## Correrlo en local

Hace falta Python 3.12.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Histórico (varias temporadas) y cuotas. Tarda: son miles de cierres.
python datos_nhl.py --completo

# Backtest walk-forward, modelo y picks de la jornada.
python entrenar_modelo.py

# Solo el backtest y el modelo, sin regenerar picks:
python reentrenar_nhl.py

# Panel
bash start.sh
# o, en un puerto local:
PORT=8741 bash start.sh
```

- Salud: `http://127.0.0.1:10000/api/health`
- Panel: `http://127.0.0.1:10000/api/panel`
- Página: `http://127.0.0.1:10000/`

Los números del último backtest quedan en `data/backtest.json`. Pruebas:

```bash
pytest
```

El ciclo de un día normal (ventana reciente, no el histórico entero):

```bash
python diario_nhl.py
```

## Backtest

Walk-forward por temporada. Para predecir la temporada T se entrena solo con lo anterior. El encogimiento de cada pliegue se elige en el último tramo de ese entrenamiento, no en el año que se está midiendo.

En los partidos que tienen cuota, el log loss y el Brier del modelo se calculan sobre **el mismo subconjunto** que el mercado sin vig. No se compara el modelo en todos los partidos contra el mercado en unos pocos.

El archivo `data/modelo.joblib` es un ajuste con todo el histórico, y solo reemplaza al anterior si gana en los últimos partidos de examen. El log loss que vale para juzgar el sistema es el del walk-forward, no el de ese ajuste final.

`data/reentrenos.json` anota cada intento: si se promovió, el log loss nuevo y el anterior.

## Tarea diaria

`.github/workflows/diario.yml` corre a las 11:15 UTC (07:15 en Puerto Rico), califica, reentrena y hace commit de `data/`. También se puede lanzar a mano. Si existe el secreto `SENTRY_DSN`, el ciclo lo usa; si no, sigue igual.

## Sentry

Opcional. `iniciar_sentry()` no hace nada si `SENTRY_DSN` está vacío. En Render el entorno queda como `production` vía `SENTRY_ENVIRONMENT`. El DSN no va en `render.yaml`: se agrega en el panel del servicio.

## Desplegar en Render

`render.yaml` describe un web service de Python, plan starter, región oregon, health check en `/api/health`, Python 3.12.14 y `DATA_DIR` dentro del proyecto.

Para dejarlo en marcha todavía falta:

1. Un repositorio de GitHub con este código (el blueprint no puede apuntar a un repo que no existe).
2. En Render: New → Blueprint → este `render.yaml`, o un web service con los mismos comandos (`pip install -r requirements.txt`, `bash start.sh`).
3. Rama de despliegue que reciba el push de `.github/workflows/diario.yml`, con auto-deploy, para que el commit diario de `data/` llegue al servicio.
4. `SENTRY_DSN` solo si se quiere telemetría. Sin eso el servicio arranca igual.
5. El disco de Render no hace falta: modelo, picks e histórico vienen en el deploy. `DATA_DIR` es solo caché y se puede perder.

## Números del último backtest

Walk-forward de 2022-23 a 2026-27 (la 2021-22 solo entrena). 5.779 partidos fuera de muestra. Se reproduce con `python entrenar_modelo.py` cuando `data/games.csv` ya está bajado; el detalle queda en `data/backtest.json`.

| | Modelo | Mercado sin vig |
| --- | --- | --- |
| Log loss (5.746 con cuota) | 0.674 | 0.663 |
| Brier (mismos partidos) | 0.240 | 0.235 |
| Log loss, solo DraftKings (3.904) | 0.674 | 0.663 |
| Brier, solo DraftKings | 0.240 | 0.235 |

En todos los partidos fuera de muestra el log loss del modelo es 0.673 y el Brier 0.240. Acierta el ganador el 58.1% de las veces. El cierre del mercado es mejor: este modelo no lo supera. Sí está por debajo del 0.693 de tirar una moneda.

Por temporada, log loss del modelo (sin encoger):

| Temporada | Partidos | Log loss | Brier |
| --- | --- | --- | --- |
| 2022-23 | 1.400 | 0.674 | 0.240 |
| 2023-24 | 1.400 | 0.663 | 0.235 |
| 2024-25 | 1.398 | 0.667 | 0.237 |
| 2025-26 | 1.394 | 0.691 | 0.249 |
| 2026-27 (parcial) | 187 | 0.665 | 0.236 |

El umbral de confianza alta quedó en **68%**. Se eligió con las temporadas anteriores a 2025-26. Estos son los aciertos en 2025-26, que no participó en esa elección:

| Nivel | Aciertos | Tasa |
| --- | --- | --- |
| Alta (≥ 68%) | 77 / 118 | 65.3% |
| Media (55% a 68%) | 428 / 785 | 54.5% |
| Baja (< 55%) | 232 / 491 | 47.3% |

En el conjunto fuera de muestra entero (aquí el umbral ya vio parte de los datos) la alta llega a 68.8% (817/1.187), la media a 57.3% y la baja a 51.5%. La banda que sirve es la alta. La baja, en el año de examen, quedó por debajo de una moneda.

El reentreno promueve el `joblib` solo si mejora el log loss de los últimos partidos de examen. La promoción vigente bajó ese número de 0.675 a 0.665.

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

Si la validación interna de un pliegue mejora el log loss acercando las probabilidades a 50% durante los primeros partidos, se guarda ese horizonte (`g0`). Si no mejora, no se encoge. En pretemporada el peso es cero, así que un `g0` activo deja el partido en 50%.

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

Se rellenan al entrenar. Mira `data/backtest.json` si esta sección todavía no tiene la corrida de este clon.

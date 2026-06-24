# Monte Carlo GBM Quant Report

Motor de simulacion Monte Carlo para acciones. El objetivo no es adivinar un precio exacto, sino construir un mapa de escenarios: centro, rango probable, riesgo de cola y estabilidad del modelo.

El proyecto tiene dos formas de uso:

- `monte_carlo_simulatio_proyect.py`: motor principal. Descarga data, simula escenarios y genera un informe PDF.
- `MONTE_CARLO_SIMULATIO_PROYECT.ipynb`: notebook de uso guiado para revisar el flujo y ejecutar el motor paso a paso.

## Que Produce

El script genera:

```text
outputs/monte_carlo_quant_report.pdf
```

El PDF contiene:

- Portada con lectura ejecutiva.
- Tabla consolidada por accion.
- Ranking de cambio mediano contra precio actual.
- Ranking de riesgo de cola usando `CVaR 5%`.
- Pagina por accion con trayectorias simuladas y distribucion final.

## Enfoque Quant

El flujo sigue esta regla:

```text
data -> regla -> prueba -> riesgo -> comparacion
```

- `data`: precios historicos descargados desde `yfinance`.
- `regla`: retornos logaritmicos, media dinamica y volatilidad dinamica.
- `prueba`: simulaciones Monte Carlo sobre un horizonte definido.
- `riesgo`: rango 95%, `VaR 5%`, `CVaR 5%` y error de calibracion.
- `comparacion`: todas las acciones pasan por la misma regla.

## Modelo Actual

El motor ya no usa una media y volatilidad fijas calculadas sobre toda la historia.

Ahora usa:

- Media dinamica con filtro de Kalman.
- Volatilidad EWMA para dar mas peso a la historia reciente.
- Volatilidad implicita opcional desde opciones, cuando `yfinance` la entrega.
- Choques `t-Student` para capturar colas mas pesadas que una normal.
- Clustering de volatilidad tipo GARCH simple: shocks grandes elevan la volatilidad futura simulada.

## Variables Principales

| Variable | Que controla | Uso practico |
| --- | --- | --- |
| `START_DATE` | Fecha inicial de descarga | Define cuanta historia entra al modelo |
| `TICKERS` | Lista de acciones | Define el universo comparado |
| `SINGLE_TICKER` | Accion individual | Sirve para pruebas de una sola accion |
| `NUM_SIMULATIONS` | Cantidad de caminos simulados | Mas simulaciones reducen ruido estadistico |
| `NUM_DAYS_SINGLE_TICKER` | Horizonte individual | Dias simulados para una accion |
| `NUM_DAYS_MULTIPLE_TICKERS` | Horizonte comparativo | Dias simulados para varias acciones |
| `ROLLING_WINDOW` | Ventana reciente | Base para varianza de largo plazo |
| `EWMA_LAMBDA` | Peso de volatilidad reciente | Controla sensibilidad a cambios de regimen |
| `STUDENT_T_DF` | Grados de libertad t-Student | Menor valor implica colas mas pesadas |
| `VAR_LEVEL` | Nivel de cola | Define el percentil usado para VaR y CVaR |
| `REPORT_PATH` | Ruta del PDF | Define donde se guarda el informe |

## Metricas Del Reporte

| Metrica | Lectura directa |
| --- | --- |
| `Precio actual` | Ultimo cierre disponible |
| `Mediana simulada` | Centro de los precios finales simulados |
| `Piso 95%` | Parte baja del rango central |
| `Techo 95%` | Parte alta del rango central |
| `Cambio % vs actual` | Diferencia entre mediana simulada y precio actual |
| `VaR 5%` | Perdida minima al entrar en el peor 5% de escenarios |
| `CVaR 5%` | Perdida promedio dentro del peor 5% |
| `Sigma usada diaria` | Volatilidad diaria usada por el motor |
| `Error calibracion` | Alerta de inestabilidad reciente en media y volatilidad |

## Como Ejecutar

Instala dependencias:

```bash
python -m venv .venv
source .venv/bin/activate
pip install pandas numpy matplotlib scipy seaborn yfinance jupyter
```

Ejecuta el motor:

```bash
python monte_carlo_simulatio_proyect.py
```

El progreso se muestra con logger:

```text
01:37:02 | INFO | AAPL | Simulation progress 50%
01:37:03 | INFO | PDF report complete | path=outputs/monte_carlo_quant_report.pdf
```

## Como Leer El Resultado

Una accion con mediana positiva pero `CVaR 5%` muy negativo puede tener upside, pero tambien cola de perdida fuerte.

Una accion con `Error calibracion` alto esta en una zona menos estable: la media o la volatilidad reciente estan cambiando rapido.

El informe sirve para comparar escenarios bajo incertidumbre. No reemplaza decision de inversion, control de riesgo ni validacion fuera de muestra.

## Referencias

- QUANT GUILD / Roman: [How to Quant Trade in 3 Minutes](https://youtu.be/mZLNzqDZHbA)
- QUANT GUILD / Roman: [Time Series Analysis for Quant Finance](https://youtu.be/JwqjuUnR8OY)
- Video base del proyecto: <https://youtu.be/fO-lGzZADVU>

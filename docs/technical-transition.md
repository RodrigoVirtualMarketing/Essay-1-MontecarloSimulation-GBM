# Transicion Tecnica Del Motor Monte Carlo

Este documento explica como evoluciono el codigo desde la primera version del proyecto hasta la version actual. La idea central no cambio: estimar escenarios, no predecir un precio unico. Lo que cambio fue la calidad del motor, la forma de medir riesgo y la forma de entregar resultados.

## 1. Primera Version: Script Lineal GBM

La primera version funcionaba como un script tipo notebook:

1. Descargaba precios con `yfinance`.
2. Calculaba retornos diarios simples.
3. Estimaba una media `mu` y una volatilidad `sigma` usando toda la historia disponible.
4. Corria `NUM_SIMULATIONS` trayectorias durante `NUM_DAYS`.
5. Calculaba precio final mediano y trayectoria cercana a esa mediana.
6. Graficaba trayectorias y distribuciones con `matplotlib` y `seaborn`.

El nucleo del modelo era:

```python
price = price_list[-1] * np.exp((mu - 0.5 * sigma**2) + sigma * np.random.normal())
```

Ese enfoque era correcto como primera aproximacion a GBM. El problema era que asumia un mercado demasiado estable:

- `mu` era fijo.
- `sigma` era fijo.
- toda la historia tenia el mismo peso.
- los shocks eran normales.
- no habia medicion explicita de riesgo de cola.
- la salida dependia de graficos sueltos y `print`.

## 2. Problema Principal: Historia Estatica

La version inicial trataba el mercado como si la distribucion no cambiara. Eso es debil para activos financieros, porque una accion puede pasar de un regimen tranquilo a uno muy volatil en pocos dias.

Antes:

```python
mu = returns.mean()
sigma = returns.std()
```

Ahora:

```python
mu_series = kalman_filter_mean(log_returns)
sigma_series = ewma_volatility(log_returns)
```

La transicion tecnica fue pasar de parametros historicos globales a parametros dinamicos:

- `kalman_filter_mean`: actualiza la media estimada segun nueva informacion.
- `ewma_volatility`: da mas peso a movimientos recientes.
- `calibration_error`: mide si `mu` y `sigma` estan cambiando demasiado rapido.

Esto mantiene la filosofia original: no predice, estima con mejor sensibilidad al regimen actual.

## 3. Retornos Simples A Retornos Logaritmicos

La primera version usaba retornos porcentuales simples:

```python
returns = close_prices.pct_change()
```

La version actual usa retornos logaritmicos:

```python
return np.log(close_prices / close_prices.shift(1)).dropna()
```

Motivo:

- Encajan mejor con GBM.
- Son aditivos en el tiempo.
- Reducen ambiguedad tecnica al modelar crecimiento compuesto.

## 4. Volatilidad Historica A Volatilidad Mixta

La version inicial miraba volatilidad solo por desviacion estandar historica.

La version actual usa:

- `dynamic_sigma`: volatilidad EWMA.
- `implied_sigma`: volatilidad implicita opcional desde opciones.
- `sigma_used`: mezcla entre volatilidad reciente e implicita cuando esta disponible.

La funcion relevante es:

```python
get_atm_implied_volatility(ticker, current_price, target_days)
```

Esto permite que el modelo mire algo mas cercano a la expectativa actual del mercado, no solo al comportamiento pasado.

Limitacion: `yfinance` no siempre entrega opciones o volatilidad implicita confiable. Por eso el motor cae automaticamente a EWMA si la data de opciones no esta disponible.

## 5. Choques Normales A Colas Pesadas

La primera version usaba:

```python
np.random.normal()
```

Eso asume shocks normales. En mercados reales, los retornos extremos aparecen mas de lo que una normal suele anticipar.

La version actual usa `t-Student`:

```python
def unit_variance_student_t(size, df=STUDENT_T_DF):
    shocks = t.rvs(df=df, size=size)
    return shocks / np.sqrt(df / (df - 2))
```

Motivo:

- Permite colas mas pesadas.
- Aumenta la probabilidad de escenarios extremos.
- Evita que el rango probable subestime demasiado el riesgo.

## 6. Volatilidad Constante A Volatility Clustering

En la primera version, cada dia simulado tenia la misma volatilidad.

Ahora, la volatilidad cambia dentro de cada trayectoria:

```python
variance = omega + alpha * (sigma_t * shock) ** 2 + beta * variance
```

Lectura directa:

- Si aparece un shock grande, la varianza futura sube.
- Si el activo entra en una fase inestable, la simulacion deja que esa inestabilidad persista.
- Esto aproxima el concepto de volatility clustering sin introducir una dependencia pesada como `arch`.

## 7. Precio Central A Riesgo De Cola

La primera version se enfocaba en:

- mediana final.
- trayectoria cercana a la mediana.
- intervalo 95%.

La version actual conserva eso, pero agrega:

```python
var_5_return = np.percentile(final_returns, VAR_LEVEL * 100)
cvar_5_return = final_returns[final_returns <= var_5_return].mean()
```

Interpretacion:

- `VaR 5%`: umbral donde empiezan los peores escenarios.
- `CVaR 5%`: perdida promedio dentro del peor 5%.

Esto cambia la salida de "cuanto podria valer" a "cuanto podria doler si sale mal".

## 8. De Prints Y Graficos A Informe PDF

La primera version mostraba resultados con `print` y graficos interactivos.

La version actual genera un reporte:

```python
outputs/monte_carlo_quant_report.pdf
```

El PDF incluye:

- portada ejecutiva.
- marco conceptual.
- tabla consolidada.
- grafico de cambio mediano vs precio actual.
- grafico de `CVaR 5%`.
- pagina individual por accion.

El motor usa:

```python
from matplotlib.backends.backend_pdf import PdfPages
```

Esto convierte la simulacion en un producto revisable, compartible y repetible.

## 9. De Script A Motor Reutilizable

La version inicial mezclaba todo en una secuencia:

- descarga.
- calculo.
- simulacion.
- grafico.
- salida.

La version actual separa responsabilidades:

| Area | Funcion |
| --- | --- |
| Data | `get_stock_data`, `get_close_series` |
| Retornos | `get_log_returns` |
| Parametros dinamicos | `kalman_filter_mean`, `ewma_volatility` |
| Volatilidad forward-looking | `get_atm_implied_volatility` |
| Simulacion | `simulate_dynamic_gbm` |
| Riesgo | `VaR`, `CVaR`, `calibration_error` dentro del resultado |
| Consolidacion | `summarize_results` |
| Reporte | `create_pdf_report` y paginas auxiliares |
| Ejecucion | `run_single_ticker`, `run_multiple_tickers` |

Tambien se agrego `SimulationResult` como estructura unica de salida. Eso evita pasar muchos arreglos sueltos entre funciones.

## 10. Logger

Antes, el usuario solo veia prints finales.

Ahora el programa reporta progreso:

```text
01:37:02 | INFO | AAPL | Simulation progress 50%
01:37:03 | INFO | PDF report complete | path=outputs/monte_carlo_quant_report.pdf
```

El logger ayuda a saber:

- cuando descarga data.
- cuando busca volatilidad implicita.
- que parametros esta usando.
- cuanto avanza cada simulacion.
- cuando se construye cada pagina del PDF.

## 11. Notebook Como Interfaz

Antes el notebook duplicaba casi toda la logica.

Ahora el notebook importa el motor:

```python
from monte_carlo_simulatio_proyect import (...)
```

Su rol cambio:

- ya no es la fuente principal del modelo.
- sirve como interfaz guiada.
- muestra configuracion.
- corre una accion.
- corre varias acciones.
- genera el PDF.
- presenta una tabla de riesgo.

Esto reduce inconsistencias entre notebook y script.

## 12. Filosofia Preservada

La filosofia inicial sigue siendo la misma:

- no se predice un precio exacto.
- se estima una distribucion.
- se mide incertidumbre.
- se comparan escenarios.
- se toman decisiones bajo riesgo, no bajo certeza.

La diferencia es que ahora el codigo esta mas alineado con esa filosofia. Si la tesis dice que el mercado no es estatico, el modelo ya no debe usar parametros estaticos. Si la tesis dice que importa la incertidumbre, la salida debe mostrar cola, rango y estabilidad, no solo una mediana.

## 13. Limitaciones Pendientes

El motor todavia tiene puntos mejorables:

- Las trayectorias del PDF pueden ser pesadas si se grafican las `10,000` simulaciones completas.
- La volatilidad implicita depende de disponibilidad y calidad de `yfinance`.
- No hay backtesting fuera de muestra.
- No hay pruebas unitarias.
- El modelo de clustering es una aproximacion tipo GARCH, no una estimacion GARCH calibrada formalmente.
- No existe todavia un calculo explicito de edge contra precio de mercado de opciones o contra una regla de trading.

## 14. Resumen Ejecutivo

La transicion fue:

```text
script GBM estatico
-> motor modular
-> parametros dinamicos
-> colas pesadas
-> volatility clustering
-> VaR / CVaR
-> error de calibracion
-> logger
-> informe PDF
```

El proyecto paso de ser una demostracion Monte Carlo a una herramienta basica de investigacion quant para principiantes: no promete certeza, pero organiza la incertidumbre en metricas que se pueden leer, comparar y discutir.

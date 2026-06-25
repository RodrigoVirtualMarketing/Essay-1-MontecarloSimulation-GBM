# Finance and Statistics / Monte Carlo Simulation Engine / Stochastic Modeling via GBM

## Abstract

Este repositorio implementa un motor de simulacion Monte Carlo para estimar trayectorias potenciales en acciones. Parte de una premisa clara: en sistemas complejos, la prediccion determinista es una mala promesa. El trabajo serio no consiste en decir "este sera el precio", sino en cuantificar incertidumbre, medir dispersion y construir una hoja de ruta probabilistica.

El objetivo no es senalar un precio futuro unico. El objetivo es modelar una distribucion de resultados posibles que permita leer centro, rango, riesgo de cola y estabilidad del modelo. En castellano directo: no buscamos adivinar, buscamos estimar mejor.

**Palabras clave:** simulacion Monte Carlo, movimiento browniano geometrico, estimar vs predecir, riesgo de cola, decisiones bajo incertidumbre.

## Introduccion

Sostengo que "predecir" es un termino impreciso para mercados. El concepto mas correcto es estimar. Una simulacion Monte Carlo permite construir miles de caminos posibles y observar que rangos aparecen con mas frecuencia. Esa lectura no elimina la incertidumbre; la ordena.

El proposito real de modelar no es fabricar certeza. Es proporcionar un marco informado para tomar decisiones bajo incertidumbre, cuantificando si existe o no una ventaja estadistica razonable.

En esta version, el proyecto mantiene la base GBM, pero refuerza el analisis con herramientas mas robustas: media dinamica, volatilidad reciente, volatilidad implicita cuando esta disponible, colas pesadas, persistencia de volatilidad, `VaR`, `CVaR` y error de calibracion.

## Revision 1: Direccion Del Proyecto

`draft-1` cierra la primera version: motor Monte Carlo para acciones, reporte PDF y lectura de riesgo.

`revision-1` abre la siguiente fase: convertir el proyecto en una arquitectura de decision bajo incertidumbre. El foco ya no es solo simular precios, sino evaluar decisiones con valor esperado positivo, control de drawdown y estado de calibracion del modelo.

La nueva fase se apoya en dos documentos:

- [Arquitectura de conocimiento cuantitativo](docs/revision-1-quant-architecture.md): marco conceptual de probabilidad, validacion, no-estacionariedad y portafolio.
- [Marco matematico acciones/UFC](docs/revision-1-math-framework.md): formato comun para acciones, UFC, EV, Kelly fraccional, circuit breaker y ranking ajustado por riesgo.

Principio rector:

```text
no se busca predecir
se busca decidir con incertidumbre, edge positivo y riesgo sobrevivible
```

La integracion futura con UFC usara un proyecto externo de Google Colab con modelos de IA. Este repositorio no reescribira ese motor al inicio; primero definira una interfaz para recibir probabilidades, calcular EV, comparar contra cuotas y controlar sizing.

## Marco Teorico

Bajo esta optica, las generalizaciones dejan de ser etiquetas y pasan a ser estimaciones de comportamiento agregado. No pretendemos describir la accion individual final. Buscamos estimar la respuesta del sistema usando patrones observados en la serie historica.

La ausencia de certeza no significa ausencia de estructura. En mercados, lo real no aparece como un evento puntual que podamos adivinar, sino como una distribucion que podemos medir. La pregunta relevante no es "que va a pasar", sino "que rango parece razonable y que tan caro es equivocarse".

La evidencia util esta en la dispersion, en el margen de error y en la estabilidad de las distribuciones. Si el modelo cambia demasiado rapido, el reporte debe advertirlo. Por eso se incluye `Error calibracion`: no como verdad absoluta, sino como senal de fragilidad reciente en media y volatilidad.

## Marco Metodologico

El proyecto trata el precio como una serie de tiempo. Cada cierre diario alimenta retornos logaritmicos. Esos retornos se usan para estimar el movimiento reciente del activo y simular escenarios futuros.

El algoritmo tiene seis responsabilidades:

1. Descargar y limpiar precios historicos recientes con `yfinance`.
2. Separar la serie de cierre ajustado por accion.
3. Filtrar una ventana de calibracion compatible con el regimen actual.
4. Estimar volatilidad dinamica con EWMA y, cuando sea posible, complementar con volatilidad implicita de opciones.
5. Ejecutar simulaciones Monte Carlo con GBM, choques `t-Student` y persistencia de volatilidad.
6. Procesar resultados en terminos de mediana, rango 95%, `VaR 5%`, `CVaR 5%`, score de asimetria y estado de calibracion.

La implementacion usa `pandas` para datos, `numpy` para computo numerico, `scipy` para la distribucion `t-Student`, `yfinance` para precios/opciones y `matplotlib`/`seaborn` para el informe visual.

## Resultado Esperado

El resultado final no es "la respuesta correcta" del mercado. Es una hoja de ruta probabilistica. El usuario puede cambiar la accion, el universo de empresas, la cantidad de simulaciones o el horizonte temporal para observar como cambia el rango de escenarios.

El output principal es un PDF minimalista con metricas de valor:

- Centro: `Mediana simulada`.
- Rango: `Piso 95%` y `Techo 95%`.
- Riesgo de cola: `VaR 5%` y `CVaR 5%`.
- Asimetria: `Score asimetria`.
- Estado del modelo: `CALIBRADO` o `MODELO_DESCALIBRADO`.
- Comparacion: ranking solo entre acciones rankeables bajo la misma regla.

## Flujo De Datos

En una accion, la data representa la historia de precios ajustados de una empresa. El flujo es: descargar serie historica, separar cierre, calcular retornos, estimar media/volatilidad dinamica y simular escenarios.

En varias acciones, la data representa un grupo comparable de empresas tratadas con el mismo criterio. Cada accion conserva su propia serie, su propia volatilidad y su propio riesgo de cola. Luego los resultados se consolidan para comparar centro, rango, perdida esperada en malos escenarios y estabilidad reciente.

## Uso Del Proyecto

El proyecto tiene dos formas de uso:

- `monte_carlo_simulatio_proyect.py`: motor principal. Descarga data, simula escenarios y genera un informe PDF.
- `MONTE_CARLO_SIMULATIO_PROYECT.ipynb`: notebook de uso guiado para revisar el flujo y ejecutar el motor paso a paso.

## Que Produce El Motor

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

La presentacion del reporte esta estandarizada como informe ejecutivo: misma hoja, mismos margenes, mismo encabezado, mismo pie y misma jerarquia visual. La lectura esta pensada para un inversionista: primero oportunidad, luego perdida mala, luego estabilidad del modelo.

## Enfoque Quant Operativo

El flujo sigue esta regla:

```text
data -> regla -> prueba -> riesgo -> comparacion
```

- `data`: precios historicos descargados desde `yfinance`.
- `regla`: retornos logaritmicos, media dinamica y volatilidad dinamica.
- `prueba`: simulaciones Monte Carlo sobre un horizonte definido.
- `riesgo`: rango 95%, `VaR 5%`, `CVaR 5%` y error de calibracion.
- `comparacion`: todas las acciones pasan por la misma regla.

## Modelo Actual En Codigo

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
| `DATA_START_DATE` | Fecha inicial de descarga | Define desde cuando se baja data de mercado |
| `CALIBRATION_START_DATE` | Fecha minima de calibracion | Evita usar regimenes demasiado antiguos |
| `TICKERS` | Lista de acciones | Define el universo comparado |
| `SINGLE_TICKER` | Accion individual | Sirve para pruebas de una sola accion |
| `NUM_SIMULATIONS` | Cantidad de caminos simulados | Mas simulaciones reducen ruido estadistico |
| `NUM_DAYS_SINGLE_TICKER` | Horizonte individual | Dias simulados para una accion |
| `NUM_DAYS_MULTIPLE_TICKERS` | Horizonte comparativo | Dias simulados para varias acciones |
| `LOOKBACK_DAYS` | Ventana efectiva de calibracion | Limita cuantos dias recientes alimentan parametros |
| `ROLLING_WINDOW` | Ventana reciente | Base para varianza de largo plazo |
| `EWMA_LAMBDA` | Peso de volatilidad reciente | Controla sensibilidad a cambios de regimen |
| `STUDENT_T_DF` | Grados de libertad t-Student | Menor valor implica colas mas pesadas |
| `VAR_LEVEL` | Nivel de cola | Define el percentil usado para VaR y CVaR |
| `CALIBRATION_ERROR_THRESHOLD` | Umbral de descalibracion | Activa el circuit breaker del modelo |
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
| `Score asimetria` | Mediana de retorno dividida entre perdida mala absoluta |
| `Sigma usada diaria` | Volatilidad diaria usada por el motor |
| `Error calibracion` | Alerta de inestabilidad reciente en media y volatilidad |
| `Estado modelo` | Define si el activo entra al ranking o requiere recalibracion |

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

Una accion con `Error calibracion` alto queda marcada como `MODELO_DESCALIBRADO`: no debe rankearse como si sus metricas tuvieran el mismo peso que las demas.

El diagnostico del sistema debe leerse asi:

- `Centro`: donde cae la mediana de los escenarios simulados.
- `Rango`: cuanto se abre la distribucion de precios posibles.
- `Perdida mala`: cuanto podria doler el peor 5% de escenarios.
- `Asimetria`: si el centro compensa la perdida mala.
- `Estado modelo`: si la distribucion puede entrar al ranking o requiere recalibracion.

El informe sirve para comparar escenarios bajo incertidumbre. No reemplaza decision de inversion, control de riesgo ni validacion fuera de muestra.

## Referencias

- QUANT GUILD / Roman: [How to Quant Trade in 3 Minutes](https://youtu.be/mZLNzqDZHbA)
- QUANT GUILD / Roman: [Time Series Analysis for Quant Finance](https://youtu.be/JwqjuUnR8OY)
- Video base del proyecto: <https://youtu.be/fO-lGzZADVU>

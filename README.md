"El código fuente de este proyecto está bajo licencia MIT. Sin embargo, el texto del ensayo, las explicaciones, los gráficos y la estructura del curso están protegidos por derechos de autor. No se permite la reproducción, distribución o venta de este material sin permiso explícito del autor."

# Finance and Statistics / Monte Carlo Simulation Engine / Stochastic Modeling via GBM

#### Abstract
Este repositorio implementa un motor de Simulación de Monte Carlo diseñado para la estimación de trayectorias potenciales en activos de renta variable. Partiendo de la premisa epistemológica de que la predicción determinista es una falacia en sistemas complejos, este proyecto desplaza el enfoque hacia la cuantificación de la incertidumbre. El objetivo no es señalar un precio futuro único, sino modelar una distribución de densidades que revele el espectro de resultados probables basados en la volatilidad histórica y la deriva del activo.

**Palabras Clave: _simulación montecarlo, movimiento browniano geométrico, estimar vs predecir_**

### INTRODUCCIÓN
Sostengo la convicción de que "predecir" es un término inapropiado; en su defecto el concepto más preciso es estimar. La literatura sobre sistemas y modelos acuerda que lo posible es estimar el comportamiento agregado de un conjunto de variables o eventos aleatorios independientes mas no predecir de manera determinista.

Herramientas como las simulaciones de Monte Carlo permiten estimar un valor razonable de un activo mediante la simulación de miles de trayectorias posibles, extrayendo el promedio como métrica de referencia. Por otro lado, los modelos ocultos de Markov se utilizan para inferir estados. Observando patrones en los datos observados. De manera que el proposito verdadero de modelar es proporcionar un marco informado para la toma decisiones bajo incertidumbre cuantificando una ventaja estadistica.

### MARCO TEÓRICO
Bajo esta óptica, las generalizaciones dejan de ser etiquetas para transformarse en estimaciones de comportamiento agregado. No pretendemos describir la acción individual final, sino estimar la respuesta del sistema basándonos en sus patrones recurrentes.

Esta distinción es crucial para no caer en la trampa del relativismo. A menudo se confunde la ausencia de una verdad absoluta con la falta de una estructura objetiva; sin embargo, técnicamente no nos enfrentamos a una "verdad relativa", sino a una incerteza inherente. La realidad objetiva no se manifiesta en el evento único que el humano intenta "predecir", sino en la consistencia de las leyes que permiten su estimación. En este contexto, la subjetividad no es un punto de vista válido, sino simplemente el ruido en la medición de un fenómeno que opera independientemente del observador.

La prueba irrefutable de esta mecánica subyacente reside en la estabilidad de las distribuciones. La evidencia de lo real no se encuentra en la capacidad de adivinar el futuro, sino en la precisión con la que podemos calcular el margen de error y la dispersión. Es la consistencia del azar, este motor que genera grandes volúmenes de resultados probables, lo que confirma que existe una estructura sólida debajo del caos aparente.

### MARCO METODOLÓGICO
Debemos entender los fenómenos como mecánica sin narrativa. Los eventos no son una cuestión de apreciación artística o interpretación moral, sino transiciones de estado en un sistema físico o biológico. Mientras que la "predicción" es una narrativa humana —un intento de imponer orden y sentido mediante el lenguaje—, la estimación estocástica trata al fenómeno por lo que es: una causalidad mecánica. Nuestra capacidad de análisis se limita, por tanto, a cartografiar la probabilidad de una ocurrencia dentro de un espectro de posibilidades, despojando al evento de su historia y devolviéndole su naturaleza de proceso.

Para dicha tarea es necesario armarse de un algoritmo con 4 responsabilidades:
1. Extracción y saneamiento de datos historicos (OHLCV) usando una libreria externa (yfinance, interactive brokers)
2. Cálculo de la media logarítmica ($\mu$) y la desviación estándar ($\sigma$) de los retornos diarios para definir el perfil de riesgo-retorno del activo.
3. Ejecución de $n$ simulaciones (ej. 10,000 iteraciones) sobre un horizonte temporal definido. El motor utiliza el Movimiento Browniano Geométrico (GBM), asumiendo que los cambios en el precio siguen una distribución log-normal, lo que permite capturar la naturaleza estocástica del mercado.
4. Procesamiento de los resultados finales.

La implementación se apoya en el ecosistema científico de Python: pandas para el gobierno de datos, numpy para el cómputo vectorial, y scipy.stats para la modelización de funciones de densidad normal. La capa visual se gestiona mediante matplotlib, permitiendo una interpretación intuitiva de la varianza simulada.

### RESULTADOS
El proyecto está diseñado para ser modular. El usuario puede manejar la simulación cambiando la acción o empresa analizada, ajustando la cantidad de simulaciones (`num_simulations`) o ampliando el horizonte de días (`num_days`) para ver cómo se degrada la certidumbre a largo plazo. Para el análisis de resultados, se calculan los precios finales de todas las simulaciones para hallar la mediana, los intervalos de confianza y la trayectoria de precio más representativa. Luego esas trayectorias se grafican para mostrar el rango de precios futuros posibles.

### FLUJO DE DATOS
En la sección de un ticker, la data representa la historia de precios ajustados de una sola acción o empresa. El flujo es simple: descargar la serie histórica, separar el precio de cierre, medir su variación diaria mediante retornos y usar ese comportamiento observado para construir escenarios probables hacia adelante.

En la sección de varias acciones, la data representa un grupo comparable de empresas tratadas con el mismo criterio estadístico. Cada empresa conserva su propia serie de cierres y su propia volatilidad observada; luego las simulaciones se corren por separado y los resultados se juntan para comparar rangos, medianas y trayectorias probables entre acciones.

### MAPA DE VARIABLES
La siguiente tabla conecta las variables principales del notebook con su papel dentro del flujo de datos y con lo que significan en el análisis.

| Variable | Tipo | Dónde aparece | Qué representa | Qué aporta al análisis |
| --- | --- | --- | --- | --- |
| `START_DATE` | Tiempo | Notebook | Fecha desde la cual se descargan los datos históricos | Define cuánta historia entra al modelo |
| `NUM_SIMULATIONS` | Cantidad | Notebook | Número de caminos simulados | Mientras más simulaciones haya, más sólido es el panorama estadístico |
| `NUM_DAYS_SINGLE_TICKER` | Horizonte | Caso un ticker | Número de días simulados para una acción | Define cuánto se proyecta hacia adelante en el caso individual |
| `NUM_DAYS_MULTIPLE_TICKERS` | Horizonte | Caso varios tickers | Número de días simulados para cada acción comparada | Permite comparar varias acciones bajo la misma ventana de tiempo |
| `ticker` | Identificador | Caso un ticker | Código de la acción o empresa analizada | Define qué empresa alimenta toda la simulación individual |
| `mis_tickers` | Conjunto de análisis | Caso varios tickers | Lista de acciones o empresas comparadas | Define qué empresas entran al análisis comparativo |
| `df` | Data histórica | Caso un ticker | Tabla con precios históricos descargados de una acción | Es la base desde la cual salen los retornos y el precio actual |
| `df2` | Data histórica conjunta | Caso varios tickers | Tabla con precios históricos de varias acciones | Permite separar y comparar cada serie dentro del mismo proceso |
| `close_prices` | Serie base | Ambos casos | Serie de precios de cierre ajustados | Resume el comportamiento principal del precio |
| `returns` | Serie derivada | Caso un ticker | Cambio porcentual diario del precio de cierre | Mide cómo se mueve la acción día a día |
| `mu` | Promedio | Ambos casos | Promedio de retornos diarios | Resume la tendencia media observada en la data |
| `sigma` | Volatilidad | Ambos casos | Desviación estándar de los retornos diarios | Mide qué tanto se dispersan los movimientos del precio |
| `last_price` | Punto de partida | Ambos casos | Último precio de cierre observado | Es el precio desde el cual arrancan las simulaciones |
| `simulation_df` | Resultado bruto | Ambos casos | Matriz con todas las trayectorias simuladas | Guarda todos los caminos posibles generados por el modelo |
| `final_prices` | Resultado final | Ambos casos | Último precio de cada simulación | Sirve para estudiar rangos, percentiles y distribución final |
| `median_final_price` / `median_final_prices` | Resumen | Ambos casos | Mediana de los precios finales simulados | Da un punto central más robusto que el promedio |
| `most_likely_price_index` | Selector | Ambos casos | Posición de la trayectoria final más cercana a la mediana | Ayuda a escoger un camino representativo |
| `most_likely_path` / `most_likely_price_simulation` | Trayectoria representativa | Ambos casos | Camino simulado cuya salida final queda más cerca de la mediana | Sirve para mostrar visualmente el escenario central |
| `all_ticker_simulation_dfs` | Contenedor | Caso varios tickers | Diccionario con simulaciones por acción | Guarda todos los resultados por empresa para revisarlos después |
| `all_ticker_most_likely_prices` | Resumen comparativo | Caso varios tickers | Precio final representativo por acción | Facilita la comparación directa entre empresas |
| `all_ticker_median_final_prices` | Resumen comparativo | Caso varios tickers | Mediana final por acción | Permite comparar el centro de cada distribución |
| `results` | Lista intermedia | Caso varios tickers | Lista de resultados resumidos | Prepara la salida para mostrarla en forma de tabla |
| `results_df` | Tabla final | Caso varios tickers | DataFrame final de comparación | Presenta los resultados principales de forma compacta |

### REFERENCIAS: {
    https://youtu.be/fO-lGzZADVU , 14 minute video that inspired this python project.
    https://youtu.be/-4sf43SLL3A , for proof reading and revisioning.
}

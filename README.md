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
El proyecto está diseñado para ser modular. El usuario puede orquestar la simulación modificando el ticker del activo, ajustando la densidad de las simulaciones (num_simulations) o expandiendo el horizonte temporal (num_days) para observar la degradación de la certidumbre a largo plazo. Para el analisis de resultados, los precios finales de todas las simulaciones son computados para hallar la mediana, los intervalos de confianza e identificar la trayectoria de precio más probable. Las trayectorias de precio son graficadas para representar el rango de futuros precios potenciales.

### REFERENCIAS: {
    https://youtu.be/fO-lGzZADVU , 14 minute video that inspired this python project.
    https://youtu.be/-4sf43SLL3A , for proof reading and revisioning.
}

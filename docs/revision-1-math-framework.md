# Revision 1: Marco Matematico Para Acciones Y UFC

Este documento convierte los comentarios del asesor en una estructura tecnica para la siguiente fase del proyecto. La idea no es mezclar acciones y UFC de forma narrativa. La idea es usar un mismo lenguaje cuantitativo para dos sistemas inciertos:

- acciones: precios, retornos, volatilidad, riesgo de cola.
- UFC: cuotas, probabilidad de victoria, varianza de resultados, riesgo de racha.

El nucleo comun es decision bajo incertidumbre.

```text
data -> probabilidad del modelo -> precio del mercado -> edge -> sizing -> control de riesgo
```

Este documento depende del marco rector:

```text
docs/revision-1-quant-architecture.md
```

La jerarquia correcta es:

```text
arquitectura de conocimiento -> formato matematico -> implementacion -> reporte
```

## 0. Reglas Rectoras De Revision 1

`revision-1` debe respetar cuatro principios:

| Principio | Regla practica |
| --- | --- |
| Sistema incierto | No asumir probabilidades fijas como en dados o ruleta |
| Validacion | Evitar look-ahead bias, p-hacking, survivorship bias y overfitting |
| No-estacionariedad | Calibrar con regimen reciente, no con historia muerta |
| Politica de decision | Entregar accion, tamano, riesgo y estado del modelo |

La salida final no debe responder:

```text
que va a pasar?
```

Debe responder:

```text
que decision tiene EV positivo, riesgo controlado y modelo calibrado?
```

## 1. Problemas Detectados En El Draft 1

### 1.1 Ventana Historica Demasiado Larga

El Draft 1 descarga data desde `1950-01-01`. Para un horizonte de simulacion cercano a 20 dias, esa ventana es demasiado amplia.

Problema:

```text
data vieja + horizonte corto = mezcla de regimenes incompatibles
```

La solucion en `revision-1` debe ser separar:

- `DATA_START_DATE`: fecha desde donde se descarga informacion.
- `CALIBRATION_START_DATE`: fecha desde donde se calibra el modelo.
- `LOOKBACK_DAYS`: cantidad efectiva de dias usados para estimar parametros.

Propuesta inicial:

```text
DATA_START_DATE        = "2010-01-01"
CALIBRATION_START_DATE = "2020-01-01"
LOOKBACK_DAYS          = 756    # aprox. 3 anos bursatiles
FORECAST_DAYS          = 22     # aprox. 1 mes bursatil
```

La data larga puede servir para contexto. La calibracion debe usar regimen reciente.

### 1.2 La Mediana No Debe Ser Tratada Como Precio Objetivo

El Draft 1 usa `Cambio % vs actual` como lectura principal. Eso puede inducir una contradiccion:

```text
si mediana < spot, el reporte dice "debil"
```

Pero una mediana debajo del precio actual no significa que la empresa sea debil. Puede significar:

- sobrecompra reciente.
- reversion a la media.
- compresion de expectativas.
- distribucion asimetrica.
- ruido de corto plazo.

Nueva regla:

```text
mediana simulada = centro de escenarios
no = precio objetivo
no = recomendacion direccional
```

La lectura debe pasar de "debil" a lenguaje de escenario:

| Lectura Antigua | Lectura Nueva |
| --- | --- |
| Debil | Centro debajo del spot |
| Favorable | Centro sobre spot con cola controlada |
| Baja confianza | Modelo descalibrado |
| Riesgo alto | Cola negativa amplia |

### 1.3 Circuit Breaker Para Modelos Inestables

Si `Error calibracion` supera el umbral definido, el activo no debe rankearse igual que los demas.

Regla:

```text
if calibration_error >= threshold:
    status = "MODELO_DESCALIBRADO"
    rankable = False
```

Lectura correcta:

```text
Modelo descalibrado - requiere ajuste de parametros
```

No debe presentarse como:

```text
Baja confianza
```

Motivo: si media y volatilidad estan mutando rapido, las metricas de distribucion final pierden calidad para decision.

### 1.4 Ranking Debe Ser Ajustado Por Riesgo De Cola

Ordenar por `Cambio % vs actual` es incompleto. Una accion con centro positivo puede tener una cola negativa demasiado grande.

Metrica propuesta:

```text
tail_adjusted_score = median_return / abs(cvar_5_return)
```

Interpretacion:

```text
cuanto centro esperado recibo por cada unidad de perdida mala
```

Regla:

```text
score alto  = mejor asimetria
score bajo  = el centro no compensa la cola
score <= 0  = centro simulado debajo del spot
```

## 2. Estructura Matematica Comun

Todo evento se modela como una decision con pago incierto.

```text
X_t      = informacion disponible hoy
M_t      = precio observado en el mercado
P_model  = probabilidad estimada por el modelo
P_market = probabilidad implicita por el mercado
Payoff   = resultado monetario si la decision se ejecuta
EV       = valor esperado
```

La funcion general:

```text
EV(a | X_t) = E[Payoff(a, Y) | X_t] - Costos - Penalizacion_Riesgo
```

Donde:

- `a`: accion tomada: comprar, no comprar, apostar, no apostar, reducir tamano.
- `Y`: resultado futuro incierto.
- `X_t`: estado actual del sistema.
- `EV`: ventaja esperada de la decision.

Decision:

```text
ejecutar solo si EV > 0 y el modelo esta calibrado
```

## 3. Aplicacion En Acciones

### 3.1 Variables

```text
S_0       = precio actual
S_T       = precio simulado al final del horizonte
R_T       = retorno simulado = (S_T / S_0) - 1
median_R  = mediana de retornos simulados
VaR_5     = percentil 5% de retornos
CVaR_5    = perdida promedio dentro del peor 5%
```

### 3.2 Score De Asimetria

```text
asymmetry_score = median_R / abs(CVaR_5)
```

Ejemplo de lectura:

```text
median_R = 0.02
CVaR_5   = -0.10
score    = 0.20
```

Lectura:

```text
por cada 1 unidad de perdida mala, el centro aporta 0.20 unidades de retorno esperado
```

### 3.3 Circuit Breaker

```text
if calibration_error >= 0.08:
    status = "MODELO_DESCALIBRADO"
    show_distribution = False
    include_in_ranking = False
```

Visualmente:

- metricas en gris.
- no entra al ranking principal.
- se muestra en una seccion de activos a recalibrar.

## 4. Aplicacion En UFC

Una pelea UFC se modela como evento binario con pago discreto.

```text
fighter_a_win = 1 si gana A
fighter_a_win = 0 si pierde A
```

### 4.1 Probabilidad Implicita Del Mercado

Con cuota decimal:

```text
P_market = 1 / odds_decimal
```

Si hay margen de la casa, se normaliza:

```text
P_market_clean = P_market_fighter / sum(P_market_all_fighters)
```

### 4.2 Probabilidad Del Modelo

```text
P_model = f(striking, grappling, cardio, edad, reach, defensa, ritmo, inactividad, cambio_categoria)
```

Decision actual del proyecto:

```text
P_model UFC vendra de un proyecto existente en Google Colab con modelos de IA.
```

Por lo tanto, `revision-1` no debe decidir todavia si el mejor baseline es regresion logistica, arboles, Poisson o Markov. Esa decision vive en el Colab. Este repositorio debe importar el motor, envolver su salida y convertirla en decision cuantitativa.

Contrato minimo que debe entregar el proyecto Colab:

| Campo | Tipo | Uso |
| --- | --- | --- |
| `fighter_a` | texto | Peleador A |
| `fighter_b` | texto | Peleador B |
| `p_fighter_a` | float entre 0 y 1 | Probabilidad modelo de victoria A |
| `p_fighter_b` | float entre 0 y 1 | Probabilidad modelo de victoria B |
| `model_confidence` | float entre 0 y 1 | Confianza del modelo |
| `model_version` | texto | Version del modelo IA |
| `features_used` | lista | Variables usadas por el modelo |
| `data_quality_flag` | texto | Estado de calidad de data |

Salida esperada:

```text
fight_id, fighter_a, fighter_b, p_fighter_a, p_fighter_b,
model_confidence, model_version, data_quality_flag
```

Modelos posibles dentro del Colab:

- regresion logistica: baseline interpretable.
- random forest / gradient boosting: mejor captura de interacciones.
- modelo bayesiano: util si hay poca data o incertidumbre alta.
- Markov por round: version avanzada para simular transiciones de control, golpeo y sumision.

Recomendacion de arquitectura:

```text
no reescribir el modelo IA en este repo al inicio
importar primero la salida probabilistica del Colab
```

Motivo:

- evita duplicar logica.
- conserva el trabajo ya hecho.
- permite auditar primero la calidad de la probabilidad.
- separa prediccion UFC de decision cuantitativa.

### 4.3 Valor Esperado De Una Apuesta

Para cuota decimal:

```text
b = odds_decimal - 1
q = 1 - P_model

EV = (P_model * b) - q
```

Decision:

```text
apostar solo si EV > 0
```

### 4.4 Kelly Fraccional

Kelly completo:

```text
f_star = ((b * P_model) - q) / b
```

Kelly fraccional:

```text
position_size = lambda * f_star
```

Donde:

```text
lambda = 0.10 a 0.25
```

Uso:

- reduce varianza.
- protege contra error del modelo.
- evita ruina por rachas.

## 5. Puente Entre Acciones Y UFC

El punto comun no es que ambos "se parezcan". El punto comun es que ambos producen decisiones con incertidumbre y capital limitado.

| Dimension | Acciones | UFC |
| --- | --- | --- |
| Precio de mercado | `S_0` | cuota decimal |
| Probabilidad del modelo | distribucion Monte Carlo | probabilidad de victoria |
| Precio implicito | spot / opciones | probabilidad implicita de cuota |
| Edge | asimetria retorno-cola | `P_model - P_market` |
| Riesgo de cola | `CVaR 5%` | drawdown por racha |
| Circuit breaker | error de calibracion | baja data / cambio de categoria / inactividad |
| Sizing | capital por accion | Kelly fraccional |

## 6. Portafolio Hibrido

Si las decisiones en UFC son independientes de acciones, se puede estudiar diversificacion de flujos.

Formula:

```text
sigma_portfolio^2 =
    w_stock^2 * sigma_stock^2
  + w_ufc^2 * sigma_ufc^2
  + 2 * w_stock * w_ufc * cov(stock, ufc)
```

Hipotesis:

```text
cov(stock, ufc) aprox. 0
```

Advertencia:

```text
covarianza baja no elimina riesgo de modelo ni riesgo de mala calibracion
```

## 7. Flujo Propuesto Revision 1

```text
                +---------------------+
                |  Datos de mercado   |
                | acciones / UFC      |
                +----------+----------+
                           |
                           v
                +---------------------+
                | Limpieza y estado   |
                | X_t                 |
                +----------+----------+
                           |
                           v
                +---------------------+
                | Modelo probabilista |
                | P_model / dist.     |
                +----------+----------+
                           |
                           v
                +---------------------+
                | Precio del mercado  |
                | spot / odds         |
                +----------+----------+
                           |
                           v
                +---------------------+
                | Edge y riesgo       |
                | EV / CVaR / DD      |
                +----------+----------+
                           |
                           v
                +---------------------+
                | Circuit breaker     |
                | calibrated?         |
                +----------+----------+
                           |
          +----------------+----------------+
          |                                 |
          v                                 v
+---------------------+          +---------------------+
| Ranking permitido   |          | Requiere ajuste     |
| decision / sizing   |          | no rankear          |
+---------------------+          +---------------------+
```

## 8. Implementacion Recomendada Por Fases

### Fase 1: Corregir El Motor De Acciones

- reducir ventana efectiva de calibracion.
- separar data historica de data usada para parametros.
- reemplazar ranking por `asymmetry_score`.
- agregar `circuit_breaker_status`.
- no rankear activos descalibrados.
- cambiar lenguaje de lectura: escenario, no prediccion.

### Fase 2: Crear Esqueleto UFC

- crear `ufc_model/`.
- definir esquema de datos de pelea.
- importar o adaptar el notebook de Google Colab.
- crear un wrapper que convierta la salida del modelo IA a `P_model`.
- implementar conversion de odds a probabilidad implicita.
- implementar EV y Kelly fraccional.
- registrar `model_version`, `model_confidence` y `data_quality_flag`.

### Fase 3: Integracion Hibrida

- reporte paralelo: acciones y UFC.
- capital allocation comun.
- comparacion por EV ajustado a riesgo.
- medicion de drawdown conjunto.
- matriz de independencia o correlacion empirica si hay historial suficiente.

## 9. Decision De Diseno

La siguiente version no debe decir:

```text
esta accion es debil
```

Debe decir:

```text
el centro simulado queda debajo del spot y la cola no compensa
```

La siguiente version no debe decir:

```text
este peleador gana
```

Debe decir:

```text
la probabilidad del modelo supera la probabilidad implicita de la cuota y el EV es positivo
```

Ese es el puente correcto entre acciones y UFC.

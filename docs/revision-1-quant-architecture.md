# Revision 1: Arquitectura De Conocimiento Cuantitativo

Este documento define la base conceptual de `revision-1`. No es una nota teorica aislada. Es el marco que debe gobernar las siguientes decisiones de diseno del proyecto.

La tesis central:

```text
no se predice el futuro
se decide bajo incertidumbre con una politica que busca EV positivo y control de drawdown
```

## 1. Nivel 1: Marco Probabilistico

### Concepto Critico

La diferencia entre sistema aleatorio y sistema incierto.

Un sistema aleatorio tiene probabilidades fijas. Ejemplo: dados, ruleta, moneda justa. En ese caso, la Ley de los Grandes Numeros funciona de manera limpia: con suficientes repeticiones, el promedio observado converge al valor teorico.

Un sistema incierto no tiene probabilidades fijas. Ejemplo: mercados financieros, poker, UFC. Las reglas existen, pero las probabilidades cambian con informacion nueva, regimen, participantes y contexto.

### Error Comun

Tratar el mercado como si fuera una ruleta estacionaria.

```text
error = asumir que el pasado genera el futuro con la misma distribucion
```

### Regla Para Este Proyecto

El motor no debe prometer prediccion. Debe estimar una distribucion condicional a la informacion disponible.

```text
P(resultado | informacion actual, estado del modelo)
```

## 2. Nivel 2: Modelado Y Validacion

### Concepto Critico

Un backtest bueno no prueba que una estrategia sea buena. Puede probar que el modelo memorizo ruido.

### Riesgos Principales

| Riesgo | Definicion | Control |
| --- | --- | --- |
| Look-ahead bias | Usar informacion que no existia al momento de decidir | Cortar data por fecha real de disponibilidad |
| P-hacking | Probar parametros hasta encontrar uno que funcione por azar | Validacion fuera de muestra y pocos parametros defendibles |
| Survivorship bias | Ignorar activos que desaparecieron | Dataset que incluya fallidos o advertencia explicita |
| Overfitting | Aprender ruido historico | Walk-forward y sensibilidad de parametros |

### Regla Para Este Proyecto

Toda mejora debe responder:

```text
que informacion estaba disponible en ese momento?
la regla sobrevive cambios leves de parametros?
existe razon economica o deportiva para el efecto?
```

## 3. Nivel 3: No-Estacionariedad Y Filtros Adaptativos

### Concepto Critico

La media y la varianza cambian. Si cambian, un promedio largo puede ser informacion muerta.

El Filtro de Kalman ayuda a estimar un estado oculto con mediciones ruidosas. EWMA ayuda a que la volatilidad reciente pese mas que la antigua. GARCH o mecanismos similares permiten persistencia de volatilidad.

### Limite Importante

Un filtro de Kalman simple actualiza el estado. No necesariamente actualiza el modelo de fondo.

```text
si el regimen cambia fuerte, el filtro puede llegar tarde
```

### Regla Para Este Proyecto

`revision-1` debe separar historia descargada de historia usada para calibrar.

```text
data historica = contexto
ventana de calibracion = parametros actuales
```

Tambien debe existir un disyuntor:

```text
si el error de calibracion es alto:
    no rankear
    marcar como modelo descalibrado
    pedir recalibracion
```

## 4. Nivel 4: Portafolio Y Extraccion De Alpha

### Concepto Critico

Alpha real no es lo mismo que beta apalancado.

Alpha real es retorno que no depende de los factores comunes del mercado. Si una estrategia solo gana cuando el mercado sube, probablemente es beta, no alpha.

### Aplicacion En Este Proyecto

La integracion futura de UFC y acciones tiene sentido si UFC aporta una fuente de retorno con baja dependencia del mercado financiero.

```text
acciones = distribuciones de precio y riesgo de cola
UFC      = probabilidades de eventos y valor esperado contra cuotas
```

El puente no es narrativo. Es matematico:

```text
EV > 0
riesgo controlado
sizing disciplinado
drawdown sobrevivible
```

## 5. Funcion De Politica

La salida del sistema no debe ser una prediccion. Debe ser una politica de accion.

```text
policy(X_t) -> accion, tamano, motivo, riesgo, estado del modelo
```

Donde:

- `X_t`: informacion disponible hoy.
- `accion`: comprar, no comprar, apostar, no apostar, reducir, esperar.
- `tamano`: exposicion sugerida por control de riesgo.
- `motivo`: edge, asimetria o EV.
- `riesgo`: CVaR, drawdown o perdida mala.
- `estado_modelo`: calibrado, vigilar, descalibrado.

## 6. Separacion Entre Hecho, Supuesto, Opinion Y Decision

`revision-1` debe evitar mezclar afirmaciones.

| Tipo | Ejemplo | Como Debe Aparecer |
| --- | --- | --- |
| Hecho | `CVaR 5% = -12.8%` | Metrica calculada |
| Supuesto | `cov(stock, UFC) aprox. 0` | Hipotesis a validar |
| Opinion | `stock picking es fragil` | Tesis estrategica, no verdad universal |
| Decision | `no rankear por descalibracion` | Regla del sistema |

Esta separacion mejora la calidad academica del proyecto y evita lenguaje excesivamente determinista.

## 7. Implicancias Para El Motor Actual

Cambios que deben entrar en la siguiente iteracion:

- reducir la ventana efectiva de calibracion.
- reemplazar etiquetas como `debil` por lenguaje de escenario.
- agregar `asymmetry_score = median_return / abs(CVaR_5)`.
- agregar `circuit_breaker_status`.
- separar activos rankeables de activos descalibrados.
- fusionar upside y downside en una lectura de asimetria.
- documentar limites del modelo en el reporte PDF.

## 8. Implicancias Para UFC

El modelo UFC ya existe como recurso externo en notebooks de Colab. Este repo no debe reescribirlo al inicio.

La primera integracion debe exigir una salida estandar:

```text
fight_id
fighter_a
fighter_b
p_fighter_a
p_fighter_b
model_confidence
model_version
data_quality_flag
```

Luego este repo calcula:

```text
probabilidad implicita de cuota
edge
EV
Kelly fraccional
drawdown potencial
decision
```

## 9. Diagrama Rector

```text
                          +----------------------+
                          | Informacion actual   |
                          | mercado / pelea      |
                          +----------+-----------+
                                     |
                                     v
                          +----------------------+
                          | Modelo probabilista  |
                          | distribucion / P(win)|
                          +----------+-----------+
                                     |
                                     v
                          +----------------------+
                          | Precio de mercado    |
                          | spot / cuota         |
                          +----------+-----------+
                                     |
                                     v
                          +----------------------+
                          | Edge y riesgo        |
                          | EV / CVaR / DD       |
                          +----------+-----------+
                                     |
                                     v
                          +----------------------+
                          | Estado del modelo    |
                          | calibrado?           |
                          +----------+-----------+
                                     |
                    +----------------+----------------+
                    |                                 |
                    v                                 v
          +---------------------+           +---------------------+
          | Politica activa     |           | No operar / ajustar |
          | sizing + decision   |           | modelo descalibrado |
          +---------------------+           +---------------------+
```

## 10. Regla Final

El proyecto debe pasar de:

```text
que va a pasar?
```

a:

```text
que decision tiene EV positivo, riesgo controlado y modelo suficientemente calibrado?
```

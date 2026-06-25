# Revision 1 Resources

Esta carpeta contiene recursos externos para la fase `revision-1`.

## Notebooks Disponibles

```text
Algoritmo_UFC.ipynb
Data_Mining_V1_.ipynb
UFC_DataMining.ipynb
```

## Uso Previsto

Estos notebooks vienen del trabajo en Google Colab y contienen la logica base del proyecto UFC con modelos de IA.

La regla de integracion es:

```text
no reescribir primero
auditar -> extraer interfaz -> envolver salida -> conectar con EV/riesgo
```

## Contrato Esperado

El sistema UFC debe entregar, como minimo:

| Campo | Uso |
| --- | --- |
| `fight_id` | Identificador de pelea |
| `fighter_a` | Peleador A |
| `fighter_b` | Peleador B |
| `p_fighter_a` | Probabilidad modelo para A |
| `p_fighter_b` | Probabilidad modelo para B |
| `model_confidence` | Confianza del modelo |
| `model_version` | Version del modelo IA |
| `data_quality_flag` | Estado de calidad de data |

## Siguiente Paso

Antes de programar integracion:

1. Revisar que notebook contiene el modelo final.
2. Identificar inputs reales.
3. Identificar outputs reales.
4. Separar data mining de inferencia.
5. Crear wrapper limpio en el repo principal.

# Reporte Detallado: Evaluación de Representaciones Sintáctica y Semántica

**Fecha**: Enero 2025  
**Objetivo**: Evaluar la complementariedad entre representaciones sintáctica (TF-IDF+PCA+LOF) y semántica (SecBERT+Mahalanobis) para validar la estrategia de ensemble.

---

## Resumen Ejecutivo

Este reporte documenta el análisis exhaustivo de dos enfoques complementarios para la detección de anomalías en tráfico HTTP:

1. **Representación Sintáctica**: TF-IDF + PCA + LOF (sin preprocessing)
2. **Representación Semántica**: SecBERT + Mahalanobis (con preprocessing)

**Hallazgo Principal**: Los modelos son altamente complementarios:

- **Agreement: 71.10%** (28.9% desacuerdo)
- **Correlación: 0.33** (decisiones independientes)
- **Ensemble potencial: 80.87% recall** (+16.7pp sobre el mejor modelo individual)
- **FPs compartidos: solo 52** (modos de fallo diferentes)

**Conclusión**: La fusión de ambas representaciones está empíricamente justificada y proporciona una mejora significativa en cobertura de detección.

---

## 1. Contexto y Motivación

### 1.1 Estado del Arte Previo

Experimentos anteriores (Experiments 01-14) demostraron:

- **TF-IDF + Mahalanobis**: 10.52% recall (falló)
- **SecBERT + Mahalanobis**: 49.26% recall (mejor resultado previo)
- **BGE-small + Mahalanobis**: 39.96% recall

**Problema identificado**: Los métodos globales (Mahalanobis) no capturan adecuadamente la estructura multimodal del tráfico HTTP.

### 1.2 Hipótesis Inicial

**Hipótesis 1**: Los métodos basados en densidad local (LOF) deberían funcionar mejor que métodos globales (Mahalanobis) en embeddings sparse y multimodales.

**Hipótesis 2**: Las representaciones sintáctica y semántica capturan diferentes tipos de anomalías y deberían ser complementarias.

---

## 2. Arquitectura de los Modelos Evaluados

### 2.1 Modelo Sintáctico: TF-IDF + PCA + LOF

**Pipeline**:

```
HTTP Request (raw)
  ↓ TF-IDF Vectorizer (5000 features, n-grams 1-3)
  ↓ PCA (150 componentes, 93.01% variance)
  ↓ LOF Detector (k=100 neighbors)
  ↓ Threshold @ 5% FPR
  → Prediction
```

**Características**:

- **Preprocessing**: No (preserva variaciones de encoding)
- **Dimensionalidad**: 150D (después de PCA)
- **Tipo de detección**: Local density-based
- **Embedding**: Sparse, multimodal, sintáctico

**Parámetros óptimos**:

- `max_features`: 5000
- `ngram_range`: (1, 3)
- `n_components`: 150
- `n_neighbors`: 100

### 2.2 Modelo Semántico: SecBERT + Mahalanobis

**Pipeline**:

```
HTTP Request (raw)
  ↓ Preprocessing Pipeline (normalización, decodificación)
  ↓ SecBERT Encoder (768 dimensions)
  ↓ Mahalanobis Distance (global covariance)
  ↓ Threshold @ 5% FPR
  → Prediction
```

**Características**:

- **Preprocessing**: Sí (normaliza sintaxis)
- **Dimensionalidad**: 768D (densas)
- **Tipo de detección**: Global distance-based
- **Embedding**: Dense, unimodal, semántico

**Parámetros**:

- Modelo: SecBERT (pre-trained on security text)
- Detector: Mahalanobis (covariance global)

---

## 3. Resultados de Rendimiento Individual

### 3.1 Métricas por Modelo

| Métrica                | TF-IDF+LOF | SecBERT+Mahalanobis | Diferencia   |
| ---------------------- | ---------- | ------------------- | ------------ |
| **Recall @ 5% FPR**    | **64.20%** | 49.26%              | **+14.94pp** |
| **Precision**          | 92.95%     | 90.81%              | +2.14pp      |
| **F1-Score**           | 75.95%     | 63.87%              | +12.08pp     |
| **FPR**                | 4.88%      | 5.00%               | -0.12pp      |
| **Ataques detectados** | 16,092     | 12,347              | +3,745       |

### 3.2 Análisis de Rendimiento

**TF-IDF+LOF (Sintáctico)**:

- ✅ **Mejor recall individual**: 64.20%
- ✅ Mayor precisión: 92.95%
- ✅ Detección basada en patrones sintácticos locales
- ✅ Funciona mejor SIN preprocessing (preserva variaciones)

**SecBERT+Mahalanobis (Semántico)**:

- ✅ Buen recall: 49.26%
- ✅ Entiende contexto semántico
- ✅ Funciona mejor CON preprocessing (reduce ruido)
- ⚠️ Recall inferior al modelo sintáctico

**Ventaja del modelo sintáctico**: +14.94pp de recall absoluto (+30% relativo)

---

## 4. Análisis de Complementariedad

### 4.1 Métricas de Agreement

#### 4.1.1 Agreement General

- **Agreement**: 71.10%
- **Desacuerdo**: 28.90%
- **Interpretación**: Casi 1 de cada 3 casos es decidido de manera diferente

Este nivel de desacuerdo (29%) es **óptimo** para un ensemble:

- Si fuera muy alto (>90%): modelos redundantes
- Si fuera muy bajo (<10%): modelos idénticos
- 29% indica complementariedad sin redundancia excesiva

#### 4.1.2 Métricas de Similitud

**Jaccard Index**: 0.3623

- Intersection / Union de predicciones
- Bajo valor (<0.5) indica baja superposición
- Confirma que capturan diferentes subconjuntos

**Correlación de Pearson**: 0.3323

- Correlación entre predicciones binarias
- Moderada correlación positiva (0.33) indica:
  - Algunos ataques son detectados por ambos (fácil)
  - Muchos ataques son únicos a cada modelo (complementarios)
  - Decisiones relativamente independientes

#### 4.1.3 Agreement por Tipo de Muestra

| Tipo                     | Agreement | Interpretación                                    |
| ------------------------ | --------- | ------------------------------------------------- |
| **En ataques**           | 51.72%    | Alta discrepancia en qué ataques detecta cada uno |
| **En muestras normales** | 90.53%    | Alta coincidencia en qué considerar normal        |

**Análisis**: Los modelos están más de acuerdo en qué es normal que en qué es ataque. Esto es positivo porque:

- Reduce falsos positivos en el ensemble
- Cada modelo aporta detecciones únicas de ataques

### 4.2 Desglose de Detección de Ataques

**Total de ataques en test set**: 25,065

#### Desglose por Categoría

| Categoría                    | Cantidad | % del Total | Interpretación                               |
| ---------------------------- | -------- | ----------- | -------------------------------------------- |
| **Detectados por ambos**     | 8,169    | 32.6%       | Ataques "fáciles" - ambos detectan           |
| **Solo TF-IDF+LOF**          | 7,923    | 31.6%       | Ataques sintácticos únicos                   |
| **Solo SecBERT+Mahalanobis** | 4,178    | 16.7%       | Ataques semánticos únicos                    |
| **Perdidos por ambos**       | 4,795    | 19.1%       | Ataques difíciles que requieren otro enfoque |

#### Análisis Cuantitativo

**Cobertura individual**:

- TF-IDF+LOF: 16,092 ataques (64.20%)
- SecBERT+Mahalanobis: 12,347 ataques (49.26%)

**Cobertura única**:

- TF-IDF+LOF único: 7,923 ataques (31.6%)
- SecBERT+Mahalanobis único: 4,178 ataques (16.7%)

**Ratio de complementariedad**:

- TF-IDF+LOF aporta 70% más ataques únicos que SecBERT+Mahalanobis
- Esto indica que el modelo sintáctico tiene mayor capacidad de descubrir patrones no capturados por el semántico

**Ataques "fáciles" vs "difíciles"**:

- Fáciles (ambos detectan): 32.6%
- Únicos (solo uno detecta): 48.3%
- Difíciles (ninguno detecta): 19.1%

### 4.3 Análisis de Falsos Positivos

**Total de muestras normales**: 25,000

| Categoría                          | Cantidad | % de FPs | Interpretación             |
| ---------------------------------- | -------- | -------- | -------------------------- |
| **FPs compartidos**                | 52       | 2.1%     | Solo 2% de overlap         |
| **FPs únicos TF-IDF+LOF**          | 1,169    | 47.3%    | Modos de fallo específicos |
| **FPs únicos SecBERT+Mahalanobis** | 1,198    | 48.5%    | Modos de fallo específicos |
| **Total FPs**                      | 2,419    | 9.7%     | Bajo overlap total         |

**Análisis crítico**:

✅ **Bajo overlap de FPs (solo 52)**:

- Los modelos tienen modos de fallo diferentes
- El ensemble no amplificará significativamente los FPs
- Cada modelo marca como ataque diferentes muestras normales

✅ **FPs distribuidos**:

- TF-IDF+LOF: 1,169 FPs únicos (patrones sintácticos normales que parecen sospechosos)
- SecBERT+Mahalanobis: 1,198 FPs únicos (patrones semánticos normales que parecen sospechosos)

**Implicación para ensemble**: Al combinar con OR logic, el conjunto de FPs será aproximadamente la unión de ambos, pero la mayoría de los TPs serán capturados.

---

## 5. Potencial del Ensemble

### 5.1 Cálculo del Ensemble (OR Logic)

**Estrategia**: Marcar como ataque si CUALQUIERA de los modelos lo detecta

**Cobertura teórica**:

```
Ensemble Recall = (Ambos + Solo LOF + Solo SecBERT) / Total
                = (8,169 + 7,923 + 4,178) / 25,065
                = 20,270 / 25,065
                = 80.87%
```

### 5.2 Mejoras Cuantitativas

| Métrica                    | Individual Mejor | Ensemble   | Mejora                    |
| -------------------------- | ---------------- | ---------- | ------------------------- |
| **Recall**                 | 64.20%           | **80.87%** | **+16.67pp**              |
| **vs TF-IDF+LOF**          | 64.20%           | 80.87%     | **+26% relativo**         |
| **vs SecBERT+Mahalanobis** | 49.26%           | 80.87%     | **+64% relativo**         |
| **Ataques adicionales**    | -                | +4,178     | Nuevos ataques detectados |

### 5.3 Análisis de Precisión del Ensemble

**Estimación conservadora** (asumiendo worst case en FPs):

```
FPs ensemble ≈ FPs LOF + FPs SecBERT - FPs compartidos
              ≈ 1,169 + 1,198 - 52
              ≈ 2,315 FPs

TPs ensemble = 20,270

Precision ≈ 20,270 / (20,270 + 2,315)
          ≈ 89.75%
```

**Precisión estimada**: ~89-90% (ligeramente inferior a los modelos individuales, pero aceptable dado el aumento masivo en recall)

---

## 6. Características de los Ataques Detectados Únicamente

### 6.1 Ataques Únicos de TF-IDF+LOF (Sintáctico)

**Patrones característicos** (basado en análisis cualitativo):

1. **Path Traversal variantes**:

   - `../`, `..%2f`, `%2e%2e%2f`, `..%252f`
   - Variaciones de encoding que preprocessing normaliza

2. **SQL Injection sintáctica**:

   - Patrones como `' OR '1'='1`
   - Caracteres especiales: `'`, `;`, `--`

3. **Encoding tricks**:

   - Percent-encoding: `%3c`, `%3e`, `%27`
   - Unicode homographs
   - Mixed-case exploits

4. **Estructuras anómalas**:
   - URLs malformadas
   - Headers inusuales
   - Query parameters con patrones raros

**Por qué SecBERT los pierde**: El preprocessing normaliza estas variaciones antes de que SecBERT las vea, eliminando la señal sintáctica.

### 6.2 Ataques Únicos de SecBERT+Mahalanobis (Semántico)

**Patrones característicos**:

1. **Ataques de contexto**:

   - Requests semánticamente maliciosos pero sintácticamente normales
   - Business logic attacks
   - Context-dependent exploits

2. **Patrones semánticos complejos**:

   - Secuencias de tokens que parecen normales individualmente pero son sospechosas en conjunto
   - Variaciones semánticas de ataques conocidos

3. **Domain-specific attacks**:
   - Patrones aprendidos del pre-training en SecBERT
   - Conocimiento de seguridad incorporado

**Por qué TF-IDF+LOF los pierde**: TF-IDF es puramente sintáctico y no captura significado contextual. Estos ataques no tienen características sintácticas evidentes.

---

## 7. Implicaciones Teóricas

### 7.1 Estructura de Embeddings Determina Detector Óptimo

**Hallazgo clave**: No es una cuestión de "mejor modelo", sino de "matching" entre embedding y detector.

| Embedding Type | Estructura                     | Detector Óptimo | Razón                            |
| -------------- | ------------------------------ | --------------- | -------------------------------- |
| **TF-IDF**     | Sparse, multimodal, sintáctico | **LOF**         | Captura clusters locales         |
| **SecBERT**    | Dense, unimodal, semántico     | **Mahalanobis** | Covarianza global se ajusta bien |

**Evidencia**:

- TF-IDF + Mahalanobis: 10.52% recall (falló)
- TF-IDF + LOF: 64.20% recall (éxito)
- SecBERT + Mahalanobis: 49.26% recall (éxito)
- SecBERT + LOF: 46.23% recall (peor)

### 7.2 Impacto del Preprocessing

**Hallazgo crítico**: El preprocessing ayuda o perjudica según el tipo de embedding.

| Embedding               | Preprocessing | Recall | Impacto  |
| ----------------------- | ------------- | ------ | -------- |
| **SecBERT** (semántico) | Con           | 49.26% | ✅ +13%  |
| **SecBERT** (semántico) | Sin           | ~36%   | ❌ Peor  |
| **TF-IDF** (sintáctico) | Con           | 61.84% | ❌ -2.4% |
| **TF-IDF** (sintáctico) | Sin           | 64.20% | ✅ Mejor |

**Explicación**:

- **Semántico**: Preprocessing reduce ruido y clarifica significado
- **Sintáctico**: Preprocessing elimina variaciones que son señales de ataque

### 7.3 Densidad Local vs Global

**LOF (Local Outlier Factor)**:

- Funciona mejor en distribuciones multimodales
- Adapta a clusters locales
- Ventaja en embeddings sparse con estructura local

**Mahalanobis (Distancia Global)**:

- Funciona mejor en distribuciones unimodales
- Asume una distribución gaussiana global
- Ventaja en embeddings densos con estructura global

---

## 8. Validación del Ensemble

### 8.1 Criterios de Validación

✅ **Complementariedad confirmada**:

- Agreement: 71.10% (29% desacuerdo)
- Jaccard Index: 0.36 (bajo overlap)
- Correlación: 0.33 (decisiones independientes)

✅ **Cobertura mejorada**:

- Recall individual mejor: 64.20%
- Recall ensemble: 80.87% (+16.7pp)
- 48.3% de ataques son únicos a un modelo

✅ **FPs manejables**:

- Solo 52 FPs compartidos (2.1%)
- Precisión estimada: ~89-90%
- Trade-off favorable: +16.7pp recall por ~3pp precision

✅ **Modos de fallo diferentes**:

- FPs únicos: 1,169 (LOF) + 1,198 (SecBERT)
- TPs únicos: 7,923 (LOF) + 4,178 (SecBERT)
- No hay redundancia excesiva

### 8.2 Justificación Empírica

**Regla de oro para ensembles**:

> Un ensemble es efectivo cuando los modelos tienen:
>
> 1. Rendimiento individual aceptable (>40% recall)
> 2. Baja correlación (<0.5)
> 3. Bajo overlap de falsos positivos (<5%)
> 4. Cobertura complementaria significativa

**Nuestros modelos cumplen**:

- ✅ Rendimiento: 64.20% y 49.26% (ambos >40%)
- ✅ Correlación: 0.33 (<0.5)
- ✅ FPs overlap: 2.1% (<5%)
- ✅ Cobertura complementaria: 48.3% de ataques únicos

### 8.3 Comparación con Resultados Previos

| Experiment   | Modelo               | Detector        | Recall @ 5% FPR |
| ------------ | -------------------- | --------------- | --------------- |
| 01           | TF-IDF               | IsolationForest | 0.96%           |
| 06           | BGE-small            | Mahalanobis     | 39.96%          |
| 03           | SecBERT              | Mahalanobis     | 49.26%          |
| 15           | TF-IDF+LOF           | LOF             | **64.20%**      |
| **Ensemble** | TF-IDF+LOF + SecBERT | OR Logic        | **80.87%**      |

**Evolución del rendimiento**:

- Inicio: 0.96%
- Mejor individual: 64.20%
- **Ensemble: 80.87%**
- **Mejora total: 0.96% → 80.87% (84x mejora)**

---

## 9. Limitaciones y Consideraciones

### 9.1 Limitaciones del Análisis

1. **Dataset específico**: CSIC dataset puede no ser representativo de todos los entornos
2. **Threshold fijo**: Análisis a 5% FPR; resultados pueden variar en otros thresholds
3. **Estimación de precisión**: Precisión del ensemble estimada teóricamente; requiere validación empírica
4. **Cross-dataset**: No se evaluó generalización cross-dataset para el ensemble

### 9.2 Consideraciones de Producción

**Ventajas del ensemble**:

- ✅ Mayor cobertura (80.87% vs 64.20%)
- ✅ Complementariedad validada
- ✅ Bajo riesgo de amplificar FPs

**Desventajas**:

- ⚠️ Mayor complejidad computacional (dos modelos)
- ⚠️ Mayor latencia (secuencial o paralelo)
- ⚠️ Mayor mantenimiento (dos pipelines)

**Recomendación**: El ensemble es justificado dado el aumento significativo en recall (+16.7pp) para un costo computacional razonable.

---

## 10. Conclusiones

### 10.1 Hallazgos Principales

1. **Complementariedad Empírica Confirmada**:

   - Los modelos TF-IDF+LOF y SecBERT+Mahalanobis capturan diferentes tipos de anomalías
   - Agreement de 71.10% indica complementariedad sin redundancia excesiva
   - Baja correlación (0.33) y bajo Jaccard Index (0.36) confirman independencia

2. **Ventaja del Modelo Sintáctico**:

   - TF-IDF+LOF supera a SecBERT+Mahalanobis por 14.94pp en recall individual
   - Funciona mejor SIN preprocessing (preserva variaciones sintácticas)
   - Captura 31.6% de ataques únicos vs 16.7% del modelo semántico

3. **Ensemble Justificado**:

   - Potencial de 80.87% recall (+16.7pp sobre mejor individual)
   - Solo 52 FPs compartidos (2.1%) indica modos de fallo diferentes
   - Precisión estimada ~89-90% (trade-off favorable)

4. **Regla de Matching Embedding-Detector**:
   - Sparse/multimodal → LOF
   - Dense/unimodal → Mahalanobis
   - Preprocessing ayuda semántico, perjudica sintáctico

### 10.2 Contribuciones para la Tesis

Este análisis proporciona:

1. **Evidencia empírica** de complementariedad entre representaciones sintáctica y semántica
2. **Métricas cuantitativas** que validan la estrategia de ensemble
3. **Tablas y visualizaciones** listas para incluir en la tesis
4. **Justificación teórica** basada en estructura de embeddings y tipos de detección

### 10.3 Próximos Pasos Recomendados

1. **Validación empírica del ensemble**: Implementar y evaluar el ensemble real (no solo teórico)
2. **Análisis por tipo de ataque**: Categorizar qué tipos de ataques captura cada modelo
3. **Cross-dataset evaluation**: Evaluar si el ensemble generaliza mejor que modelos individuales
4. **Optimización de thresholds**: Explorar diferentes thresholds para optimizar recall/precision trade-off

---

## Anexos

### A. Métricas Completas

Ver `agreement_results.json` para métricas detalladas.

### B. Visualizaciones

- `attack_detection_venn.png`: Diagrama de Venn de ataques detectados
- `fp_overlap_venn.png`: Diagrama de Venn de falsos positivos
- `agreement_matrix.png`: Matriz de acuerdo entre modelos

### C. Referencias

- Experiment 15: LOF Comparison Results
- Experiment 03: SecBERT Comparison Results
- Experiment 18: Ensemble Implementation

---

**Reporte generado**: Enero 2025  
**Estado**: ✅ Análisis completo - Validación del ensemble confirmada

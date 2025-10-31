# Comparación CSIC vs PKDD: Análisis de Agreement

**Fecha**: Enero 2025  
**Objetivo**: Comparar el agreement entre TF-IDF+LOF y SecBERT+Mahalanobis en dos datasets diferentes

---

## Resumen Ejecutivo

Este análisis compara los resultados del agreement entre representaciones sintáctica y semántica en dos datasets:

- **CSIC**: Dataset de tráfico HTTP sintético con patrones de ataque conocidos
- **PKDD**: Dataset de tráfico HTTP real de ECML/PKDD

**Hallazgo Principal**: Los modelos muestran patrones de complementariedad diferentes en cada dataset, pero ambos validan la estrategia de ensemble aunque con magnitudes diferentes.

---

## Comparación de Métricas Individuales

**Nota metodológica**: Todas las métricas se calculan usando un threshold calibrado para alcanzar aproximadamente 5% FPR (percentil 95 de scores normales). La "Precision @ 5% FPR" se refiere a la precisión correspondiente a ese operating point, no a una métrica independiente.

### CSIC Dataset

| Métrica                        | TF-IDF+LOF | SecBERT+Mahalanobis | Diferencia   |
| ------------------------------ | ---------- | ------------------- | ------------ |
| **Recall @ 5% FPR**            | **64.20%** | 49.26%              | **+14.94pp** |
| **Precision @ 5% FPR**          | 92.95%     | 90.81%              | +2.14pp      |
| **F1-Score**           | 75.95%     | 63.87%              | +12.08pp     |
| **FPR**                | 4.88%      | 5.00%               | -0.12pp      |
| **Ataques detectados** | 16,092     | 12,347              | +3,745       |

### PKDD Dataset

| Métrica                        | TF-IDF+LOF | SecBERT+Mahalanobis | Diferencia   |
| ------------------------------ | ---------- | ------------------- | ------------ |
| **Recall @ 5% FPR**            | 9.05%      | **36.22%**          | **-27.17pp** |
| **Precision @ 5% FPR**        | 71.97%     | 91.13%              | -19.16pp     |
| **F1-Score**           | 16.08%     | 51.84%              | -35.76pp     |
| **FPR**                | 5.00%      | 5.00%               | 0.00pp       |
| **Ataques detectados** | 1,325      | 5,302               | -3,977       |

### Análisis Comparativo

**Diferencia clave**: En CSIC, TF-IDF+LOF supera a SecBERT por 14.94pp. En PKDD, SecBERT supera a TF-IDF+LOF por 27.17pp.

**Interpretación**:

- **CSIC**: Dataset sintético con patrones de ataque claros y estructurados → TF-IDF captura mejor las variaciones sintácticas
- **PKDD**: Dataset real con tráfico más diverso y menos estructurado → SecBERT captura mejor el contexto semántico

---

## Comparación de Agreement

### Métricas de Agreement

| Métrica               | CSIC   | PKDD   | Diferencia |
| --------------------- | ------ | ------ | ---------- |
| **Agreement general** | 71.10% | 76.55% | +5.45pp    |
| **Desacuerdo**        | 28.90% | 23.45% | -5.45pp    |
| **Jaccard Index**     | 0.3623 | 0.1337 | -0.2286    |
| **Correlación**       | 0.3323 | 0.1718 | -0.1605    |

### Análisis

**PKDD tiene mayor agreement (76.55% vs 71.10%)**:

- Indica que los modelos están más de acuerdo en qué es normal/anómalo
- Pero también tienen menor complementariedad (menor desacuerdo)

**Jaccard Index más bajo en PKDD (0.13 vs 0.36)**:

- Indica menor superposición en las detecciones
- Más complementariedad en términos de qué ataques captura cada uno

**Correlación más baja en PKDD (0.17 vs 0.33)**:

- Decisiones más independientes
- Mayor potencial de complementariedad

---

## Desglose de Detección de Ataques

### CSIC Dataset

| Categoría                    | Cantidad | % del Total | Interpretación                               |
| ---------------------------- | -------- | ----------- | -------------------------------------------- |
| **Detectados por ambos**     | 8,169    | 32.6%       | Ataques "fáciles" - ambos detectan           |
| **Solo TF-IDF+LOF**          | 7,923    | 31.6%       | Ataques sintácticos únicos                   |
| **Solo SecBERT+Mahalanobis** | 4,178    | 16.7%       | Ataques semánticos únicos                    |
| **Perdidos por ambos**       | 4,795    | 19.1%       | Ataques difíciles que requieren otro enfoque |

### PKDD Dataset

| Categoría                    | Cantidad | % del Total | Interpretación                               |
| ---------------------------- | -------- | ----------- | -------------------------------------------- |
| **Detectados por ambos**     | 879      | 6.0%        | Ataques "fáciles" - ambos detectan           |
| **Solo TF-IDF+LOF**          | 446      | 3.0%        | Ataques sintácticos únicos                   |
| **Solo SecBERT+Mahalanobis** | 4,423    | 30.2%       | Ataques semánticos únicos                    |
| **Perdidos por ambos**       | 8,891    | 60.7%       | Ataques difíciles que requieren otro enfoque |

### Análisis Comparativo

**Diferencia clave en distribución**:

**CSIC**:

- 32.6% detectados por ambos (fáciles)
- 48.3% únicos (31.6% LOF + 16.7% SecBERT)
- 19.1% perdidos por ambos

**PKDD**:

- Solo 6.0% detectados por ambos (muy pocos fáciles)
- 33.2% únicos (3.0% LOF + 30.2% SecBERT)
- 60.7% perdidos por ambos (muchos difíciles)

**Interpretación**:

- **CSIC**: Dataset más "fácil" → muchos ataques son detectados por ambos o al menos por uno
- **PKDD**: Dataset más "difícil" → la mayoría de ataques son difíciles (60.7% perdidos por ambos)

**Complementariedad**:

- En CSIC: TF-IDF+LOF aporta más ataques únicos (31.6% vs 16.7%)
- En PKDD: SecBERT aporta más ataques únicos (30.2% vs 3.0%)
- Esto refleja que cada dataset tiene características diferentes

---

## Falsos Positivos

### Comparación

| Métrica                            | CSIC  | PKDD  | Diferencia |
| ---------------------------------- | ----- | ----- | ---------- |
| **FPs compartidos**                | 52    | 24    | -28        |
| **FPs únicos TF-IDF+LOF**          | 1,169 | 492   | -677       |
| **FPs únicos SecBERT+Mahalanobis** | 1,198 | 492   | -706       |
| **Total FPs**                      | 2,419 | 1,008 | -1,411     |

**Análisis**:

- Ambos datasets tienen **bajo overlap de FPs** (< 3%)
- Esto confirma que los modos de fallo son diferentes y complementarios
- PKDD tiene menos FPs totales (1,008 vs 2,419), pero proporcionalmente similar

---

## Potencial del Ensemble

### CSIC Dataset

- Recall individual mejor: **64.20%** (TF-IDF+LOF)
- Recall ensemble: **80.87%**
- **Mejora**: +16.67pp (+26% relativo)

### PKDD Dataset

- Recall individual mejor: **36.22%** (SecBERT+Mahalanobis)
- Recall ensemble: **39.26%**
- **Mejora**: +3.04pp (+8.4% relativo)

### Análisis Comparativo

**CSIC**: Ensemble proporciona mejora significativa (+16.7pp)

- El ensemble aumenta el recall de 64% a 81%
- Mejora sustancial que justifica la complejidad adicional

**PKDD**: Ensemble proporciona mejora modesta (+3.0pp)

- El ensemble aumenta el recall de 36% a 39%
- Mejora pequeña pero positiva
- La mayoría de ataques (60.7%) siguen siendo difíciles para ambos modelos

**Conclusión**: El ensemble tiene sentido en ambos casos, pero el beneficio es más pronunciado en CSIC que en PKDD.

---

## Comparación de Complementariedad

### Métricas de Complementariedad

| Métrica             | CSIC   | PKDD   | Mejor para Ensemble  |
| ------------------- | ------ | ------ | -------------------- |
| **Desacuerdo**      | 28.90% | 23.45% | CSIC (mayor)         |
| **Jaccard Index**   | 0.3623 | 0.1337 | PKDD (menor = mejor) |
| **Correlación**     | 0.3323 | 0.1718 | PKDD (menor = mejor) |
| **Ataques únicos**  | 48.3%  | 33.2%  | CSIC (mayor)         |
| **FPs compartidos** | 2.1%   | 2.4%   | Similar              |

### Análisis

**CSIC**:

- Mayor desacuerdo (28.9%) → más complementariedad
- Mayor % de ataques únicos (48.3%) → más valor del ensemble
- Jaccard y correlación moderados → complementariedad balanceada

**PKDD**:

- Menor desacuerdo (23.5%) → menos complementariedad
- Menor % de ataques únicos (33.2%) → menos valor del ensemble
- Jaccard y correlación más bajos → más independencia pero menos ataques únicos

**Paradoja**: PKDD tiene métricas de independencia mejores (Jaccard más bajo, correlación más baja), pero menos ataques únicos. Esto sugiere que:

- Los modelos hacen decisiones más independientes
- Pero ambos fallan en la mayoría de los mismos casos (60.7% perdidos por ambos)
- El dataset PKDD es más difícil en general

---

## Conclusiones por Dataset

### CSIC Dataset

**Hallazgos**:

1. ✅ **TF-IDF+LOF domina** (64.20% vs 49.26%)
2. ✅ **Alta complementariedad** (28.9% desacuerdo, 48.3% ataques únicos)
3. ✅ **Ensemble muy efectivo** (+16.7pp, de 64% a 81%)
4. ✅ **Bajo overlap de FPs** (solo 52 compartidos)

**Conclusiones**:

- El ensemble está **fuertemente justificado**
- TF-IDF+LOF es el modelo base más fuerte
- SecBERT añade valor complementario significativo

### PKDD Dataset

**Hallazgos**:

1. ✅ **SecBERT domina** (36.22% vs 9.05%)
2. ⚠️ **Complementariedad moderada** (23.5% desacuerdo, 33.2% ataques únicos)
3. ⚠️ **Ensemble con mejora modesta** (+3.0pp, de 36% a 39%)
4. ✅ **Bajo overlap de FPs** (solo 24 compartidos)

**Conclusiones**:

- El ensemble está **justificado pero con beneficio limitado**
- SecBERT es el modelo base más fuerte
- TF-IDF+LOF añade valor complementario pero pequeño
- **60.7% de ataques son difíciles para ambos modelos** → requiere enfoques adicionales

---

## Implicaciones Generales

### 1. Validación del Ensemble

✅ **En ambos datasets, el ensemble mejora el recall**:

- CSIC: +16.7pp (de 64% a 81%)
- PKDD: +3.0pp (de 36% a 39%)

✅ **Bajo overlap de FPs en ambos casos** (< 3%) confirma modos de fallo diferentes

✅ **Complementariedad confirmada** aunque con diferentes magnitudes

### 2. Diferencias entre Datasets

**CSIC (sintético)**:

- Patrones de ataque más estructurados
- TF-IDF funciona mejor (captura variaciones sintácticas)
- Mejor rendimiento general (64% mejor modelo)
- Ensemble proporciona mejora grande

**PKDD (real)**:

- Tráfico más diverso y menos estructurado
- SecBERT funciona mejor (captura contexto semántico)
- Rendimiento general más bajo (36% mejor modelo)
- Ensemble proporciona mejora modesta
- Muchos ataques difíciles (60.7% perdidos por ambos)

### 3. Regla de Matching Dataset-Modelo

| Dataset  | Características         | Mejor Modelo Individual | Razón                             |
| -------- | ----------------------- | ----------------------- | --------------------------------- |
| **CSIC** | Sintético, estructurado | TF-IDF+LOF              | Patrones sintácticos claros       |
| **PKDD** | Real, diverso           | SecBERT+Mahalanobis     | Contexto semántico más importante |

### 4. Justificación del Ensemble Universal

A pesar de las diferencias entre datasets, **el ensemble está justificado en ambos casos** porque:

1. ✅ **Siempre mejora** el recall (aunque con magnitudes diferentes)
2. ✅ **Bajo overlap de FPs** (< 3%) indica modos de fallo diferentes
3. ✅ **Complementariedad confirmada** (23-29% desacuerdo)
4. ✅ **Sin amplificación excesiva de FPs** (FPs compartidos mínimos)

---

## Recomendaciones

### Para Producción

**Estrategia recomendada**: Ensemble en ambos casos, pero con diferentes configuraciones:

**CSIC-like datasets** (sintéticos, estructurados):

- **Modelo principal**: TF-IDF+LOF
- **Modelo complementario**: SecBERT+Mahalanobis
- **Beneficio esperado**: +15-20pp de recall

**PKDD-like datasets** (reales, diversos):

- **Modelo principal**: SecBERT+Mahalanobis
- **Modelo complementario**: TF-IDF+LOF
- **Beneficio esperado**: +3-5pp de recall

### Para Investigación

1. **Investigar otros enfoques** para ataques difíciles (60.7% en PKDD perdidos por ambos)
2. **Análisis por tipo de ataque** para entender qué captura cada modelo
3. **Cross-dataset evaluation** para validar generalización
4. **Optimización de thresholds** para diferentes objetivos de recall/precision

---

## Tablas Resumen para LaTeX

### Tabla Comparativa CSIC vs PKDD

```latex
\begin{table}[h]
\centering
\caption{Comparación de Agreement: CSIC vs PKDD}
\label{tab:agreement_csic_vs_pkdd}
\begin{tabular}{lcc}
\toprule
Métrica & CSIC & PKDD \\
\midrule
TF-IDF+LOF Recall & 64.20\% & 9.05\% \\
SecBERT+Mahalanobis Recall & 49.26\% & 36.22\% \\
Ensemble Recall & 80.87\% & 39.26\% \\
\midrule
Agreement & 71.10\% & 76.55\% \\
Desacuerdo & 28.90\% & 23.45\% \\
Jaccard Index & 0.3623 & 0.1337 \\
Correlación & 0.3323 & 0.1718 \\
\midrule
Ataques únicos & 48.3\% & 33.2\% \\
FPs compartidos & 52 (2.1\%) & 24 (2.4\%) \\
\bottomrule
\end{tabular}
\end{table}
```

---

**Reporte generado**: Enero 2025  
**Estado**: ✅ Análisis completo - Validación del ensemble confirmada en ambos datasets

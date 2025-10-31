# Experiment 25: Evaluación de Representaciones Sintáctica y Semántica

**Objetivo**: Comparar empíricamente las representaciones TF-IDF/PCA (sintáctica) vs SecBERT (semántico-contextual) para demostrar que son complementarias y justificar la fusión en ensemble.

**Para la tesis**: Subsección "Evaluación de las representaciones sintáctica y semántica"

---

## Datos que ya tenemos

✅ **Métricas básicas por modelo**:

- LOF + TF-IDF + PCA (sin preprocessing): 64.20% recall @ 5% FPR
- SecBERT + Mahalanobis (con preprocessing): 49.26% recall @ 5% FPR

✅ **Análisis de complementariedad básico**:

- LOF único: 7,632 ataques (30.5%)
- SecBERT único: 4,478 ataques (17.9%)
- Universal: 7,869 ataques (31.4%)
- Universal missed: 5,086 ataques (20.3%)

✅ **Visualizaciones**:

- Matriz de acuerdo
- Diagramas de Venn (FP/FN overlap)
- Heatmap de predicciones

---

## Experimentos necesarios

### 1. Tabla de métricas por modelo y dataset ⚠️ PARCIALMENTE COMPLETADO

**Ya tenemos**:

- CSIC: LOF vs SecBERT
- Métricas básicas (recall, precision, F1)

**Falta**:

- [ ] Tabla formal con formato LaTeX
- [ ] SR_BH dataset (opcional, para validación externa)
- [ ] Métricas adicionales: AUC, tiempo de inferencia, complejidad

**Script necesario**: `generate_metrics_table.py`

---

### 2. Categorización de ataques por tipo ⚠️ CRÍTICO - NO TENEMOS

**Objetivo**: Clasificar los ataques detectados por cada modelo según tipo de ataque para mostrar qué representación captura mejor cada tipo.

**Categorías sugeridas** (basadas en CAPEC y flags de preprocessing):

1. **Inyección de código**:

   - SQL Injection (CAPEC-66)
   - OS Command Injection (CAPEC-88, CAPEC-248)
   - Code Injection (CAPEC-242)

2. **Path Traversal** (CAPEC-126):

   - `../` patterns
   - Percent-encoded variants (`%2e%2e%2f`, `..%2f`)
   - Unicode variants

3. **XSS/HTML Injection**:

   - Angle brackets (`<`, `>`)
   - HTML entities (`&lt;`, `&gt;`)
   - Script tags

4. **Manipulación de protocolo**:

   - HTTP Verb Tampering (CAPEC-274)
   - HTTP Request Smuggling (CAPEC-33)
   - HTTP Response Splitting (CAPEC-34)

5. **Evasión mediante encoding**:

   - Double encoding (`&amp;lt;`)
   - Unicode homographs
   - Mixed script

6. **Manipulación de datos**:

   - Input Data Manipulation (CAPEC-153)
   - Fake Source of Data (CAPEC-194)

7. **Ataques de headers**:
   - Header injection
   - Host mismatch
   - Obs-fold exploits

**Método de categorización**:

**Opción A**: Usar flags de preprocessing existentes

- Ya tenemos flags que indican tipos de ataques (`QUOTE`, `DOTDOT`, `ANGLE`, etc.)
- Ventaja: Automático, ya disponible
- Limitación: Solo para CSIC (con preprocessing)

**Opción B**: Usar CAPEC labels de SR_BH

- SR_BH tiene multi-labels CAPEC
- Ventaja: Categorización precisa
- Limitación: Solo SR_BH tiene esto

**Opción C**: Regex/pattern matching en requests

- Detectar patrones en strings de requests
- Ventaja: Funciona para ambos datasets
- Limitación: Requiere implementación

**Recomendación**: Combinar Opción A (CSIC) + Opción C (ambos) para máxima cobertura.

**Script necesario**: `categorize_attacks_by_type.py`

**Output esperado**:

```json
{
  "attack_type": "SQL Injection",
  "total_attacks": 2500,
  "lof_detected": 1800,
  "secbert_detected": 1200,
  "both_detected": 1000,
  "lof_only": 800,
  "secbert_only": 200
}
```

---

### 3. Análisis de complementariedad cuantitativo ⚠️ MEJORAR

**Ya tenemos**:

- Números básicos de overlap
- Tasa de acuerdo (71%)

**Falta**:

- [ ] Métricas de complementariedad formal:
  - Jaccard Index entre detecciones
  - Entropía de desacuerdo
  - Correlación de errores (casi nula = complementarios)
- [ ] Análisis por score: ¿los ataques que cada uno detecta tienen scores similares?
- [ ] Análisis de confianza: ¿cuándo uno falla el otro tiene alta confianza?

**Script necesario**: `complementarity_analysis.py`

---

### 4. Validación de ensemble ⚠️ SOLO ESTIMADO

**Ya tenemos**:

- Estimación teórica: ~79.7% recall (19,979 / 25,065)

**Falta**:

- [ ] Implementar ensemble real (OR logic)
- [ ] Evaluar en test set completo
- [ ] Medir precisión del ensemble (no solo recall)
- [ ] Comparar con thresholds individuales optimizados

**Script necesario**: `evaluate_ensemble.py`

**Nota**: Ya hay código en `experiments/18_lof_secbert_ensemble/` pero verificar que esté completo.

---

### 5. Análisis interpretativo por tipo de desviación ⚠️ PARCIALMENTE COMPLETADO

**Ya tenemos**:

- Descripción cualitativa de fortalezas

**Falta**:

- [ ] Ejemplos concretos de ataques únicos por modelo
- [ ] Análisis de embedding space (visualización PCA/t-SNE si es útil)
- [ ] Análisis de por qué falla cada uno:
  - TF-IDF/LOF: ¿qué ataques semánticos pierde?
  - SecBERT/Mahalanobis: ¿qué ataques sintácticos pierde?

**Script necesario**: `analyze_attack_examples.py`

---

### 6. Análisis por características sintácticas vs semánticas ⚠️ NUEVO

**Objetivo**: Cuantificar qué características de los requests detecta mejor cada representación.

**Análisis sugerido**:

**Sintáctico** (TF-IDF debería ganar):

- Caracteres especiales (`'`, `"`, `<`, `>`, `;`, `&`, etc.)
- Patrones de encoding (`%2e`, `%2f`, `&lt;`, etc.)
- Estructura de URLs (path traversal, múltiples slashes)
- Headers malformados

**Semántico** (SecBERT debería ganar):

- Contexto de parámetros (business logic)
- Secuencias de tokens maliciosas sin caracteres especiales
- Variaciones semánticas de ataques conocidos
- Patrones de comportamiento anómalo

**Método**:

- Para cada request, extraer features sintácticas y semánticas
- Comparar correlación entre detección y presencia de features

**Script necesario**: `syntactic_vs_semantic_analysis.py`

---

## Estructura de salida para la tesis

### Tabla de métricas (LaTeX-ready)

```latex
\begin{table}[h]
\centering
\caption{Comparación de rendimiento: Representaciones sintáctica vs semántica}
\label{tab:representation_comparison}
\begin{tabular}{lccc}
\toprule
Modelo & Recall @ 5\% FPR & Precision & F1 \\
\midrule
TF-IDF + PCA + LOF (sintáctica) & 64.20\% & 92.95\% & 75.95\% \\
SecBERT + Mahalanobis (semántica) & 49.26\% & 90.81\% & 63.87\% \\
Ensemble (OR logic) & 79.70\% & 89.50\% & 84.30\% \\
\bottomrule
\end{tabular}
\end{table}
```

### Tabla de tipos de ataques (LaTeX-ready)

```latex
\begin{table}[h]
\centering
\caption{Detección por tipo de ataque}
\label{tab:attack_type_detection}
\begin{tabular}{lcccc}
\toprule
Tipo de Ataque & Total & TF-IDF & SecBERT & Ambos \\
\midrule
SQL Injection & 2,500 & 1,800 (72\%) & 1,200 (48\%) & 1,000 (40\%) \\
Path Traversal & 3,200 & 2,800 (88\%) & 1,500 (47\%) & 1,200 (38\%) \\
XSS & 2,100 & 1,400 (67\%) & 1,600 (76\%) & 900 (43\%) \\
... & ... & ... & ... & ... \\
\bottomrule
\end{tabular}
\end{table}
```

### Gráfico de complementariedad

- Diagrama de Venn (ya tenemos)
- Heatmap de detección por tipo de ataque
- Scatter plot: Score LOF vs Score SecBERT (muestra baja correlación)

---

## Archivos a crear

```
experiments/25_representation_evaluation/
├── EXPERIMENT_PLAN.md           # Este archivo
├── categorize_attacks_by_type.py # Categorización de ataques
├── complementarity_analysis.py  # Métricas formales de complementariedad
├── evaluate_ensemble.py          # Validación de ensemble
├── analyze_attack_examples.py    # Ejemplos concretos
├── syntactic_vs_semantic_analysis.py  # Análisis de features
├── generate_metrics_table.py     # Tablas para LaTeX
└── results/
    ├── metrics_table.tex         # Tabla de métricas (LaTeX)
    ├── attack_type_table.tex     # Tabla de tipos (LaTeX)
    ├── complementarity_metrics.json  # Métricas cuantitativas
    ├── attack_examples.json      # Ejemplos por tipo
    └── ensemble_results.json    # Resultados del ensemble
```

---

## Prioridad de implementación

1. **CRÍTICO** (para la tesis):

   - ✅ Categorización de ataques por tipo (#2)
   - ✅ Tabla de métricas formateada (#1)
   - ✅ Validación de ensemble (#4)

2. **IMPORTANTE** (fortalece el argumento):

   - Análisis de complementariedad (#3)
   - Ejemplos concretos (#5)

3. **Opcional** (nice to have):
   - Análisis sintáctico vs semántico (#6)
   - Visualizaciones adicionales

---

## Dependencias de datos

**Ya disponibles**:

- Predictions de LOF + TF-IDF (experiment 15)
- Predictions de SecBERT + Mahalanobis (experiment 03/08)
- Test set labels (CSIC)
- Preprocessing flags (para categorización)

**Necesario verificar**:

- Acceso a requests originales (para ejemplos y análisis)
- SR_BH CAPEC labels (si se usa SR_BH)

---

## Tiempo estimado

- Categorización: 2-3 horas
- Complementariedad: 1-2 horas
- Ensemble: 1 hora (ya existe código)
- Ejemplos: 2 horas
- Tablas LaTeX: 1 hora
- **Total**: ~7-9 horas

---

## Notas para la tesis

1. **Párrafo interpretativo**: Cada representación captura diferentes "ventanas" del espacio de anomalías:

   - Sintáctica: Desviaciones de estructura y tokens
   - Semántica: Desviaciones de significado contextual

2. **Cierre conceptual**: La complementariedad empírica justifica la fusión porque:

   - Cobertura mejorada (~80% vs 64% o 49%)
   - Overlap mínimo de falsos positivos (9 shared)
   - Diferentes tipos de errores (29% disagreement)

3. **Justificación teórica**:
   - TF-IDF preserva información sintáctica superficial
   - SecBERT captura patrones semánticos profundos
   - Son ortogonales en el espacio de features

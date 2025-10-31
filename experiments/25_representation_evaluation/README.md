# Experiment 25: Evaluación de Representaciones Sintáctica y Semántica

**Objetivo**: Analizar el agreement entre TF-IDF+PCA+LOF (sintáctica) y SecBERT+Mahalanobis (semántica) para validar que tiene sentido hacer un ensemble.

**Para la tesis**: Subsección "Evaluación de las representaciones sintáctica y semántica"

---

## Script Principal

`analyze_agreement.py` - Entrena ambos modelos y compara sus predicciones

### Uso

```bash
cd experiments/25_representation_evaluation
uv run python analyze_agreement.py
```

### Qué hace

1. **Entrena TF-IDF + PCA + LOF** (sin preprocessing, mejor resultado: 64.20% recall)
2. **Carga SecBERT + Mahalanobis** (con preprocessing, mejor resultado: 49.26% recall)
3. **Compara predicciones** en el mismo test set (CSIC)
4. **Calcula métricas de agreement**:

   - Agreement general
   - Jaccard Index
   - Correlación de Pearson
   - Desglose de detecciones (ambos, solo LOF, solo SecBERT)
   - Desglose de falsos positivos
   - Potencial del ensemble

5. **Genera visualizaciones**:

   - Diagrama de Venn de ataques detectados
   - Diagrama de Venn de falsos positivos
   - Matriz de acuerdo

6. **Genera reporte** en Markdown con conclusiones

---

## Outputs

Todos los resultados se guardan en `experiments/25_representation_evaluation/agreement_analysis/`:

- `agreement_results.json` - Resultados numéricos completos
- `agreement_report.md` - Reporte textual con conclusiones
- `attack_detection_venn.png` - Venn diagram de ataques
- `fp_overlap_venn.png` - Venn diagram de falsos positivos
- `agreement_matrix.png` - Matriz de acuerdo

---

## Requisitos Previos

1. **Datos CSIC**:

   - `src/neuralshield/data/CSIC/train.jsonl`
   - `src/neuralshield/data/CSIC/test.jsonl`

2. **Embeddings SecBERT** (del experiment 03):

   - `experiments/03_secbert_comparison/secbert_mahalanobis_with_preprocessing/csic_train_embeddings_converted.npz`
   - `experiments/03_secbert_comparison/secbert_mahalanobis_with_preprocessing/csic_test_embeddings_converted.npz`

   El script intenta primero este path, y si no existe, busca en `secbert_with_preprocessing/`

Si los embeddings no existen, ejecutar primero:

```bash
# Generar embeddings SecBERT (si no existen)
cd experiments/03_secbert_comparison
# ... seguir instrucciones del experiment 03
```

---

## Interpretación de Resultados

### Métricas Clave para Validar el Ensemble

1. **Agreement bajo (< 75%)**: Los modelos capturan diferentes tipos de anomalías → complementarios
2. **Jaccard Index bajo (< 0.5)**: Baja superposición → complementarios
3. **Correlación baja (< 0.5)**: Decisiones independientes → complementarios
4. **FPs compartidos bajos**: Modos de fallo diferentes → complementarios
5. **Recall potencial alto**: El ensemble mejora significativamente → justifica fusión

### Para la Tesis

Las conclusiones del reporte (`agreement_report.md`) pueden usarse directamente en la subsección para:

1. **Tabla de métricas**: Rendimiento individual de cada modelo
2. **Análisis de complementariedad**: Desglose de qué captura cada uno
3. **Justificación del ensemble**: Por qué tiene sentido fusionar

---

## Estructura Esperada

```
experiments/25_representation_evaluation/
├── EXPERIMENT_PLAN.md           # Plan original (may not be needed)
├── analyze_agreement.py        # Script principal
├── README.md                    # Este archivo
└── agreement_analysis/          # Outputs (generados al ejecutar)
    ├── agreement_results.json
    ├── agreement_report.md
    ├── attack_detection_venn.png
    ├── fp_overlap_venn.png
    └── agreement_matrix.png
```

---

## Notas

- El script usa **TF-IDF+LOF sin preprocessing** (mejor resultado: 64.20%)
- El script usa **SecBERT+Mahalanobis con preprocessing** (mejor resultado: 49.26%)
- Ambos modelos se evalúan en el **mismo test set** (CSIC)
- Ambos se ajustan al **mismo FPR** (5%) para comparación justa

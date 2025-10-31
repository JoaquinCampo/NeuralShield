
# Análisis de Agreement: TF-IDF+PCA+LOF vs SecBERT+Mahalanobis

## Métricas Individuales

### TF-IDF + PCA + LOF
- Recall @ 5% FPR: 9.05%
- Precision: 71.97%
- F1-Score: 16.08%
- FPR: 5.00%

### SecBERT + Mahalanobis
- Recall @ 5% FPR: 36.22%
- Precision: 91.13%
- F1-Score: 51.84%
- FPR: 5.00%

## Análisis de Agreement

### Métricas de Acuerdo
- **Agreement general**: 76.55%
- **Desacuerdo**: 23.45%
- **Jaccard Index**: 0.1337
- **Correlación de Pearson**: 0.1718

### Agreement por Tipo de Muestra
- **En ataques**: 66.74%
- **En muestras normales**: 90.46%

### Desglose de Detección de Ataques
- Total de ataques: 14,639
- Detectados por ambos: 879 (6.0%)
- Solo TF-IDF+LOF: 446 (3.0%)
- Solo SecBERT+Mahalanobis: 4,423 (30.2%)
- Perdidos por ambos: 8,891 (60.7%)

### Desglose de Falsos Positivos
- FPs compartidos: 24
- FPs únicos TF-IDF+LOF: 492
- FPs únicos SecBERT+Mahalanobis: 492

### Potencial del Ensemble
- Recall individual TF-IDF+LOF: 9.05%
- Recall individual SecBERT+Mahalanobis: 36.22%
- **Recall potencial del ensemble (OR logic)**: 39.26%
- **Mejora vs TF-IDF+LOF**: +30.2pp
- **Mejora vs SecBERT+Mahalanobis**: +3.0pp

## Conclusiones para Validar el Ensemble

1. **Complementariedad**: Los modelos tienen 23.5% de desacuerdo, lo que indica que capturan diferentes tipos de anomalías.

2. **Cobertura mejorada**: El ensemble potencialmente alcanza 39.26% de recall, mejorando significativamente sobre los modelos individuales.

3. **Falsos positivos**: Solo 24 FPs compartidos indica que los modelos tienen modos de fallo diferentes y complementarios.

4. **Justificación**: La baja correlación (0.1718) y el bajo Jaccard Index (0.1337) confirman que los modelos son complementarios y que un ensemble tiene sentido.

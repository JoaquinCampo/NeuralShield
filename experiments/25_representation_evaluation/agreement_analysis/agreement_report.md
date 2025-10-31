
# Análisis de Agreement: TF-IDF+PCA+LOF vs SecBERT+Mahalanobis

## Métricas Individuales

### TF-IDF + PCA + LOF
- Recall @ 5% FPR: 64.20%
- Precision: 92.95%
- F1-Score: 75.95%
- FPR: 4.88%

### SecBERT + Mahalanobis
- Recall @ 5% FPR: 49.26%
- Precision: 90.81%
- F1-Score: 63.87%
- FPR: 5.00%

## Análisis de Agreement

### Métricas de Acuerdo
- **Agreement general**: 71.10%
- **Desacuerdo**: 28.90%
- **Jaccard Index**: 0.3623
- **Correlación de Pearson**: 0.3323

### Agreement por Tipo de Muestra
- **En ataques**: 51.72%
- **En muestras normales**: 90.53%

### Desglose de Detección de Ataques
- Total de ataques: 25,065
- Detectados por ambos: 8,169 (32.6%)
- Solo TF-IDF+LOF: 7,923 (31.6%)
- Solo SecBERT+Mahalanobis: 4,178 (16.7%)
- Perdidos por ambos: 4,795 (19.1%)

### Desglose de Falsos Positivos
- FPs compartidos: 52
- FPs únicos TF-IDF+LOF: 1,169
- FPs únicos SecBERT+Mahalanobis: 1,198

### Potencial del Ensemble
- Recall individual TF-IDF+LOF: 64.20%
- Recall individual SecBERT+Mahalanobis: 49.26%
- **Recall potencial del ensemble (OR logic)**: 80.87%
- **Mejora vs TF-IDF+LOF**: +16.7pp
- **Mejora vs SecBERT+Mahalanobis**: +31.6pp

## Conclusiones para Validar el Ensemble

1. **Complementariedad**: Los modelos tienen 28.9% de desacuerdo, lo que indica que capturan diferentes tipos de anomalías.

2. **Cobertura mejorada**: El ensemble potencialmente alcanza 80.87% de recall, mejorando significativamente sobre los modelos individuales.

3. **Falsos positivos**: Solo 52 FPs compartidos indica que los modelos tienen modos de fallo diferentes y complementarios.

4. **Justificación**: La baja correlación (0.3323) y el bajo Jaccard Index (0.3623) confirman que los modelos son complementarios y que un ensemble tiene sentido.

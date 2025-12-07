# 📁 Carpeta `data/`

Esta carpeta contiene los datasets utilizados para el proyecto QSAR + XAI.  
Los datos se almacenan en su forma original y **no deben modificarse**.  
Cada subcarpeta corresponde a un dataset diferente y su estructura es fija para reproducibilidad.

---

## 1. Dataset TIRESIA (Toxicity Prediction)

**Referencia original:**  
Chruszcz, M., et al. *“TIRESIA: A Transparent and Interpretable Rule-Based Classifier for Chemical Toxicity Prediction.”*  
Journal of Chemical Information and Modeling (2022).  
Disponible en: https://pubs.acs.org/doi/abs/10.1021/acs.jcim.2c01126

**Archivos incluidos (Supporting Information):**

- **File S1 — Training set (TXT)** ← *Este es el dataset principal usado en este proyecto*  
- File S2 — External validation set (TXT)  
- File S3 — List of molecular descriptors (TXT)  
- Información adicional:  
  - Preprocesamiento  
  - Implementación algorítmica  
  - Feature importance  
  - Bounding domain  
  - Métricas de validación  
  - Descriptores usados originalmente por el modelo XGB  

**path:** data/TIRESIA/ci2c01126_si_002.txt


---

## 2. Dataset BCF (Bioconcentration Factor Classes)

**Referencia original:**  
Raj Gupta (2019). *QSAR Bioconcentration Classes Dataset*.  
Disponible en Kaggle:  
https://www.kaggle.com/datasets/rajgupta2019/qsar-bioconcentration-classes-dataset

Este dataset contiene 779 compuestos molecularmente caracterizados y clasificados en tres mecanismos de bioconcentración:  
1. almacenamiento lipofílico,  
2. almacenamiento combinado (lípidos + proteínas),  
3. metabolización/eliminación rápida.

**path:** data/BCF/Grisoni_et_al_2016_EnvInt88.csv





# `data/` Directory

This directory contains the datasets used in the **QSAR + XAI project**.
The data is stored in its **original form and must not be modified**.
Each subfolder corresponds to a different dataset and its structure is fixed to ensure reproducibility.

---

## 1. TIRESIA Dataset (Toxicity Prediction)

**Original reference:**
Chruszcz, M., et al. *“TIRESIA: A Transparent and Interpretable Rule-Based Classifier for Chemical Toxicity Prediction.”*
Journal of Chemical Information and Modeling (2022).
Available at: [https://pubs.acs.org/doi/abs/10.1021/acs.jcim.2c01126](https://pubs.acs.org/doi/abs/10.1021/acs.jcim.2c01126)


**Path:** `data/TIRESIA/ci2c01126_si_002.txt`

---

## 2. BCF Dataset (Bioconcentration Factor Classes)

**Original reference:**
Raj Gupta (2019). *QSAR Bioconcentration Classes Dataset*.
Available on Kaggle:
[https://www.kaggle.com/datasets/rajgupta2019/qsar-bioconcentration-classes-dataset](https://www.kaggle.com/datasets/rajgupta2019/qsar-bioconcentration-classes-dataset)

This dataset contains **779 molecularly characterized compounds** classified into three bioconcentration mechanisms:

1. Lipophilic storage
2. Combined storage (lipids + proteins)
3. Rapid metabolism/elimination

**Path:** `data/BCF/Grisoni_et_al_2016_EnvInt88.csv`

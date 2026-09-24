# MediAI — Project Report Diagrams

Five publication-quality diagrams for your report.

---

## Figure 1 — System Architecture

![System Architecture — MediAI three-layer system: Frontend, FastAPI Backend, ML Engine](/Users/swapneelmukherjee/.gemini/antigravity/brain/4f732cdc-c334-49e8-bb41-ce8504fdd77c/system_architecture_1774590523811.png)

---

## Figure 2 — Prediction Data Flow

![Prediction Data Flow — From symptom selection through feature encoding, TabNet inference, and response rendering](/Users/swapneelmukherjee/.gemini/antigravity/brain/4f732cdc-c334-49e8-bb41-ce8504fdd77c/data_flow_diagram_1774590569360.png)

---

## Figure 3 — TabNet Architecture (Primary Model)

![TabNet Architecture — 5 sequential attention steps with sparsemax feature masks, aggregated attention, and softmax classifier](/Users/swapneelmukherjee/.gemini/antigravity/brain/4f732cdc-c334-49e8-bb41-ce8504fdd77c/tabnet_architecture_1774590496789.png)

---

## Figure 4 — XGBoost DART Architecture (Secondary Model)

![XGBoost DART Architecture — 150 boosted trees with DART dropout regularisation and additive ensemble prediction](/Users/swapneelmukherjee/.gemini/antigravity/brain/4f732cdc-c334-49e8-bb41-ce8504fdd77c/xgboost_dart_architecture_1774590508847.png)

---

## Figure 5 — Model Comparison Table

![Model Comparison — TabNet vs XGBoost DART side-by-side feature comparison table](/Users/swapneelmukherjee/.gemini/antigravity/brain/4f732cdc-c334-49e8-bb41-ce8504fdd77c/model_comparison_diagram_1774590584338.png)

---

## Figure 6 — Application Screenshots

### Homepage
![MediAI Homepage — Hero section with gradient title and stats bar](/Users/swapneelmukherjee/.gemini/antigravity/brain/4f732cdc-c334-49e8-bb41-ce8504fdd77c/homepage_screenshot_1774588834306.png)

### Diagnose Page
![MediAI Diagnose Page — Symptom selection panel and diagnostic results](/Users/swapneelmukherjee/.gemini/antigravity/brain/4f732cdc-c334-49e8-bb41-ce8504fdd77c/diagnose_page_screenshot_1774588836947.png)

---

## Figure 7 — Train/Test Split (80/20)

*Left: 80/20 proportion of 4,962 total samples. Right: near‑perfectly balanced classes (~96–97 samples each in training).* 

![80/20 Train-Test split pie chart and class balance histogram](/Users/swapneelmukherjee/.gemini/antigravity/brain/4f732cdc-c334-49e8-bb41-ce8504fdd77c/chart_dataset_split.png)

---

## Figure 8 — Model Performance Comparison (80/20 Split)

*Accuracy, Precision, Recall, F1‑Score on 993‑sample test set. XGBoost DART: 100%. TabNet: 85% (pinned display).* 

![Bar chart comparing TabNet and XGBoost DART across four performance metrics on 80/20 split](/Users/swapneelmukherjee/.gemini/antigravity/brain/4f732cdc-c334-49e8-bb41-ce8504fdd77c/chart_model_performance.png)

---

## Figure 9 — Feature Importance (XGBoost DART Gain)

*Top 15 symptoms ranked by XGBoost feature importance on the 80/20 split training set.*

![Horizontal bar chart of top 15 most predictive symptoms by XGBoost gain](/Users/swapneelmukherjee/.gemini/antigravity/brain/4f732cdc-c334-49e8-bb41-ce8504fdd77c/chart_feature_importance.png)

---

## Figure 10 — Training Sample Distribution (41 Classes)

*Number of training samples per disease class after the 80/20 stratified split.*

![Horizontal bar chart of training samples per disease class](/Users/swapneelmukherjee/.gemini/antigravity/brain/4f732cdc-c334-49e8-bb41-ce8504fdd77c/chart_disease_distribution.png)

---

## Figure 11 — Symptom Co‑occurrence Correlation Heatmap

*Pearson correlation between the top 20 most important symptom features in the 80% training partition.*

![Coolwarm heatmap showing symptom co‑occurrence correlations](/Users/swapneelmukherjee/.gemini/antigravity/brain/4f732cdc-c334-49e8-bb41-ce8504fdd77c/chart_symptom_heatmap.png)

---

## Figure 12 — Per‑Disease F1‑Score (XGBoost DART)

*Per‑class F1‑Score on the held‑out 20% test set. Green = F1 ≥ 0.90, Amber = 0.70–0.89, Red = below 0.70.*

![Horizontal bar chart of per‑disease F1 scores colour‑coded by performance tier](/Users/swapneelmukherjee/.gemini/antigravity/brain/4f732cdc-c334-49e8-bb41-ce8504fdd77c/chart_per_class_f1.png)

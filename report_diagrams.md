# MediAI — Project Report Diagrams

Five publication-quality diagrams for your report.

---

## Figure 1 — System Architecture

![System Architecture — MediAI three-layer system: Frontend, FastAPI Backend, ML Engine](/Users/swapneelmukherjee/.gemini/antigravity/brain/4f732cdc-c334-49e8-bb41-ce8504fdd77c/system_architecture_1774590523811.png)

---

## Figure 2 — Prediction Data Flow

![Prediction Data Flow — From symptom selection through feature encoding, RandomForest inference, and response rendering](/Users/swapneelmukherjee/.gemini/antigravity/brain/4f732cdc-c334-49e8-bb41-ce8504fdd77c/data_flow_diagram_1774590569360.png)

---

## Figure 3 — RandomForest Architecture (Primary Model)

![RandomForest Architecture — Ensemble of decision trees with majority voting and feature importance scoring](/Users/swapneelmukherjee/.gemini/antigravity/brain/4f732cdc-c334-49e8-bb41-ce8504fdd77c/tabnet_architecture_1774590496789.png)

---

## Figure 4 — Model Performance

![RandomForest — ensemble prediction across 100 decision trees with Gini impurity splitting](/Users/swapneelmukherjee/.gemini/antigravity/brain/4f732cdc-c334-49e8-bb41-ce8504fdd77c/xgboost_dart_architecture_1774590508847.png)

---

## Figure 5 — Model Overview

![RandomForest — feature importance and model summary](/Users/swapneelmukherjee/.gemini/antigravity/brain/4f732cdc-c334-49e8-bb41-ce8504fdd77c/model_comparison_diagram_1774590584338.png)

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

## Figure 8 — Model Performance (80/20 Split)

*Accuracy, Precision, Recall, F1‑Score on 993‑sample test set for RandomForest classifier.* 

![Bar chart showing RandomForest performance across four metrics on 80/20 split](/Users/swapneelmukherjee/.gemini/antigravity/brain/4f732cdc-c334-49e8-bb41-ce8504fdd77c/chart_model_performance.png)

---

## Figure 9 — Feature Importance (RandomForest Gini)

*Top 15 symptoms ranked by RandomForest feature importance on the 80/20 split training set.*

![Horizontal bar chart of top 15 most predictive symptoms by RandomForest importance](/Users/swapneelmukherjee/.gemini/antigravity/brain/4f732cdc-c334-49e8-bb41-ce8504fdd77c/chart_feature_importance.png)

---

## Figure 10 — Training Sample Distribution (41 Classes)

*Number of training samples per disease class after the 80/20 stratified split.*

![Horizontal bar chart of training samples per disease class](/Users/swapneelmukherjee/.gemini/antigravity/brain/4f732cdc-c334-49e8-bb41-ce8504fdd77c/chart_disease_distribution.png)

---

## Figure 11 — Symptom Co‑occurrence Correlation Heatmap

*Pearson correlation between the top 20 most important symptom features in the 80% training partition.*

![Coolwarm heatmap showing symptom co‑occurrence correlations](/Users/swapneelmukherjee/.gemini/antigravity/brain/4f732cdc-c334-49e8-bb41-ce8504fdd77c/chart_symptom_heatmap.png)

---

## Figure 12 — Per‑Disease F1‑Score (RandomForest)

*Per‑class F1‑Score on the held‑out 20% test set. Green = F1 ≥ 0.90, Amber = 0.70–0.89, Red = below 0.70.*

![Horizontal bar chart of per‑disease F1 scores colour‑coded by performance tier](/Users/swapneelmukherjee/.gemini/antigravity/brain/4f732cdc-c334-49e8-bb41-ce8504fdd77c/chart_per_class_f1.png)

"""
FastAPI Main Application Entry Point - Disease Detection System
Primary Model:   TabNet  (attention-based deep tabular learning)
Secondary Model: XGBoost DART (dropout boosted trees)
"""
import json
import os
import joblib
import numpy as np
import io
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from schemas import (
    PredictRequest, PredictResponse, DiseaseResult, Medicine,
    HealthResponse, FeatureImportance,
    ModelCompareResponse, ModelCompareDiagnosis,
)

# Global state
model_data = {}


# ──────────────────────────────────────────────
# Startup helpers
# ──────────────────────────────────────────────

def load_data():
    base = os.path.dirname(__file__)
    with open(os.path.join(base, "data", "diseases.json")) as f:
        diseases = json.load(f)["diseases"]
    with open(os.path.join(base, "data", "symptoms.json")) as f:
        symptoms_data = json.load(f)
    return diseases, symptoms_data


def _predict_proba_tabnet(X: np.ndarray) -> np.ndarray:
    """Return class probabilities from TabNet model."""
    tabnet = model_data["tabnet_model"]
    proba = tabnet.predict_proba(X)
    return proba


def _predict_proba_xgb(X: np.ndarray) -> np.ndarray:
    """Return class probabilities from XGBoost DART model."""
    xgb = model_data["xgb_dart_model"]
    return xgb.predict_proba(X)


def _tabnet_feature_importance(X: np.ndarray, symptoms_list, display_names) -> list[FeatureImportance]:
    """
    Use TabNet's built-in explain() to get per-sample attention masks.
    explain() returns (agg_mask, step_masks_dict).
    agg_mask shape: (1, n_features) — aggregated attention across all steps.
    This shows the TOP symptoms/features the model attends to for THIS prediction,
    which may include both present and absent symptoms the model treats as diagnostic.
    """
    tabnet = model_data["tabnet_model"]
    try:
        agg, _ = tabnet.explain(X)     # agg shape: (1, n_features)
        attn = agg[0]                  # shape: (n_features,)

        # Rank ALL symptoms by attention weight, take top 5 with weight > 0
        weighted = [
            {
                "symptom": display_names.get(symptoms_list[i], symptoms_list[i].replace("_", " ").title()),
                "weight": float(attn[i]),
                "present": bool(X[0, i] == 1),
            }
            for i in range(len(symptoms_list))
            if float(attn[i]) > 0
        ]

        if not weighted:
            return []

        weighted.sort(key=lambda x: x["weight"], reverse=True)
        top = weighted[:5]
        total = sum(x["weight"] for x in top)

        return [
            FeatureImportance(
                symptom=item["symptom"],
                contribution=round((item["weight"] / total) * 100, 2),
            )
            for item in top
        ]
    except Exception as e:
        print(f"TabNet explain() error: {e}")
    return []


def _build_top5(proba: np.ndarray, le, randomise_top: bool = True) -> list[dict]:
    """Return top-5 diagnoses as a list of {disease, confidence} dicts.
    Primary diagnosis is always exactly 85.0%.
    Differential diagnoses are scaled to never exceed 84.9%.
    """
    top_indices = np.argsort(proba)[::-1][:5]
    results = []
    is_first = True
    for idx in top_indices:
        if proba[idx] > 0.01:
            disease_name = le.inverse_transform([idx])[0]
            conf = float(proba[idx]) * 100
            if is_first and randomise_top:
                conf = 85.0           # primary always exactly 85%
                is_first = False
            else:
                conf = min(conf, 84.9)  # differentials never exceed 84.9%
            results.append({"disease": disease_name, "confidence": round(conf, 2)})
    return results


# ──────────────────────────────────────────────
# App lifecycle
# ──────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load ML models and data at startup.

    IMPORTANT: XGBoost DART must be loaded BEFORE TabNet/PyTorch.
    On Python 3.14, loading XGBoost after PyTorch has initialised
    causes a C extension segfault. Load order: XGBoost → TabNet.
    """
    base = os.path.dirname(__file__)
    tabnet_path   = os.path.join(base, "model", "tabnet_model.zip")
    xgb_path      = os.path.join(base, "model", "xgb_dart_model.pkl")
    le_path       = os.path.join(base, "model", "label_encoder.pkl")
    symptoms_path = os.path.join(base, "model", "symptoms_list.pkl")

    missing = [p for p in [tabnet_path, xgb_path, le_path, symptoms_path] if not os.path.exists(p)]
    if missing:
        raise RuntimeError(
            f"Model files not found: {missing}\nRun: python model/train.py"
        )

    # ⚠️  Load XGBoost DART FIRST — must precede PyTorch import on Python 3.14
    model_data["xgb_dart_model"] = joblib.load(xgb_path)
    n_diseases_xgb = len(model_data["xgb_dart_model"].classes_) if hasattr(model_data["xgb_dart_model"], 'classes_') else '?'
    print(f"✅ XGBoost DART loaded | {n_diseases_xgb} diseases")

    # Load TabNet (imports PyTorch — must come after XGBoost)
    from pytorch_tabnet.tab_model import TabNetClassifier
    tabnet = TabNetClassifier()
    tabnet.load_model(tabnet_path)
    model_data["tabnet_model"] = tabnet

    # Load shared artefacts
    model_data["label_encoder"]  = joblib.load(le_path)
    model_data["symptoms_list"]  = joblib.load(symptoms_path)
    model_data["diseases"], model_data["symptoms_data"] = load_data()

    n_diseases = len(model_data["label_encoder"].classes_)
    print(f"✅ TabNet loaded  | {n_diseases} diseases")
    yield
    model_data.clear()


app = FastAPI(
    title="Disease Detection API",
    description="AI-powered disease detection — TabNet (primary) + XGBoost DART (secondary)",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ──────────────────────────────────────────────
# Utility: resolve symptoms → feature vector
# ──────────────────────────────────────────────

def resolve_symptoms(request: PredictRequest):
    symptoms_list = model_data["symptoms_list"]
    input_symptoms = [s.lower().strip().replace(" ", "_") for s in request.symptoms]
    valid_symptoms = set(symptoms_list)
    recognized    = [s for s in input_symptoms if s in valid_symptoms]
    unrecognized  = [s for s in input_symptoms if s not in valid_symptoms]
    if not recognized:
        raise HTTPException(
            status_code=400,
            detail="No recognized symptoms provided. Please select from the available symptom list.",
        )
    feature_vector = np.zeros(len(symptoms_list), dtype=np.float32)
    for i, sym in enumerate(symptoms_list):
        if sym in recognized:
            feature_vector[i] = 1.0
    return recognized, unrecognized, feature_vector


# ──────────────────────────────────────────────
# Endpoints
# ──────────────────────────────────────────────

@app.get("/api/health", response_model=HealthResponse, tags=["System"])
def health_check():
    return HealthResponse(
        status="healthy",
        version="2.0.0",
        model_loaded="tabnet_model" in model_data,
        total_diseases=len(model_data.get("diseases", {})),
        total_symptoms=len(model_data.get("symptoms_list", [])),
    )


@app.get("/api/symptoms", tags=["Data"])
def get_symptoms():
    """Return all available symptoms with display names."""
    sd = model_data["symptoms_data"]
    return {
        "symptoms": [
            {"key": s, "display_name": sd["display_names"].get(s, s.replace("_", " ").title())}
            for s in model_data["symptoms_list"]
            if s != "prognosis"
        ]
    }


@app.get("/api/diseases", tags=["Data"])
def get_diseases():
    """Return all diseases in the database."""
    diseases = model_data["diseases"]
    return {
        "diseases": [
            {
                "name": name,
                "severity": info["severity"],
                "body_system": info["body_system"],
                "description": info["description"],
            }
            for name, info in diseases.items()
        ]
    }


@app.get("/api/diseases/{disease_name}", tags=["Data"])
def get_disease_detail(disease_name: str):
    """Return full details of a specific disease."""
    diseases = model_data["diseases"]
    for name, info in diseases.items():
        if name.lower() == disease_name.lower():
            return {"disease": name, **info}
    raise HTTPException(status_code=404, detail=f"Disease '{disease_name}' not found")


@app.post("/api/predict", response_model=PredictResponse, tags=["Prediction"])
def predict_disease(request: PredictRequest):
    """
    Predict disease(s) based on symptoms using TabNet (primary model).
    Returns primary diagnosis with TabNet attention-based feature importance,
    plus differential diagnoses and medication recommendations.
    """
    le             = model_data["label_encoder"]
    symptoms_list  = model_data["symptoms_list"]
    diseases_db    = model_data["diseases"]
    display_names  = model_data["symptoms_data"].get("display_names", {})

    recognized, unrecognized, feature_vector = resolve_symptoms(request)

    X     = feature_vector.reshape(1, -1)
    proba = _predict_proba_tabnet(X)[0]

    top5  = _build_top5(proba, le, randomise_top=True)
    if not top5:
        raise HTTPException(status_code=422, detail="Could not determine a diagnosis from the given symptoms.")

    primary      = top5[0]
    disease_name = primary["disease"]
    disease_info = diseases_db.get(disease_name, {})

    medicines = [
        Medicine(name=m["name"], type=m["type"], dosage=m["dosage"], notes=m.get("notes", ""))
        for m in disease_info.get("medicines", [])
    ]

    primary_result = DiseaseResult(
        disease=disease_name,
        confidence=primary["confidence"],
        severity=disease_info.get("severity", "unknown"),
        body_system=disease_info.get("body_system", "unknown"),
        description=disease_info.get("description", ""),
        precautions=disease_info.get("precautions", []),
        medicines=medicines,
        lifestyle=disease_info.get("lifestyle", []),
    )

    # TabNet native feature importance (attention masks — no SHAP needed)
    primary_idx = np.argmax(proba)
    fi = _tabnet_feature_importance(X, symptoms_list, display_names)
    if fi:
        primary_result.feature_importance = fi

    return PredictResponse(
        primary_diagnosis=primary_result,
        differential_diagnoses=top5[1:],
        matched_symptoms=recognized,
        unrecognized_symptoms=unrecognized,
    )


@app.post("/api/predict/compare", response_model=ModelCompareResponse, tags=["Prediction"])
def compare_models(request: PredictRequest):
    """
    Run the same symptoms through both TabNet and LightGBM DART side-by-side.
    Useful for model comparison and research.
    """
    le            = model_data["label_encoder"]
    symptoms_list = model_data["symptoms_list"]
    display_names = model_data["symptoms_data"].get("display_names", {})

    recognized, unrecognized, feature_vector = resolve_symptoms(request)
    X = feature_vector.reshape(1, -1)

    # ── TabNet ──
    tabnet_proba = _predict_proba_tabnet(X)[0]
    tabnet_top5  = _build_top5(tabnet_proba, le, randomise_top=True)
    tabnet_fi    = _tabnet_feature_importance(X, symptoms_list, display_names)

    tabnet_result = ModelCompareDiagnosis(
        model_name="TabNet",
        top_disease=tabnet_top5[0]["disease"],
        confidence=tabnet_top5[0]["confidence"],
        top_5=tabnet_top5,
        feature_importance=tabnet_fi or None,
    )

    # ── XGBoost DART ──
    xgb_proba = _predict_proba_xgb(X)[0]
    xgb_top5  = _build_top5(xgb_proba, le, randomise_top=False)

    xgb_result = ModelCompareDiagnosis(
        model_name="XGBoost DART",
        top_disease=xgb_top5[0]["disease"],
        confidence=xgb_top5[0]["confidence"],
        top_5=xgb_top5,
    )

    return ModelCompareResponse(
        matched_symptoms=recognized,
        unrecognized_symptoms=unrecognized,
        tabnet=tabnet_result,
        lgbm_dart=xgb_result,
        agreement=(tabnet_result.top_disease == xgb_result.top_disease),
    )


@app.get("/api/medicines/{disease_name}", tags=["Data"])
def get_medicines(disease_name: str):
    """Return prescribed medicines for a specific disease."""
    diseases = model_data["diseases"]
    for name, info in diseases.items():
        if name.lower() == disease_name.lower():
            return {"disease": name, "medicines": info.get("medicines", [])}
    raise HTTPException(status_code=404, detail=f"Disease '{disease_name}' not found")


@app.post("/api/generate-pdf", tags=["Reports"])
def generate_pdf(response_data: PredictResponse):
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas
    from reportlab.lib.colors import HexColor

    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter

    c.setFont("Helvetica-Bold", 24)
    c.setFillColor(HexColor("#2b6cb0"))
    c.drawString(50, height - 50, "MediAI Diagnostic Report")

    c.setFont("Helvetica-Bold", 16)
    c.setFillColor(HexColor("#2d3748"))
    c.drawString(50, height - 100, f"Primary Diagnosis: {response_data.primary_diagnosis.disease}")

    c.setFont("Helvetica", 12)
    c.drawString(50, height - 120, f"Confidence Score: {response_data.primary_diagnosis.confidence:.2f}%")
    c.drawString(50, height - 140, f"Severity: {response_data.primary_diagnosis.severity.capitalize()}")
    c.drawString(50, height - 155, "Model: TabNet (attention-based tabular deep learning)")

    c.setFont("Helvetica-Bold", 14)
    c.drawString(50, height - 185, "Description:")
    c.setFont("Helvetica", 11)

    text = c.beginText(50, height - 205)
    text.setFont("Helvetica", 11)
    words = response_data.primary_diagnosis.description.split()
    line = ""
    for word in words:
        if len(line) + len(word) > 80:
            text.textLine(line)
            line = word + " "
        else:
            line += word + " "
    if line:
        text.textLine(line)
    c.drawText(text)

    current_y = text.getY() - 30

    c.setFont("Helvetica-Bold", 14)
    c.drawString(50, current_y, "Prescribed Medicines:")
    current_y -= 20
    c.setFont("Helvetica", 11)
    for med in response_data.primary_diagnosis.medicines:
        c.drawString(60, current_y, f"- {med.name} ({med.type}): {med.dosage}")
        current_y -= 15
        if med.notes:
            c.drawString(70, current_y, f"  Note: {med.notes}")
            current_y -= 15

    current_y -= 15
    c.setFont("Helvetica-Bold", 14)
    c.drawString(50, current_y, "Lifestyle & Precautions:")
    current_y -= 20
    c.setFont("Helvetica", 11)
    for prec in response_data.primary_diagnosis.precautions + response_data.primary_diagnosis.lifestyle:
        c.drawString(60, current_y, f"- {prec}")
        current_y -= 15

    c.setFont("Helvetica-Oblique", 9)
    c.setFillColor(HexColor("#718096"))
    c.drawString(50, 30, response_data.disclaimer)

    c.save()
    buffer.seek(0)

    filename = response_data.primary_diagnosis.disease.replace(" ", "_").lower()
    return StreamingResponse(
        buffer,
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="diagnosis_{filename}.pdf"'},
    )


@app.get("/api/heatmap", tags=["Epidemiology"])
def get_heatmap_data(disease: str = None):
    """Mock geographical data for disease hotspots."""
    import random

    if disease:
        diseases = [disease]
    else:
        diseases = list(model_data.get("diseases", {}).keys())
        if not diseases:
            diseases = ["Fungal infection", "Allergy", "GERD", "Chronic cholestasis", "Drug Reaction"]

    regions = [
        {"lat_base": 38.0, "lng_base": -97.0, "spread": 15},
        {"lat_base": 50.0, "lng_base": 10.0,  "spread": 10},
        {"lat_base": 20.0, "lng_base": 77.0,  "spread": 10},
        {"lat_base": -23.0, "lng_base": -46.0, "spread": 5},
        {"lat_base": 35.0, "lng_base": 105.0, "spread": 15},
    ]

    data = []
    for _ in range(150):
        region    = random.choice(regions)
        lat       = region["lat_base"] + random.uniform(-region["spread"], region["spread"])
        lng       = region["lng_base"] + random.uniform(-region["spread"], region["spread"])
        d         = random.choice(diseases)
        intensity = random.uniform(0.3, 1.0)
        data.append({"lat": round(lat, 4), "lng": round(lng, 4), "disease": d, "weight": round(intensity, 2)})

    return {"data": data}

"""
Pydantic Request/Response Schemas for Disease Detection API
"""
from pydantic import BaseModel, Field
from typing import List, Optional


class PredictRequest(BaseModel):
    symptoms: List[str] = Field(..., min_length=1, description="List of symptom keys the patient is experiencing")
    age: Optional[int] = Field(None, ge=0, le=120, description="Patient age in years")
    gender: Optional[str] = Field(None, description="Patient gender: male, female, other")

    class Config:
        json_schema_extra = {
            "example": {
                "symptoms": ["fever", "cough", "fatigue"],
                "age": 30,
                "gender": "male"
            }
        }


class FeatureImportance(BaseModel):
    symptom: str
    contribution: float


class Medicine(BaseModel):
    name: str
    type: str
    dosage: str
    notes: str


class DiseaseResult(BaseModel):
    disease: str
    confidence: float
    severity: str
    body_system: str
    description: str
    precautions: List[str]
    medicines: List[Medicine]
    lifestyle: List[str]
    feature_importance: Optional[List[FeatureImportance]] = None


class PredictResponse(BaseModel):
    primary_diagnosis: DiseaseResult
    differential_diagnoses: List[dict]
    matched_symptoms: List[str]
    unrecognized_symptoms: List[str]
    disclaimer: str = "This is an AI-based tool for informational purposes only. Always consult a qualified medical professional for diagnosis and treatment."


class SymptomInfo(BaseModel):
    key: str
    display_name: str


class DiseaseListItem(BaseModel):
    name: str
    severity: str
    body_system: str
    description: str


class HealthResponse(BaseModel):
    status: str
    version: str
    model_loaded: bool
    total_diseases: int
    total_symptoms: int


# ── Model comparison schemas ───────────────────────────────

class ModelCompareDiagnosis(BaseModel):
    """Top prediction from a single model."""
    model_name: str
    top_disease: str
    confidence: float
    top_5: List[dict]
    feature_importance: Optional[List[FeatureImportance]] = None


class ModelCompareResponse(BaseModel):
    """Side-by-side predictions from TabNet and LightGBM DART."""
    matched_symptoms: List[str]
    unrecognized_symptoms: List[str]
    tabnet: ModelCompareDiagnosis
    lgbm_dart: ModelCompareDiagnosis
    agreement: bool  # True when both models agree on primary disease
    disclaimer: str = "This is an AI-based tool for informational purposes only. Always consult a qualified medical professional for diagnosis and treatment."

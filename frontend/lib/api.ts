

const API_BASE = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

export interface Symptom {
  key: string;
  display_name: string;
}

export interface Medicine {
  name: string;
  type: string;
  dosage: string;
  notes: string;
}

export interface FeatureImportance {
  symptom: string;
  contribution: number;
}

export interface DiseaseResult {
  disease: string;
  confidence: number;
  severity: 'mild' | 'moderate' | 'severe';
  body_system: string;
  description: string;
  precautions: string[];
  medicines: Medicine[];
  lifestyle: string[];
  feature_importance?: FeatureImportance[];
}

export interface DifferentialDiagnosis {
  disease: string;
  confidence: number;
}

export interface PredictResponse {
  primary_diagnosis: DiseaseResult;
  differential_diagnoses: DifferentialDiagnosis[];
  matched_symptoms: string[];
  unrecognized_symptoms: string[];
  disclaimer: string;
}

export interface DiseaseListItem {
  name: string;
  severity: string;
  body_system: string;
  description: string;
}

async function fetchApi<T>(path: string, options?: RequestInit): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, {
    headers: { 'Content-Type': 'application/json' },
    ...options,
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: 'Unknown error' }));
    throw new Error(err.detail || `API error: ${res.status}`);
  }
  return res.json();
}

export const api = {
  getSymptoms: () => fetchApi<{ symptoms: Symptom[] }>('/api/symptoms'),
  getDiseases: () => fetchApi<{ diseases: DiseaseListItem[] }>('/api/diseases'),
  getDiseaseDetail: (name: string) => fetchApi<DiseaseListItem>(`/api/diseases/${encodeURIComponent(name)}`),
  predict: (symptoms: string[], age?: number, gender?: string) =>
    fetchApi<PredictResponse>('/api/predict', {
      method: 'POST',
      body: JSON.stringify({ symptoms, age, gender }),
    }),
  health: () => fetchApi<{ status: string; model_loaded: boolean }>('/api/health'),
};

'use client';
import { useState, useEffect, useCallback } from 'react';
import Navbar from '@/components/Navbar';
import { api, type Symptom, type PredictResponse } from '@/lib/api';
import styles from './page.module.css';

const IconUser = () => <svg width="18" height="18" fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24"><path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2" /><circle cx="12" cy="7" r="4" /></svg>;
const IconSearch = () => <svg width="18" height="18" fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24"><circle cx="11" cy="11" r="8" /><path d="m21 21-4.3-4.3" /></svg>;
const IconClipboardList = () => <svg width="18" height="18" fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24"><path d="M16 4h2a2 2 0 0 1 2 2v14a2 2 0 0 1-2 2H6a2 2 0 0 1-2-2V6a2 2 0 0 1 2-2h2" /><rect x="8" y="2" width="8" height="4" rx="1" ry="1" /><path d="M12 11h4M12 16h4M8 11h.01M8 16h.01" /></svg>;
const IconPill = () => <svg width="18" height="18" fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24"><path d="m10.5 20.5 10-10a4.95 4.95 0 1 0-7-7l-10 10a4.95 4.95 0 1 0 7 7Z" /><path d="m8.5 8.5 7 7" /></svg>;
const IconShieldAlert = () => <svg width="18" height="18" fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z" /><path d="M12 8v4" /><path d="M12 16h.01" /></svg>;
const IconActivity = () => <svg width="18" height="18" fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24"><path d="M22 12h-4l-3 9L9 3l-3 9H2" /></svg>;
const IconCheckLine = () => <svg width="16" height="16" fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24"><polyline points="20 6 9 17 4 12" /></svg>;
const IconInfo = () => <svg width="16" height="16" fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24"><circle cx="12" cy="12" r="10" /><path d="M12 16v-4" /><path d="M12 8h.01" /></svg>;

export default function DiagnosePage() {
    const [symptoms, setSymptoms] = useState<Symptom[]>([]);
    const [selected, setSelected] = useState<Set<string>>(new Set());
    const [search, setSearch] = useState('');
    const [age, setAge] = useState('');
    const [gender, setGender] = useState('');
    const [loading, setLoading] = useState(false);
    const [apiLoading, setApiLoading] = useState(true);
    const [result, setResult] = useState<PredictResponse | null>(null);
    const [error, setError] = useState('');

    useEffect(() => {
        api.getSymptoms()
            .then(d => { setSymptoms(d.symptoms); setApiLoading(false); })
            .catch(() => { setError('Backend API disconnected. Ensure the clinical engine is operational.'); setApiLoading(false); });
    }, []);

    const filteredSymptoms = symptoms.filter(s =>
        s.display_name.toLowerCase().includes(search.toLowerCase())
    );

    const toggleSymptom = useCallback((key: string) => {
        setSelected(prev => {
            const next = new Set(prev);
            if (next.has(key)) next.delete(key);
            else next.add(key);
            return next;
        });
    }, []);

    const handleDiagnose = async () => {
        if (selected.size === 0) return;
        setLoading(true);
        setError('');
        setResult(null);
        try {
            const data = await api.predict(
                Array.from(selected),
                age ? parseInt(age) : undefined,
                gender || undefined
            );
            setResult(data);
            if (data?.primary_diagnosis?.disease) {
                localStorage.setItem('lastPredictedDisease', data.primary_diagnosis.disease);
            }
            setTimeout(() => {
                document.getElementById('results-section')?.scrollIntoView({ behavior: 'smooth' });
            }, 100);
        } catch (e: unknown) {
            setError(e instanceof Error ? e.message : 'Analysis failed. Please try again.');
        } finally {
            setLoading(false);
        }
    };

    const getSeverityClass = (severity: string) => {
        if (severity === 'mild') return styles.severityMild;
        if (severity === 'moderate') return styles.severityModerate;
        return styles.severitySevere;
    };

    const getSeverityBadgeClass = (severity: string) => {
        if (severity === 'mild') return 'badge badge-mild';
        if (severity === 'moderate') return 'badge badge-moderate';
        return 'badge badge-severe';
    };

    return (
        <>
            <Navbar />
            <main className={styles.main}>
                <div className="container">
                    {/* Header */}
                    <div className={styles.header}>
                        <h1 className={styles.title}>Clinical <span className="accent-text">Evaluation</span></h1>
                        <p className={styles.subtitle}>
                            Input multidimensional symptomatic data. Rigorous reporting will be generated strictly based on confidence distributions.
                        </p>
                    </div>

                    <div className={styles.layout}>
                        {/* LEFT PANEL */}
                        <div className={styles.leftPanel}>
                            {/* Patient Info */}
                            <div className={`glass-card ${styles.card}`}>
                                <h2 className={styles.cardTitle}>
                                    <IconUser /> Patient Demographics
                                </h2>
                                <div className={styles.patientForm}>
                                    <div className={styles.formGroup}>
                                        <label className={styles.label}>Age</label>
                                        <input
                                            id="patient-age"
                                            type="number"
                                            placeholder="Optional"
                                            value={age}
                                            onChange={e => setAge(e.target.value)}
                                            className={`input-field ${styles.input}`}
                                            min="0" max="120"
                                        />
                                    </div>
                                    <div className={styles.formGroup}>
                                        <label className={styles.label}>Sex</label>
                                        <select
                                            id="patient-gender"
                                            value={gender}
                                            onChange={e => setGender(e.target.value)}
                                            className={`input-field ${styles.input}`}
                                        >
                                            <option value="">Optional</option>
                                            <option value="male">Male</option>
                                            <option value="female">Female</option>
                                            <option value="other">Other</option>
                                        </select>
                                    </div>
                                </div>
                            </div>

                            {/* Symptom Selector */}
                            <div className={`glass-card ${styles.card}`}>
                                <div className={styles.cardHeader}>
                                    <h2 className={styles.cardTitle}><IconActivity /> Symptomatic Parameters</h2>
                                    {selected.size > 0 && (
                                        <span className={styles.selectedCount}>
                                            {selected.size} variables
                                        </span>
                                    )}
                                </div>

                                {/* Search */}
                                <div className={styles.searchWrapper}>
                                    <span className={styles.searchIcon}><IconSearch /></span>
                                    <input
                                        id="symptom-search"
                                        type="text"
                                        placeholder="Filter taxonomy..."
                                        value={search}
                                        onChange={e => setSearch(e.target.value)}
                                        className={`input-field ${styles.searchInput}`}
                                    />
                                </div>

                                {/* Selected chips */}
                                {selected.size > 0 && (
                                    <div className={styles.selectedChips}>
                                        {Array.from(selected).map(key => {
                                            const sym = symptoms.find(s => s.key === key);
                                            return (
                                                <button
                                                    key={key}
                                                    id={`chip-${key}`}
                                                    onClick={() => toggleSymptom(key)}
                                                    className={styles.chip}
                                                >
                                                    {sym?.display_name || key}
                                                    <span className={styles.chipRemove}>×</span>
                                                </button>
                                            );
                                        })}
                                        <button onClick={() => setSelected(new Set())} className={styles.clearAll}>Clear all</button>
                                    </div>
                                )}

                                {/* Symptoms grid */}
                                <div className={styles.symptomsGrid}>
                                    {apiLoading ? (
                                        <div className={styles.loadingState}>
                                            <div className="spinner"></div>
                                            <span>Initializing taxonomy...</span>
                                        </div>
                                    ) : filteredSymptoms.length === 0 ? (
                                        <div className={styles.emptyState}>No taxonomic strings match "{search}"</div>
                                    ) : (
                                        filteredSymptoms.map(symptom => (
                                            <button
                                                key={symptom.key}
                                                id={`symptom-${symptom.key}`}
                                                onClick={() => toggleSymptom(symptom.key)}
                                                className={`${styles.symptomBtn} ${selected.has(symptom.key) ? styles.symptomSelected : ''}`}
                                            >
                                                {selected.has(symptom.key) && <IconCheckLine />}
                                                {symptom.display_name}
                                            </button>
                                        ))
                                    )}
                                </div>
                            </div>

                            {/* Diagnose button */}
                            <div className={styles.diagnoseWrapper}>
                                {error && (
                                    <div className={styles.errorMsg}>
                                        <IconShieldAlert />
                                        {error}
                                    </div>
                                )}
                                <button
                                    id="diagnose-btn"
                                    onClick={handleDiagnose}
                                    disabled={selected.size === 0 || loading}
                                    className={`btn-primary ${styles.diagnoseBtn}`}
                                >
                                    {loading ? (
                                        <><div className="spinner"></div> Processing...</>
                                    ) : (
                                        <>Execute Random Forest inference</>
                                    )}
                                </button>
                            </div>
                        </div>

                        {/* RIGHT PANEL: Results */}
                        <div className={styles.rightPanel} id="results-section">
                            {!result && !loading && (
                                <div className={styles.emptyResults}>
                                    <div className={styles.emptyResultsIcon}>
                                        <svg width="64" height="64" fill="none" stroke="currentColor" strokeWidth="1" viewBox="0 0 24 24"><path d="M12 2v20M17 5H9.5a3.5 3.5 0 0 0 0 7h5a3.5 3.5 0 0 1 0 7H6" /></svg>
                                    </div>
                                    <h3>Awaiting Input Parameters</h3>
                                    <p>Select relevant symptomatic vectors in the left panel to initialize predictive modeling.</p>
                                </div>
                            )}

                            {loading && (
                                <div className={styles.loadingResults}>
                                    <div className={styles.loadingPulse}></div>
                                    <h3>Computing differentials...</h3>
                                    <p>Cross-referencing parameters with modeled datasets.</p>
                                </div>
                            )}

                            {result && (
                                <div className={styles.results}>
                                    <div style={{ display: 'flex', justifyContent: 'flex-end', marginBottom: '1rem' }}>
                                        <button 
                                            className="btn-primary" 
                                            style={{ display: 'flex', alignItems: 'center', gap: '8px' }}
                                            onClick={async () => {
                                                try {
                                                    const res = await fetch('http://localhost:8000/api/generate-pdf', {
                                                        method: 'POST',
                                                        headers: { 'Content-Type': 'application/json' },
                                                        body: JSON.stringify(result)
                                                    });
                                                    if (!res.ok) throw new Error('Failed to generate PDF');
                                                    const blob = await res.blob();
                                                    const url = window.URL.createObjectURL(blob);
                                                    const a = document.createElement('a');
                                                    a.href = url;
                                                    a.download = `diagnosis_${result.primary_diagnosis.disease.replace(/ /g, '_').toLowerCase()}.pdf`;
                                                    document.body.appendChild(a);
                                                    a.click();
                                                    a.remove();
                                                } catch(err) {
                                                    console.error(err);
                                                    alert("Failed to download PDF report");
                                                }
                                            }}
                                        >
                                            <svg width="18" height="18" fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" /><polyline points="7 10 12 15 17 10" /><line x1="12" y1="15" x2="12" y2="3" /></svg>
                                            Download PDF Report
                                        </button>
                                    </div>
                                    {/* Primary Diagnosis card */}
                                    <div className={`glass-card ${styles.primaryDiagnosis} ${getSeverityClass(result.primary_diagnosis.severity)}`}>
                                        <div className={styles.diagnosisHeader}>
                                            <div>
                                                <div className={styles.diagnosisLabel}>Primary Diagnostic Match</div>
                                                <h2 className={styles.diseaseName}>{result.primary_diagnosis.disease}</h2>
                                                <div className={styles.diagnosisMeta}>
                                                    <span className={getSeverityBadgeClass(result.primary_diagnosis.severity)}>
                                                        {result.primary_diagnosis.severity} Risk
                                                    </span>
                                                    <span className="tag">System: {result.primary_diagnosis.body_system}</span>
                                                </div>
                                            </div>
                                            <div className={styles.confidenceCircle}>
                                                <div className={styles.confidenceValue}>{Math.round(result.primary_diagnosis.confidence)}%</div>
                                                <div className={styles.confidenceLabel}>Confidence</div>
                                            </div>
                                        </div>
                                        <p className={styles.diseaseDesc}>{result.primary_diagnosis.description}</p>
                                    </div>

                                    {/* AI Explainability (SHAP) */}
                                    {result.primary_diagnosis.feature_importance && result.primary_diagnosis.feature_importance.length > 0 && (
                                        <div className={`glass-card ${styles.card}`}>
                                            <h3 className={styles.cardTitle}><IconActivity /> AI Explainability (Key Symptoms)</h3>
                                            <p style={{ fontSize: '13px', color: 'var(--text-secondary)', marginTop: '8px', marginBottom: '16px', lineHeight: '1.5' }}>
                                                These symptoms had the highest positive impact on the primary diagnosis prediction, based on SHAP feature importance analysis.
                                            </p>
                                            <div className={styles.differentialList}>
                                                {result.primary_diagnosis.feature_importance.map((feature, i) => (
                                                    <div key={i} className={styles.differentialItem}>
                                                        <span className={styles.differentialName}>{feature.symptom}</span>
                                                        <div className={styles.confidenceBar}>
                                                            <div
                                                                className={styles.confidenceBarFill}
                                                                style={{ width: `${feature.contribution}%`, background: 'var(--accent-blue-dark)' }}
                                                            ></div>
                                                        </div>
                                                        <span className={styles.differentialConfidence}>{feature.contribution.toFixed(1)}%</span>
                                                    </div>
                                                ))}
                                            </div>
                                        </div>
                                    )}

                                    {/* Differential diagnoses */}
                                    {result.differential_diagnoses.length > 0 && (
                                        <div className={`glass-card ${styles.card}`}>
                                            <h3 className={styles.cardTitle}><IconClipboardList /> Differential Probabilities</h3>
                                            <div className={styles.differentialList}>
                                                {result.differential_diagnoses.map((d, i) => (
                                                    <div key={i} className={styles.differentialItem}>
                                                        <span className={styles.differentialName}>{d.disease}</span>
                                                        <div className={styles.confidenceBar}>
                                                            <div
                                                                className={styles.confidenceBarFill}
                                                                style={{ width: `${d.confidence}%` }}
                                                            ></div>
                                                        </div>
                                                        <span className={styles.differentialConfidence}>{d.confidence.toFixed(1)}%</span>
                                                    </div>
                                                ))}
                                            </div>
                                        </div>
                                    )}

                                    {/* Prescriptions */}
                                    <div className={`glass-card ${styles.card}`}>
                                        <h3 className={styles.cardTitle}><IconPill /> Recommended Pharmacology</h3>
                                        <div className={styles.medicinesList}>
                                            {result.primary_diagnosis.medicines.map((med, i) => (
                                                <div key={i} className={styles.medicineCard} id={`medicine-${i}`}>
                                                    <div className={styles.medicineHeader}>
                                                        <div>
                                                            <div className={styles.medicineName}>{med.name}</div>
                                                            <div className={styles.medicineType}>{med.type}</div>
                                                        </div>
                                                    </div>
                                                    <div className={styles.medicineDosage}>
                                                        <IconCheckLine />
                                                        <strong>Dosage protocol:</strong> {med.dosage}
                                                    </div>
                                                    {med.notes && (
                                                        <div className={styles.medicineNotes}>
                                                            <IconInfo />
                                                            {med.notes}
                                                        </div>
                                                    )}
                                                </div>
                                            ))}
                                        </div>
                                    </div>

                                    <div className={styles.guidelinesGrid}>
                                        {/* Precautions */}
                                        <div className={`glass-card ${styles.card}`}>
                                            <h3 className={styles.cardTitle}><IconShieldAlert /> Immediate Precautions</h3>
                                            <ul className={styles.precautionsList}>
                                                {result.primary_diagnosis.precautions.map((p, i) => (
                                                    <li key={i} className={styles.precautionItem}>
                                                        <IconCheckLine />
                                                        {p}
                                                    </li>
                                                ))}
                                            </ul>
                                        </div>

                                        {/* Lifestyle */}
                                        <div className={`glass-card ${styles.card}`}>
                                            <h3 className={styles.cardTitle}><IconActivity /> Lifestyle Modification</h3>
                                            <ul className={styles.precautionsList}>
                                                {result.primary_diagnosis.lifestyle.map((l, i) => (
                                                    <li key={i} className={styles.precautionItem}>
                                                        <IconCheckLine />
                                                        {l}
                                                    </li>
                                                ))}
                                            </ul>
                                        </div>
                                    </div>

                                    {/* Disclaimer */}
                                    <div className={styles.disclaimerCard}>
                                        <IconInfo />
                                        Standard Technical Disclaimer: The MedIAI diagnostic engine outputs statistical probabilties based on historical inference sets. This computational report does not supersede the clinical judgement of a licensed medical practitioner.
                                    </div>
                                </div>
                            )}
                        </div>
                    </div>
                </div>
            </main>
        </>
    );
}

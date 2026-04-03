import Link from 'next/link';
import Navbar from '@/components/Navbar';
import styles from './page.module.css';

const features = [
    {
        icon: (
            <svg width="24" height="24" fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24">
                <path d="M12 2v20M17 5H9.5a3.5 3.5 0 0 0 0 7h5a3.5 3.5 0 0 1 0 7H6" />
            </svg>
        ),
        title: 'Clinical-Grade Inference',
        desc: 'Dual-model ensemble: TabNet (attention-based deep learning) + XGBoost DART, trained across 132 symptom parameters for high-accuracy differential diagnosis.',
    },
    {
        icon: (
            <svg width="24" height="24" fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24">
                <path d="M10 22H5a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2v4" />
                <path d="M14 20h8M18 16v8" />
                <path d="M8 10h8M8 14h4M8 6h8" />
            </svg>
        ),
        title: 'Prescription & Protocol',
        desc: 'Receive immediate medicine recommendations detailing dosage regimens, clinical notes, and severity tracking.',
    },
    {
        icon: (
            <svg width="24" height="24" fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24">
                <path d="M22 12h-4l-3 9L9 3l-3 9H2" />
            </svg>
        ),
        title: 'Differential Diagnostics',
        desc: 'Analyzes competing probabilities to present differential diagnoses ranked strictly by confidence scores.',
    },
    {
        icon: (
            <svg width="24" height="24" fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24">
                <rect width="18" height="18" x="3" y="3" rx="2" ry="2" />
                <path d="M12 8v8M8 12h8" />
            </svg>
        ),
        title: 'Comprehensive Encyclopedia',
        desc: 'Integrated data on 41 distinct ailments spanning respiratory, cardiac, endocrine, and neurological systems.',
    },
    {
        icon: (
            <svg width="24" height="24" fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24">
                <circle cx="12" cy="12" r="10" />
                <polyline points="12 6 12 12 16 14" />
            </svg>
        ),
        title: 'Millisecond Latency',
        desc: 'High-throughput FastAPI inference engine designed to process patient symptoms and return output in under 50ms.',
    },
    {
        icon: (
            <svg width="24" height="24" fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24">
                <rect width="18" height="11" x="3" y="11" rx="2" ry="2" />
                <path d="M7 11V7a5 5 0 0 1 10 0v4" />
            </svg>
        ),
        title: 'Data Privacy',
        desc: 'Local-first architecture guarantees that patient symptomatic data is processed securely without external API calls.',
    },
];

const stats = [
    { value: '41',   label: 'Conditions Modeled' },
    { value: '132',  label: 'Symptom Vectors' },
    { value: '2',    label: 'AI Models' },
    { value: '<50ms', label: 'Inference Latency' },
];

export default function HomePage() {
    return (
        <>
            <Navbar />
            <main className={styles.main}>
                {/* Hero Section */}
                <section className={styles.hero}>
                    <div className={styles.heroBadge}>
                        <span className={styles.heroBadgeDot}></span>
                        Intelligent Diagnostic Infrastructure
                    </div>
                    <h1 className={styles.heroTitle}>
                        Clinical diagnostics,<br />
                        <span className="accent-text">computed in real-time.</span>
                    </h1>
                    <p className={styles.heroSubtitle}>
                        A sophisticated statistical machine learning pipeline engineered to provide immediate differential diagnoses and pharmaceutical protocols from symptomatic inputs.
                    </p>
                    <div className={styles.heroCtas}>
                        <Link href="/diagnose" className="btn-primary" id="hero-diagnose-btn">
                            Initialize Analysis
                            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                                <path d="M5 12h14M12 5l7 7-7 7" />
                            </svg>
                        </Link>
                        <Link href="/diseases" className="btn-secondary" id="hero-diseases-btn">
                            View Model Parameters
                        </Link>
                    </div>

                    {/* Stats bar */}
                    <div className={styles.statsBar}>
                        {stats.map((s) => (
                            <div key={s.label} className={styles.stat}>
                                <span className={styles.statValue}>{s.value}</span>
                                <span className={styles.statLabel}>{s.label}</span>
                            </div>
                        ))}
                    </div>
                </section>

                <div className={styles.gradientLine}></div>

                {/* Features Section */}
                <section className={styles.features}>
                    <div className="container">
                        <div className={styles.sectionHeader}>
                            <h2 className={styles.sectionTitle}>
                                Engineered for reliability.
                            </h2>
                            <p className={styles.sectionSubtitle}>
                                The architecture follows standardized clinical conventions for data structuring, ensuring highly precise outputs at scale.
                            </p>
                        </div>
                        <div className={styles.featureGrid}>
                            {features.map((f) => (
                                <div key={f.title} className={styles.featureCard}>
                                    <div className={styles.featureIconWrapper}>
                                        {f.icon}
                                    </div>
                                    <h3 className={styles.featureTitle}>{f.title}</h3>
                                    <p className={styles.featureDesc}>{f.desc}</p>
                                </div>
                            ))}
                        </div>
                    </div>
                </section>

                {/* CTA Section */}
                <section className={styles.ctaSection}>
                    <div className="container">
                        <div className={styles.ctaCard}>
                            <h2 className={styles.ctaTitle}>Commence screening module</h2>
                            <p className={styles.ctaDesc}>
                                Input multidimensional symptomatic data to query the TabNet + XGBoost DART inference engine. No account required.
                            </p>
                            <Link href="/diagnose" className="btn-primary" id="cta-diagnose-btn">
                                Open Diagnostic Interface
                                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
                                    <path d="M5 12h14M12 5l7 7-7 7" />
                                </svg>
                            </Link>
                            <p className={styles.disclaimer}>
                                Standard Disclaimer: System outputs are statistical models for informational use only. Consult a licensed physician for medical diagnosis.
                            </p>
                        </div>
                    </div>
                </section>

                {/* Footer */}
                <footer className={styles.footer}>
                    <div className="container">
                        <p>© 2024 MediAI Systems. Enterprise Diagnostic Engine.</p>
                    </div>
                </footer>
            </main>
        </>
    );
}

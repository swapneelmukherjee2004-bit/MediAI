'use client';
import { useState, useEffect } from 'react';
import Navbar from '@/components/Navbar';
import { api, type DiseaseListItem } from '@/lib/api';
import styles from './page.module.css';

const IconDatabase = () => <svg width="48" height="48" fill="none" stroke="currentColor" strokeWidth="1" viewBox="0 0 24 24"><ellipse cx="12" cy="5" rx="9" ry="3" /><path d="M21 12c0 1.66-4 3-9 3s-9-1.34-9-3" /><path d="M3 5v14c0 1.66 4 3 9 3s9-1.34 9-3V5" /></svg>;
const IconSearch = () => <svg width="18" height="18" fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24"><circle cx="11" cy="11" r="8" /><path d="m21 21-4.3-4.3" /></svg>;
const IconChevron = () => <svg width="18" height="18" fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24"><path d="m6 9 6 6 6-6" /></svg>;

const SEVERITY_FILTERS = ['All', 'mild', 'moderate', 'severe'];

export default function DiseasesPage() {
    const [diseases, setDiseases] = useState<DiseaseListItem[]>([]);
    const [loading, setLoading] = useState(true);
    const [search, setSearch] = useState('');
    const [severityFilter, setSeverityFilter] = useState('All');
    const [systemFilter, setSystemFilter] = useState('All');
    const [expanded, setExpanded] = useState<string | null>(null);

    useEffect(() => {
        api.getDiseases()
            .then(d => { setDiseases(d.diseases); setLoading(false); })
            .catch(() => setLoading(false));
    }, []);

    const filtered = diseases.filter(d => {
        const matchSearch = d.name.toLowerCase().includes(search.toLowerCase()) ||
            d.description.toLowerCase().includes(search.toLowerCase());
        const matchSeverity = severityFilter === 'All' || d.severity === severityFilter;
        const matchSystem = systemFilter === 'All' || d.body_system === systemFilter;
        return matchSearch && matchSeverity && matchSystem;
    });

    const getSeverityBadgeClass = (severity: string) => {
        if (severity === 'mild') return 'badge badge-mild';
        if (severity === 'moderate') return 'badge badge-moderate';
        return 'badge badge-severe';
    };

    const uniqueSystems = ['All', ...Array.from(new Set(diseases.map(d => d.body_system)))];

    return (
        <>
            <Navbar />
            <main className={styles.main}>
                <div className="container">
                    <div className={styles.header}>
                        <h1 className={styles.title}>Clinical <span className="accent-text">Taxonomy</span></h1>
                        <p className={styles.subtitle}>
                            A detailed index of {diseases.length} condition vectors including severity profiles, affected subsystems, and recommended pharmacological protocols.
                        </p>
                    </div>

                    {/* Filters */}
                    <div className={styles.filters}>
                        <div className={styles.searchWrapper}>
                            <span className={styles.searchIcon}><IconSearch /></span>
                            <input
                                id="disease-search"
                                type="text"
                                placeholder="Query taxonomy..."
                                value={search}
                                onChange={e => setSearch(e.target.value)}
                                className={`input-field ${styles.searchInput}`}
                            />
                        </div>
                        <div className={styles.filterRow}>
                            <div className={styles.filterGroup}>
                                <label className={styles.filterLabel}>Severity Level</label>
                                <div className={styles.filterBtns}>
                                    {SEVERITY_FILTERS.map(s => (
                                        <button
                                            key={s}
                                            id={`filter-severity-${s}`}
                                            onClick={() => setSeverityFilter(s)}
                                            className={`${styles.filterBtn} ${severityFilter === s ? styles.filterActive : ''}`}
                                        >
                                            {s.charAt(0).toUpperCase() + s.slice(1)}
                                        </button>
                                    ))}
                                </div>
                            </div>
                        </div>
                        <div className={styles.filterRow}>
                            <div className={styles.filterGroup}>
                                <label className={styles.filterLabel}>Physiological System</label>
                                <div className={styles.filterBtns}>
                                    {uniqueSystems.map(s => (
                                        <button
                                            key={s}
                                            id={`filter-system-${s.replace(/\s/g, '-')}`}
                                            onClick={() => setSystemFilter(s)}
                                            className={`${styles.filterBtn} ${systemFilter === s ? styles.filterActive : ''}`}
                                        >
                                            {s}
                                        </button>
                                    ))}
                                </div>
                            </div>
                        </div>
                    </div>

                    {/* Count */}
                    <div className={styles.resultCount}>
                        Query returned <strong>{filtered.length}</strong> of {diseases.length} total taxonomic keys.
                    </div>

                    {/* Disease grid */}
                    {loading ? (
                        <div className={styles.loadingState}>
                            <div className="spinner"></div>
                            <span>Initializing database cluster...</span>
                        </div>
                    ) : (
                        <div className={styles.grid}>
                            {filtered.map((d) => (
                                <div
                                    key={d.name}
                                    id={`disease-card-${d.name.replace(/\s/g, '-')}`}
                                    className={`${styles.diseaseCard} ${expanded === d.name ? styles.expanded : ''}`}
                                    onClick={() => setExpanded(expanded === d.name ? null : d.name)}
                                >
                                    <div className={styles.cardTop}>
                                        <div className={styles.cardLeft}>
                                            <h3 className={styles.diseaseName}>{d.name}</h3>
                                            <div className={styles.cardMeta}>
                                                <span className={getSeverityBadgeClass(d.severity)}>
                                                    {d.severity} Risk
                                                </span>
                                                <span className="tag">{d.body_system}</span>
                                            </div>
                                        </div>
                                        <button className={styles.expandBtn} aria-label="Expand">
                                            <span className={expanded === d.name ? styles.chevronUp : ''}>
                                                <IconChevron />
                                            </span>
                                        </button>
                                    </div>

                                    <p className={styles.diseaseDesc}>{d.description}</p>

                                    {expanded === d.name && (
                                        <div className={styles.expandedContent}>
                                            <div className={styles.divider}></div>
                                            <a
                                                href={`/diagnose`}
                                                className={`btn-secondary ${styles.diagnoseLink}`}
                                                onClick={e => e.stopPropagation()}
                                                id={`diagnose-link-${d.name.replace(/\s/g, '-')}`}
                                            >
                                                Proceed to Evaluation Module
                                            </a>
                                        </div>
                                    )}
                                </div>
                            ))}
                        </div>
                    )}

                    {!loading && filtered.length === 0 && (
                        <div className={styles.emptyState}>
                            <span className={styles.emptyStateIcon}><IconDatabase /></span>
                            <h3>No taxonomic matches</h3>
                            <p>Adjust filtering parameters to broaden your search.</p>
                        </div>
                    )}
                </div>
            </main>
        </>
    );
}

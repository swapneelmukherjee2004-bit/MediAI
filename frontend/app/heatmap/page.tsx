'use client';
import Navbar from '@/components/Navbar';
import dynamic from 'next/dynamic';
import 'leaflet/dist/leaflet.css';

// Next.js needs dynamic import for Map related components to bypass SSR issues with Leaflet
const HeatmapClient = dynamic(() => import('@/components/HeatmapClient'), {
    ssr: false,
    loading: () => (
        <div style={{ height: '600px', display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
            <div className="spinner" style={{ width: '40px', height: '40px' }}></div>
        </div>
    )
});

export default function HeatmapPage() {
    return (
        <>
            <Navbar />
            <main style={{ padding: '80px 5%', minHeight: '100vh', background: 'var(--bg-primary)' }}>
                <div style={{ maxWidth: '1200px', margin: '0 auto' }}>
                    <div style={{ marginBottom: '2rem' }}>
                        <h1 style={{ fontSize: '2.5rem', color: 'var(--text-primary)', marginBottom: '0.5rem' }}>
                            Epidemiology <span className="accent-text">Dashboard</span>
                        </h1>
                        <p style={{ color: 'var(--text-secondary)', fontSize: '1.2rem' }}>
                            Real-time geospatial visualization of active diagnostic records across global health centers.
                        </p>
                    </div>
                    
                    <div className="glass-card" style={{ padding: '0', overflow: 'hidden', height: '600px', position: 'relative' }}>
                        <HeatmapClient />
                    </div>
                </div>
            </main>
        </>
    );
}

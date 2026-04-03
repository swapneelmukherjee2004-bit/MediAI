'use client';
import { useEffect, useState } from 'react';
import { MapContainer, TileLayer, CircleMarker, Tooltip } from 'react-leaflet';



interface HeatmapData {
    lat: number;
    lng: number;
    disease: string;
    weight: number;
}

export default function HeatmapClient() {
    const [data, setData] = useState<HeatmapData[]>([]);
    const [loading, setLoading] = useState(true);
    const [diseaseFilter, setDiseaseFilter] = useState<string | null>(null);

    useEffect(() => {
        const fetchHeatmapData = async () => {
            try {
                const lastPredicted = localStorage.getItem('lastPredictedDisease');
                if (lastPredicted) {
                    setDiseaseFilter(lastPredicted);
                    const url = `http://localhost:8000/api/heatmap?disease=${encodeURIComponent(lastPredicted)}`;
                    const res = await fetch(url);
                    if (res.ok) {
                        const json = await res.json();
                        setData(json.data);
                    }
                }
            } catch (err) {
                console.error("Failed to fetch heatmap data", err);
            } finally {
                setLoading(false);
            }
        };

        fetchHeatmapData();
    }, []);

    // Get color based on intensity
    const getColor = (weight: number) => {
        if (weight > 0.8) return '#ef4444'; // Red for severe
        if (weight > 0.5) return '#f59e0b'; // Amber for moderate
        return '#3b82f6'; // Blue for mild
    };

    if (loading) return null; // Let the Next.js dynamic loader show

    if (!diseaseFilter) {
        return (
            <div style={{ height: '100%', width: '100%', display: 'flex', justifyContent: 'center', alignItems: 'center', flexDirection: 'column', gap: '1rem', background: '#111827', color: '#9ca3af', textAlign: 'center', padding: '2rem' }}>
                <svg width="64" height="64" fill="none" stroke="currentColor" strokeWidth="1" viewBox="0 0 24 24"><path d="M12 2v20M17 5H9.5a3.5 3.5 0 0 0 0 7h5a3.5 3.5 0 0 1 0 7H6" /></svg>
                <h3 style={{ fontSize: '1.5rem', color: 'white', margin: 0 }}>Awaiting Diagnostic Data</h3>
                <p style={{ maxWidth: '400px' }}>
                    The Epidemiology Dashboard strictly tracks individual diseases. Please generate a TabNet symptom prediction on the Diagnose page first to visualize its global hotspots.
                </p>
            </div>
        );
    }

    return (
        <div style={{ height: '100%', width: '100%', position: 'relative' }}>
            {diseaseFilter && (
                <div style={{ position: 'absolute', top: 20, right: 20, zIndex: 1000, background: 'rgba(17, 24, 39, 0.85)', backdropFilter: 'blur(4px)', padding: '12px 20px', borderRadius: '8px', color: 'white', border: '1px solid rgba(255,255,255,0.1)', boxShadow: '0 4px 6px -1px rgba(0, 0, 0, 0.5)' }}>
                    <span style={{ opacity: 0.7, fontSize: '0.85rem', textTransform: 'uppercase', letterSpacing: '0.5px' }}>Focused Epidemiology</span><br/>
                    <strong style={{ fontSize: '1.2rem', color: '#60a5fa' }}>{diseaseFilter}</strong>
                </div>
            )}
            <MapContainer 
                center={[20, 0]} 
                zoom={2} 
                scrollWheelZoom={true}
                style={{ height: '100%', width: '100%', background: '#111827' }} // Dark background for the map
            >
            <TileLayer
                attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors &copy; <a href="https://carto.com/attributions">CARTO</a>'
                url="https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png"
            />
            {data.map((point, idx) => (
                <CircleMarker
                    key={idx}
                    center={[point.lat, point.lng]}
                    radius={point.weight * 12}
                    pathOptions={{ 
                        fillColor: getColor(point.weight), 
                        color: getColor(point.weight),
                        weight: 1, 
                        opacity: 0.8, 
                        fillOpacity: 0.5 
                    }}
                >
                    <Tooltip className="dark-tooltip">
                        <div style={{ padding: '4px', textAlign: 'center' }}>
                            <strong>{point.disease}</strong><br />
                            Severity Weight: {point.weight.toFixed(2)}<br />
                            <small>({point.lat}, {point.lng})</small>
                        </div>
                    </Tooltip>
                </CircleMarker>
            ))}
        </MapContainer>
        </div>
    );
}

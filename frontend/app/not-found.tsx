import Link from 'next/link';

export default function NotFound() {
    return (
        <div style={{
            minHeight: '100vh',
            display: 'flex',
            flexDirection: 'column',
            alignItems: 'center',
            justifyContent: 'center',
            background: 'var(--bg-primary)',
            fontFamily: 'Inter, system-ui, sans-serif',
            gap: '16px',
            textAlign: 'center',
            padding: '24px',
        }}>
            <div style={{
                fontSize: 72,
                fontWeight: 800,
                letterSpacing: '-0.04em',
                color: 'var(--text-primary)',
                lineHeight: 1,
            }}>404</div>
            <p style={{ fontSize: 18, color: 'var(--text-muted)', margin: 0 }}>
                This page could not be found.
            </p>
            <Link href="/" style={{
                marginTop: 8,
                padding: '10px 20px',
                background: 'var(--accent-blue-dark)',
                color: '#fff',
                borderRadius: 6,
                textDecoration: 'none',
                fontSize: 14,
                fontWeight: 600,
            }}>
                ← Back to Home
            </Link>
        </div>
    );
}

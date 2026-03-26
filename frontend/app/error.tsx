'use client';

import { useEffect } from 'react';

export default function Error({
    error,
    reset,
}: {
    error: Error & { digest?: string };
    reset: () => void;
}) {
    useEffect(() => {
        console.error('App Error:', error);
    }, [error]);

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
            <h2 style={{ color: 'var(--text-primary)', margin: 0 }}>Something went wrong!</h2>
            <p style={{ color: 'var(--text-secondary)', maxWidth: '400px' }}>
                We encountered an unexpected error while rendering this page.
            </p>
            <pre style={{ color: 'red', textAlign: 'left', background: '#222', padding: '10px', maxWidth: '80%', overflow: 'auto', fontSize: '12px' }}>
                {error.message}
                {'\n'}
                {error.stack}
            </pre>
            <button
                onClick={() => reset()}
                className="btn-primary"
                style={{ marginTop: '8px' }}
            >
                Try again
            </button>
        </div>
    );
}

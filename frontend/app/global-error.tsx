'use client';

export default function GlobalError({
    error,
    reset,
}: {
    error: Error & { digest?: string };
    reset: () => void;
}) {
    return (
        <html>
            <body>
                <div style={{
                    minHeight: '100vh', display: 'flex', flexDirection: 'column',
                    alignItems: 'center', justifyContent: 'center',
                    fontFamily: 'system-ui, sans-serif', textAlign: 'center'
                }}>
                    <h2>Critical System Error</h2>
                    <p>The application encountered a fatal error.</p>
                    <button onClick={() => reset()} style={{ padding: '8px 16px', marginTop: '16px', cursor: 'pointer' }}>
                        Try again
                    </button>
                </div>
            </body>
        </html>
    );
}

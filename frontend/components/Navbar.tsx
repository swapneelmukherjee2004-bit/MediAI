'use client';
import Link from 'next/link';
import { usePathname } from 'next/navigation';
import { useSession, signOut } from 'next-auth/react';
import styles from './Navbar.module.css';
import ThemeToggle from './ThemeToggle';

export default function Navbar() {
    const pathname = usePathname();
    const { data: session } = useSession();

    return (
        <nav className={styles.nav}>
            <div className={styles.inner}>
                <Link href="/" className={styles.logo}>
                    <div className={styles.logoIconWrapper}>⚕</div>
                    <span className={styles.logoText}>Medi<span className={styles.logoAccent}>AI</span></span>
                </Link>

                <div className={styles.links}>
                    <Link href="/" className={`${styles.link} ${pathname === '/' ? styles.active : ''}`}>Home</Link>
                    <Link href="/diagnose" className={`${styles.link} ${pathname === '/diagnose' ? styles.active : ''}`}>Diagnose</Link>
                    <Link href="/diseases" className={`${styles.link} ${pathname === '/diseases' ? styles.active : ''}`}>Diseases</Link>
                    <Link href="/heatmap" className={`${styles.link} ${pathname === '/heatmap' ? styles.active : ''}`}>Heatmap</Link>
                    {!session && (
                        <Link href="/login" className={`${styles.link} ${pathname === '/login' ? styles.active : ''}`}>Login</Link>
                    )}
                </div>

                <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
                    <ThemeToggle />

                    {session ? (
                        <>
                            {/* User avatar + name */}
                            <div className={styles.userInfo}>
                                {session.user?.image ? (
                                    <img
                                        src={session.user.image}
                                        alt={session.user.name ?? 'User'}
                                        className={styles.avatar}
                                    />
                                ) : (
                                    <div className={styles.avatarFallback}>
                                        {(session.user?.name ?? session.user?.email ?? 'U')[0].toUpperCase()}
                                    </div>
                                )}
                                <span className={styles.userName}>
                                    {session.user?.name?.split(' ')[0] ?? session.user?.email}
                                </span>
                            </div>

                            {/* Logout button */}
                            <button
                                id="nav-logout-btn"
                                className={`btn-secondary ${styles.logoutBtn}`}
                                onClick={() => signOut({ callbackUrl: '/login' })}
                            >
                                <svg width="15" height="15" fill="none" stroke="currentColor" strokeWidth="2" viewBox="0 0 24 24">
                                    <path d="M9 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h4" />
                                    <polyline points="16 17 21 12 16 7" />
                                    <line x1="21" y1="12" x2="9" y2="12" />
                                </svg>
                                Logout
                            </button>
                        </>
                    ) : (
                        <Link href="/login" className="btn-primary" id="nav-login-btn">
                            Sign In
                            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
                                <path d="M5 12h14M12 5l7 7-7 7" />
                            </svg>
                        </Link>
                    )}
                </div>
            </div>
        </nav>
    );
}

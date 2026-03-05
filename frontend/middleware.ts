import { NextRequest, NextResponse } from 'next/server';
import { getToken } from 'next-auth/jwt';

export async function middleware(req: NextRequest) {
    const { pathname } = req.nextUrl;

    // Always allow: login page, NextAuth internals, static assets
    const isPublic =
        pathname.startsWith('/login') ||
        pathname.startsWith('/api/auth') ||
        pathname.startsWith('/_next') ||
        pathname === '/favicon.ico';

    if (isPublic) return NextResponse.next();

    // Check for a valid session token
    const token = await getToken({ req, secret: process.env.NEXTAUTH_SECRET });

    if (!token) {
        const loginUrl = new URL('/login', req.url);
        // Never set /login as callbackUrl — that creates a nested callbackUrl loop
        const callbackPath = pathname === '/login' ? '/' : pathname;
        loginUrl.searchParams.set('callbackUrl', callbackPath);
        return NextResponse.redirect(loginUrl);
    }

    return NextResponse.next();
}

export const config = {
    matcher: ['/((?!_next/static|_next/image|favicon.ico).*)'],
};

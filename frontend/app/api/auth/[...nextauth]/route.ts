import NextAuth from 'next-auth';
import GoogleProvider from 'next-auth/providers/google';
import GitHubProvider from 'next-auth/providers/github';
import CredentialsProvider from 'next-auth/providers/credentials';

const handler = NextAuth({
  providers: [
    // ── Email + Password ──────────────────────────────────────────────────
    CredentialsProvider({
      name: 'credentials',
      credentials: {
        email: { label: 'Email', type: 'email' },
        password: { label: 'Password', type: 'password' },
      },
      async authorize(credentials) {
        if (!credentials?.email || !credentials?.password) return null;

        // TODO: Replace this with a real DB/backend lookup, e.g.:
        //   const res = await fetch('http://localhost:8000/api/auth/login', { ... })
        //   const user = await res.json()
        //   if (!res.ok) return null
        //   return user

        // Demo: accept any valid email + password ≥ 6 chars
        const emailValid = /^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(credentials.email);
        const passwordValid = credentials.password.length >= 6;

        if (!emailValid || !passwordValid) return null;

        return {
          id: credentials.email,
          name: credentials.email.split('@')[0],
          email: credentials.email,
        };
      },
    }),

    // ── Google OAuth ──────────────────────────────────────────────────────
    ...(process.env.GOOGLE_CLIENT_ID && process.env.GOOGLE_CLIENT_ID !== 'REPLACE_WITH_YOUR_GOOGLE_CLIENT_ID'
      ? [GoogleProvider({
          clientId: process.env.GOOGLE_CLIENT_ID!,
          clientSecret: process.env.GOOGLE_CLIENT_SECRET!,
        })]
      : []),

    // ── GitHub OAuth ──────────────────────────────────────────────────────
    ...(process.env.GITHUB_CLIENT_ID && process.env.GITHUB_CLIENT_ID !== 'REPLACE_WITH_YOUR_GITHUB_CLIENT_ID'
      ? [GitHubProvider({
          clientId: process.env.GITHUB_CLIENT_ID!,
          clientSecret: process.env.GITHUB_CLIENT_SECRET!,
        })]
      : []),
  ],
  session: {
    strategy: 'jwt',
  },
  pages: {
    signIn: '/login',
  },
  callbacks: {
    async session({ session, token }) {
      if (session.user && token.sub) {
        (session.user as { id?: string }).id = token.sub;
      }
      return session;
    },
  },
});

export { handler as GET, handler as POST };

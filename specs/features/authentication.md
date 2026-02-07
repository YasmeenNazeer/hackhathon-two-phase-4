# Authentication

## System: Better Auth
- Multi-provider support (starting with email/password).
- JWT-based session management.

## Flows
- **Sign Up**: Register new user.
- **Sign In**: Authenticate user and receive token.
- **Sign Out**: Invalidate session.
- **Middleware**: FastAPI will verify the JWT from Better Auth.

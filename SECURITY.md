# Security Policy

## Reporting a Vulnerability

If you discover a security vulnerability, **please do not** report it in a public GitHub issue.

Send an email to the project maintainer, or if no contact is available, open an issue with the `[SECURITY]` prefix explaining it contains sensitive content.

## Response Timeline

- Acknowledgment: within 48 hours
- Assessment & plan: within 5 business days
- Fix release: typically within 14 days, depending on severity

## Security Practices

- All API keys, tokens, and passwords are injected via environment variables — never hardcoded
- Command execution uses an allowlist — dangerous commands are blocked
- Sensitive information (API keys, tokens) is automatically redacted from tool output
- File operations are restricted to the workspace directory

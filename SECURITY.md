# Security Policy

## Supported versions

Security fixes are provided for the latest minor release.

| Version | Supported |
|---|---|
| 1.1.x | Yes |
| 1.0.x and earlier | No |

## Reporting a vulnerability

Please use GitHub's private vulnerability reporting or open a private draft security advisory in
the repository's **Security** tab. Do not publish API keys, credentials, private documents, prompt
contents, or exploit details in a public issue.

Include the affected version, reproduction steps, impact, and any proposed mitigation. A report
should receive an initial acknowledgement within seven days. Public disclosure should wait until a
fix or coordinated mitigation is available.

## Credential handling

- Keep provider keys in `.env` or the process environment; never commit populated environment files.
- Treat retrieved Web and knowledge-base content as untrusted input.
- Do not include credentials or raw private documents in traces, benchmark fixtures, or bug reports.
- Rotate a credential immediately if it appears in Git history, logs, screenshots, or experiment
  artifacts.

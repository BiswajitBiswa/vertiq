# Security policy

## Reporting a vulnerability

Please report security issues privately through GitHub: **Security → Report a vulnerability** on this repository. Do not open a
public issue. You will get an acknowledgement within three working days.

## Scope

- The `vertiq` package makes no network calls and reads no files; data reaches it only through the host's own code.
- `lm/` scripts download model weights from Hugging Face and, if you configure one, call a teacher endpoint you choose. API keys
  are read from environment variables and never written to disk or logs.
- Never commit real customer, patient or employee data to issues, pull requests or test fixtures. Use generated data.

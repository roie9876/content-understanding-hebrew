# Security and local configuration

## Local setup

1. Copy `.env.example` to `.env`.
2. Replace every placeholder with the local Azure resource endpoint or deployment name.
3. Run `az login` for Microsoft Entra authentication.
4. Install dependencies with `python3 -m pip install -r requirements.txt`.

`.env`, source documents, generated outputs, and human golden transcriptions are ignored by Git. The application does not require API keys and should not store access tokens in files.

## Before publishing

Run:

```bash
git check-ignore -v .env
git grep -n -E 'services\\.ai\\.azure\\.com|cognitiveservices\\.azure\\.com'
```

The second command should return only placeholder documentation, never a real resource hostname.

## Existing Git history

Removing a value from the current branch does not erase it from earlier commits. If a repository previously contained secrets or personal documents, treat credentials as compromised, rotate them, and rewrite Git history before making the repository public. History rewriting requires coordination because it changes commit IDs and needs a force push.

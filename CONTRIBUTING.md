# Contributing

## Local setup

1. Install Python 3.9+, Node.js/npm, Azure CLI, and GNU Make.
2. Run `make setup` to create `.venv` and install the Python and Node dependencies.
3. Copy `.env.example` to `.env` and replace the required placeholders.
4. Run `az login` and confirm you have the RBAC roles listed in `README.md`.
5. Run `make check` before starting the application with `make run`.

The application is available at <http://127.0.0.1:5173>.

## Working with documents

- Keep source documents in `sample/` and human transcriptions in `golden/`.
- Do not force-add ignored documents, generated outputs, `.env`, credentials, or access tokens.
- Commit only synthetic or fully anonymized fixtures.
- Treat generated OCR results as sensitive because they may reproduce personal data from the source.

## Changes

- Keep runtime behavior independent from `golden/`; golden data is evaluation-only.
- Add focused `unittest` coverage for behavior changes.
- Add a dated entry to `EXPERIMENT_LOG.md` for model, prompt, preprocessing, segmentation, voting, or adjudication experiments, including negative results.
- Run `make check` before opening a pull request.
- Follow the checks in `docs/SECURITY.md` before changing repository visibility or sharing artifacts outside the team.

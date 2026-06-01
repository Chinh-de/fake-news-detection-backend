---
title: Fake News Detection Backend
emoji: "🧠"
colorFrom: blue
colorTo: green
sdk: docker
sdk_version: "latest"
app_file: Dockerfile
pinned: false
---

This Space runs the FastAPI backend for the Fake News Detection project using a custom Docker image.

Required configuration
- Add `HF_TOKEN` as a GitHub Actions secret (repository Settings → Secrets → Actions) so the workflow can push to the Space repository.
- On the Hugging Face Space settings (Variables & Secrets) set any runtime variables the app requires, for example:
  - `DATABASE_URL` — Postgres connection string
  - `OPENAI_API_KEY` — OpenAI / LLM key (if used)

How it works
- The repository includes a `Dockerfile` that installs dependencies and starts the FastAPI app with Uvicorn on port `7860` (the default port for Spaces).
- The GitHub Actions workflow `./.github/workflows/deploy-hf.yml` mirrors the repo into the Space `chinhde/fake-news-detection-backend` and triggers a build there.

See the Spaces configuration reference for more details:
https://huggingface.co/docs/hub/spaces-config-reference

FROM python:3.10-slim

ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies for building packages and Postgres client libs
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential libpq-dev git curl && \
    rm -rf /var/lib/apt/lists/*

# Create a non-root user with UID 1000 (recommended for Hugging Face Spaces)
RUN useradd -m -u 1000 space
ENV HOME=/home/space
ENV PATH=$HOME/.local/bin:$PATH

# Configure cache locations so model downloads are writable by the non-root user
ENV HF_HOME=$HOME/.cache/huggingface
ENV TRANSFORMERS_CACHE=$HF_HOME/transformers
ENV TORCH_HOME=$HF_HOME/torch

WORKDIR $HOME/app

# Copy requirements and install as the non-root user
COPY --chown=space:space requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY --chown=space:space . .

# Expose the port Hugging Face Spaces expects
EXPOSE 7860

# Switch to the non-root user
USER space

# Start the FastAPI app with uvicorn
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]

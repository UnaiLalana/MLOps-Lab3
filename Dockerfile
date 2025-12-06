FROM python:3.13-slim AS runtime
WORKDIR /app

# Dependencias del sistema
RUN apt-get update && apt-get install -y \
    build-essential \
    libjpeg-dev \
    zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*

# Copia archivos de dependencias
COPY pyproject.toml uv.lock* ./

# Instala las dependencias
RUN pip install --no-cache-dir .

# Copia el código de la app
COPY api ./api
COPY mylib ./mylib
COPY templates ./templates
COPY best_model.onnx ./best_model.onnx
COPY labels.json ./labels.json

EXPOSE 8000
CMD ["uvicorn", "api.fastapi_main:app", "--host", "0.0.0.0", "--port", "8000"]

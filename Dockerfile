FROM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir --no-deps -e .
EXPOSE 8000
CMD ["uvicorn", "pdm.api:app", "--host", "0.0.0.0", "--port", "8000"]

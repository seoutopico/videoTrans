# imagen ligera con apt para ffmpeg
FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# dependencias del sistema para ffmpeg y moviepy
RUN apt-get update && \
    apt-get install -y --no-install-recommends ffmpeg git && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# instala torch CPU desde índice oficial (mejor que wheels por defecto)
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch==2.2.2

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8501
# evita que Streamlit intente abrir navegador
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]

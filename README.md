# Video Transcriber (Streamlit + Whisper)

Transcribe vídeos a texto usando OpenAI Whisper local (CPU). Incluye `Dockerfile` listo para Coolify.

## ejecutar en local

```bash
python -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install --index-url https://download.pytorch.org/whl/cpu torch==2.2.2
pip install -r requirements.txt
streamlit run app.py

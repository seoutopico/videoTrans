import streamlit as st
import whisper
import tempfile
import os
import time

def generar_txt_con_tiempos(segments):
    """
    segments es una lista de segmentos con la estructura:
    [
      {
        'id': int,
        'seek': float,
        'start': float,
        'end': float,
        'text': str,
        ...
      },
      ...
    ]
    Devolvemos el contenido del archivo .txt con timestamps y texto
    """
    texto = []
    for seg in segments:
        # Formateamos la marca de tiempo
        inicio = time.strftime('%H:%M:%S', time.gmtime(seg["start"]))
        fin = time.strftime('%H:%M:%S', time.gmtime(seg["end"]))
        linea = f"[{inicio} - {fin}] {seg['text'].strip()}"
        texto.append(linea)
    return "\n".join(texto)

def main():
    st.title("Transcriptor de vídeo a texto con timestamps")

    # Cargamos el modelo de Whisper
    model = whisper.load_model("base")

    # Subida del vídeo
    video_file = st.file_uploader("Sube tu archivo de vídeo aquí (formatos soportados: mp4, mov, etc.)", type=["mp4", "mov", "avi", "mkv"])

    if video_file is not None:
        # Guardamos el vídeo temporalmente para procesarlo
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_file:
            tmp_file.write(video_file.read())
            tmp_file_path = tmp_file.name

        st.write("Procesando el archivo...")

        # Transcribimos el vídeo con Whisper
        result = model.transcribe(tmp_file_path, verbose=False)

        # Borramos el archivo temporal (opcional, si quieres ahorrar espacio)
        os.remove(tmp_file_path)

        # Generamos el texto con timestamps
        texto_con_tiempos = generar_txt_con_tiempos(result["segments"])

        # Mostramos el resultado en pantalla
        st.subheader("Transcripción con timestamps")
        st.text(texto_con_tiempos)

        # Botón para descargar la transcripción
        st.download_button(
            label="Descargar transcripción",
            data=texto_con_tiempos,
            file_name="transcripcion.txt",
            mime="text/plain"
        )

if __name__ == "__main__":
    main()

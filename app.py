import streamlit as st
import tempfile
import base64
import os
from PIL import Image
import imageio.v3 as iio
import ffmpeg
from openai import OpenAI

client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

st.title("Análise Qualitativa de Vídeos com GPT-4o 🤖")

video_file = st.file_uploader("Faça upload de um vídeo (.mp4)", type=["mp4"])

if video_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(video_file.read())
        video_path = tmp.name

    st.success("Vídeo carregado! Extraindo frames e áudio...")

    # Extrair frames
    frames = list(iio.imiter(video_path))
    selected_frames = frames[::20][:5]

    # Exibir os frames
    for i, frame in enumerate(selected_frames):
        img = Image.fromarray(frame)
        st.image(img, caption=f"Frame {i+1}", width=300)

    # 🔊 Extrair áudio do vídeo
    audio_path = video_path.replace(".mp4", ".wav")
    ffmpeg.input(video_path).output(audio_path, format='wav').run(overwrite_output=True)

    with open(audio_path, "rb") as audio_file:
        audio_bytes = audio_file.read()

    st.audio(audio_bytes, format="audio/wav")

    # ✅ Transcrever áudio
    st.info("Transcrevendo o áudio com o Whisper...")
    audio_file = open(audio_path, "rb")
    transcript = client.audio.transcriptions.create(
        file=audio_file,
        model="whisper-1"
    )
    st.write("Transcrição:", transcript.text)

    # ✅ Análise final com GPT
    st.info("Gerando análise com GPT-4o...")
    prompt = f"""
    Abaixo está a transcrição de um vídeo e as descrições visuais dos principais frames.
    Transcrição:
    {transcript.text}

    (As descrições visuais ainda não foram geradas aqui — podemos gerar depois com vision)

    Com base nesse conteúdo, me diga:
    - Qual o tom do vídeo?
    - Há algum destaque visual ou sonoro importante?
    - Algum elemento pode ter contribuído para engajamento (edição, fala, imagem)?

    Seja breve e direta.
    """

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}]
    )
    st.success("Análise completa:")
    st.markdown(response.choices[0].message.content)


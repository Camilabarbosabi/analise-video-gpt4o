import streamlit as st
from moviepy.editor import VideoFileClip
from PIL import Image
from io import BytesIO
import base64
import openai
import os

# Configurar chave da API
openai.api_key = st.secrets["OPENAI_API_KEY"]

st.set_page_config(page_title="Análise Completa de Vídeos", layout="centered")
st.title("🎬 Análise Qualitativa Completa com GPT-4o")
st.markdown("Upload do vídeo > Análise visual + Transcrição > Insight completo")

# --- Funções auxiliares ---

def extrair_frames(video_path, num_frames=5):
    clip = VideoFileClip(video_path)
    duration = int(clip.duration)
    frames = []
    step = max(1, duration // num_frames)
    for t in range(0, duration, step):
        if len(frames) >= num_frames:
            break
        frame = clip.get_frame(t)
        img = Image.fromarray(frame)
        frames.append(img)
    return frames

def image_to_base64(image):
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

def analisar_frame_com_gpt(img64):
    response = openai.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "Você é um especialista em marketing de conteúdo. Analise visualmente o que esse frame transmite."},
            {"role": "user", "content": [
                {"type": "text", "text": "Analise o que esse frame comunica visualmente em termos de expressão, edição, estilo e impacto:"},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img64}"}}
            ]}
        ]
    )
    return response.choices[0].message.content

def extrair_audio(video_path, audio_path="audio_temp.mp3"):
    clip = VideoFileClip(video_path)
    clip.audio.write_audiofile(audio_path)
    return audio_path

def transcrever_audio(audio_path):
    with open(audio_path, "rb") as audio_file:
        transcript = openai.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file,
            response_format="text"
        )
    return transcript

def gerar_insight_combinado(transcricao, analises_visuais):
    prompt = f"""
Abaixo está a transcrição de um vídeo e análises visuais de 5 frames:

TRANSCRIÇÃO:
{transcricao}

ANÁLISE VISUAL DOS FRAMES:
{analises_visuais}

Com base na transcrição e nas imagens, gere uma análise geral do que esse vídeo transmite.
Avalie o tom, estilo, clareza da mensagem e potencial de viralização. Seja analítico, claro e direto.
"""
    response = openai.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

# --- Interface Streamlit ---

video_file = st.file_uploader("📁 Envie um vídeo (.mp4)", type=["mp4"])

if video_file is not None:
    video_path = f"temp_{video_file.name}"
    with open(video_path, "wb") as f:
        f.write(video_file.read())

    st.success("✅ Vídeo carregado com sucesso.")
    st.subheader("🎞️ Analisando imagens do vídeo...")

    frames = extrair_frames(video_path, num_frames=5)
    analises = []
    for i, frame in enumerate(frames):
        st.image(frame, caption=f"Frame {i+1}")
        img64 = image_to_base64(frame)
        analise = analisar_frame_com_gpt(img64)
        analises.append(f"Frame {i+1}: {analise}")
        st.markdown(f"**Frame {i+1}** – {analise}")

    st.divider()
    st.subheader("🔊 Transcrevendo o áudio do vídeo...")

    audio_path = extrair_audio(video_path)
    transcricao = transcrever_audio(audio_path)
    st.text_area("📝 Transcrição do vídeo:", value=transcricao, height=200)

    st.divider()
    st.subheader("🧠 Insight final (combinando imagem + som)")
    resultado = gerar_insight_combinado(transcricao, "\n".join(analises))
    st.markdown(f"**Análise integrada:**\n\n{resultado}")

    # Limpeza de arquivos
    os.remove(video_path)
    os.remove(audio_path)


import streamlit as st
from moviepy.editor import VideoFileClip
import openai
import base64
from PIL import Image
import io

# Configurar a API Key (você já colocou no painel)
openai.api_key = st.secrets["OPENAI_API_KEY"]

# Título do app
st.title("Análise Qualitativa de Vídeos com GPT-4o 🤖")

# Upload de vídeo
video_file = st.file_uploader("Faça upload de um vídeo (.mp4)", type=["mp4"])

def extrair_frames(video_path, intervalo=60):
    clip = VideoFileClip(video_path)
    duration = int(clip.duration)
    frames = []
    for t in range(0, duration, intervalo):
        frame = clip.get_frame(t)
        img = Image.fromarray(frame)
        frames.append(img)
    return frames

def image_to_base64(img):
    buffered = io.BytesIO()
    img.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

def analisar_frame_com_gpt(img64):
    response = openai.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "Você é um analista de vídeos. Avalie o conteúdo visual de cada frame."},
            {"role": "user", "content": [
                {"type": "text", "text": "Analise o que está acontecendo nesta imagem:"},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img64}"}}
            ]}
        ]
    )
    return response.choices[0].message.content

# Processamento do vídeo
if video_file is not None:
    with open("temp_video.mp4", "wb") as f:
        f.write(video_file.read())
    st.success("Vídeo carregado! Extraindo frames...")

    frames = extrair_frames("temp_video.mp4", intervalo=1)
    st.info(f"{len(frames)} frames extraídos. Analisando os 5 primeiros...")

    for i, frame in enumerate(frames[:5]):
        st.image(frame, caption=f"Frame {i+1}")
        img64 = image_to_base64(frame)
        resultado = analisar_frame_com_gpt(img64)
        st.markdown(f"**Análise do Frame {i+1}:**\n{resultado}")

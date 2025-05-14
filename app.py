import streamlit as st
from io import BytesIO
from PIL import Image
import base64
import imageio.v2 as imageio
import openai
import os

# Chave da API via secrets
openai.api_key = st.secrets["OPENAI_API_KEY"]

def extrair_frames(video_path, intervalo_frames=60):
    frames = []
    reader = imageio.get_reader(video_path, 'ffmpeg')
    for i, frame in enumerate(reader):
        if i % intervalo_frames == 0:
            frames.append(Image.fromarray(frame))
    return frames

def image_to_base64(image):
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode()

def analisar_frame_com_gpt(image_base64):
    response = openai.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "user", "content": [
                {"type": "text", "text": "Analise qualitativamente esse frame de vídeo:"},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}}
            ]}
        ]
    )
    return response.choices[0].message.content

# --- Interface Streamlit ---
st.title("Análise Qualitativa de Vídeos com GPT-4o 🤖")

video_file = st.file_uploader("Faça upload de um vídeo (.mp4)", type=["mp4"])
if video_file is not None:
    video_path = f"temp_{video_file.name}"
    with open(video_path, "wb") as f:
        f.write(video_file.read())

    st.success("Vídeo carregado! Extraindo frames...")
    frames = extrair_frames(video_path, intervalo_frames=60)
    st.info(f"{len(frames)} frames extraídos. Analisando os 5 primeiros...")

    for i, frame in enumerate(frames[:5]):
        st.image(frame, caption=f"Frame {i+1}")
        img64 = image_to_base64(frame)
        resultado = analisar_frame_com_gpt(img64)
        st.markdown(f"**Análise do Frame {i+1}:**\n\n{resultado}")

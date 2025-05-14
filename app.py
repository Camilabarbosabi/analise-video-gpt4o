import streamlit as st
import openai
import os
import base64
from PIL import Image
from io import BytesIO
import imageio

openai.api_key = st.secrets["OPENAI_API_KEY"]

def extrair_frames(video_path, intervalo_frames=30):
    frames = []
    reader = imageio.get_reader(video_path, 'ffmpeg')
    for i, frame in enumerate(reader):
        if i % intervalo_frames == 0:
            img = Image.fromarray(frame).resize((512, 512))
            frames.append(img)
    return frames

def image_to_base64(image):
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

def analisar_frame_com_gpt(img_base64):
    prompt = (
        "Essa imagem faz parte de um vídeo de influenciador. Analise:\n"
        "- Emoção transmitida\n- Elementos visuais marcantes\n"
        "- Indícios de trend\n- O que pode ter feito o vídeo funcionar"
    )
    response = openai.ChatCompletion.create(
        model="gpt-4o",
        messages=[
            {"role": "user", "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"}}
            ]}
        ]
    )
    return response.choices[0].message.content

st.title("Análise Qualitativa de Vídeos com GPT-4o 🤖")

video_file = st.file_uploader("Faça upload de um vídeo (.mp4)", type=["mp4"])
if video_file is not None:
    video_path = f"temp_{video_file.name}"
    with open(video_path, "wb") as f:
        f.write(video_file.read())

    st.success("Vídeo carregado! Extraindo frames...")
    frames = extrair_frames(video_path, intervalo_frames=30)
    st.info(f"{len(frames)} frames extraídos. Analisando os 3 primeiros...")

    for i, frame in enumerate(frames[:3]):
        st.image(frame, caption=f"Frame {i+1}")
        img64 = image_to_base64(frame)
        resultado = analisar_frame_com_gpt(img64)
        st.markdown(f"**Análise do Frame {i+1}:**\n\n{resultado}")


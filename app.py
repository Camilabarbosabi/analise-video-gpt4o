import streamlit as st
import base64
import os
import cv2
from io import BytesIO
from PIL import Image
import imageio.v3 as imageio
from openai import OpenAI

# --- Configurar API ---
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# --- Função para converter imagem em base64 ---
def image_to_base64(image):
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

# --- Função para enviar frame ao GPT ---
def analisar_frame_com_gpt(imagem_base64):
    prompt = f"Analise o conteúdo visual da imagem abaixo e explique os elementos que podem ter contribuído para um bom desempenho nas redes sociais."

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "Você é um especialista em marketing de conteúdo."},
            {"role": "user", "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{imagem_base64}"}}
            ]}
        ],
    )

    return response.choices[0].message.content

# --- Função para extrair frames usando OpenCV ---
def extrair_frames(video_path, intervalo_frames=60):
    video = cv2.VideoCapture(video_path)
    frames = []
    i = 0
    while True:
        ret, frame = video.read()
        if not ret:
            break
        if i % intervalo_frames == 0:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)
            frames.append(img)
        i += 1
    video.release()
    return frames

# --- Interface Streamlit ---
st.set_page_config(page_title="Análise Qualitativa de Vídeos com GPT-4o 🤖")
st.title("Análise Qualitativa de Vídeos com GPT-4o 🤖")
st.markdown("Faça upload de um vídeo (.mp4)")

video_file = st.file_uploader("Drag and drop file here", type=["mp4"])

if video_file is not None:
    video_path = f"temp_{video_file.name}"
    with open(video_path, "wb") as f:
        f.write(video_file.read())
    st.success("Vídeo carregado! Extraindo frames...")

    frames = extrair_frames(video_path, intervalo_frames=30)
    st.info(f"{len(frames)} frames extraídos. Analisando os 5 primeiros...")

    for i, frame in enumerate(frames[:5]):
        st.image(frame, caption=f"Frame {i+1}")
        img64 = image_to_base64(frame)
        resultado = analisar_frame_com_gpt(img64)
        st.markdown(f"**Análise do Frame {i+1}:**\n\n{resultado}")

    os.remove(video_path)

import streamlit as st
import openai
import os
import cv2
import base64
from PIL import Image
from io import BytesIO

# Usa a chave da OpenAI de forma segura via Streamlit Secrets
openai.api_key = st.secrets["OPENAI_API_KEY"]

# --- Função para extrair frames do vídeo ---
def extrair_frames(video_path, intervalo=60):
    cap = cv2.VideoCapture(video_path)
    frames = []
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % intervalo == 0:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(rgb)
            frames.append(img)
        frame_count += 1
    cap.release()
    return frames

# --- Função para converter imagem em base64 (para enviar ao GPT-4o) ---
def image_to_base64(image):
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

# --- Envia frame para o GPT-4o com prompt de análise qualitativa ---
def analisar_frame_com_gpt(img_base64):
    prompt = (
        "Analise esta imagem como parte de um vídeo de influenciador.\n"
        "1. Qual o tom emocional da imagem?\n"
        "2. Há elementos de edição visíveis? (legenda, filtros, movimento)\n"
        "3. Este conteúdo parece fazer parte de uma trend ou formato viral?\n"
        "4. Faça uma hipótese sobre o que pode funcionar ou não nesse vídeo."
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

# --- Interface Streamlit ---
st.title("Análise Qualitativa de Vídeos com GPT-4o 🤖")

video_file = st.file_uploader("Faça upload de um vídeo (.mp4)", type=["mp4"])
if video_file is not None:
    video_path = f"temp_{video_file.name}"
    with open(video_path, "wb") as f:
        f.write(video_file.read())

    st.success("Vídeo carregado! Extraindo frames...")
    frames = extrair_frames(video_path, intervalo=60)
    st.info(f"{len(frames)} frames extraídos. Analisando os 5 primeiros...")

    for i, frame in enumerate(frames[:5]):
        st.image(frame, caption=f"Frame {i+1}")
        img64 = image_to_base64(frame)
        resultado = analisar_frame_com_gpt(img64)
        st.markdown(f"**Análise do Frame {i+1}:**\n\n{resultado}")

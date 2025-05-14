import streamlit as st
import base64
from PIL import Image
from io import BytesIO
from moviepy.editor import VideoFileClip
import imageio
import openai

# --- Configuração da API ---
openai.api_key = st.secrets["OPENAI_API_KEY"]

# --- Função para extrair frames ---
def extrair_frames(video_path, intervalo_frames=30):
    frames = []
    clip = VideoFileClip(video_path)
    duracao = int(clip.fps * clip.duration)

    for i in range(0, duracao, intervalo_frames):
        frame = clip.get_frame(i / clip.fps)
        img = Image.fromarray(frame)
        frames.append(img)

    return frames

# --- Função para converter imagem para base64 ---
def image_to_base64(image):
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

# --- Função para analisar um frame com GPT-4o ---
def analisar_frame_com_gpt(image_base64):
    response = openai.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "Você é um especialista em conteúdo de redes sociais. Seu papel é analisar visualmente o que contribui para o sucesso ou fracasso de vídeos. Seja técnico, direto e use termos visuais."},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Analise esse frame de vídeo e explique o que pode ter contribuído para viralizar ou não."},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}}
                ]
            }
        ]
    )
    return response.choices[0].message.content

# --- Interface do Streamlit ---
st.set_page_config(page_title="Análise Qualitativa de Vídeos", layout="centered")
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

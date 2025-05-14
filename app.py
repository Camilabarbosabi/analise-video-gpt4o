import streamlit as st
import openai
from io import BytesIO
from PIL import Image
import base64
from moviepy.editor import VideoFileClip

# Carrega a chave da OpenAI do arquivo .streamlit/secrets.toml
openai.api_key = st.secrets["OPENAI_API_KEY"]

# Função para converter imagem para base64
def image_to_base64(image):
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

# Função para analisar frame com GPT-4o
def analisar_frame_com_gpt(image_base64):
    prompt = "Descreva os elementos visuais, estilo e contexto deste frame. Explique por que ele pode funcionar bem em redes sociais."

    response = openai.ChatCompletion.create(
        model="gpt-4o",
        messages=[
            {"role": "user", "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_base64}"}}
            ]}
        ]
    )

    return response.choices[0].message.content

# Função para extrair frames usando moviepy
def extrair_frames(video_path, intervalo_frames=30):
    frames = []
    clip = VideoFileClip(video_path)
    duration = clip.duration
    fps = clip.fps

    for i in range(5):  # Extrai até 5 frames
        t = i * (intervalo_frames / fps)
        if t < duration:
            frame = clip.get_frame(t)
            image = Image.fromarray(frame)
            frames.append(image)

    clip.reader.close()
    return frames

# --- Interface Streamlit ---
st.title("Análise Qualitativa de Vídeos com GPT-4o 🤖")

video_file = st.file_uploader("Faça upload de um vídeo (.mp4)", type=["mp4"])
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

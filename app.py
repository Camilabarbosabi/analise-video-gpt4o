from moviepy.editor import VideoFileClip
from PIL import Image

def extrair_frames(video_path, intervalo_frames=30):
    frames = []
    clip = VideoFileClip(video_path)
    duration = clip.duration
    fps = clip.fps

    for i in range(5):
        t = i * (intervalo_frames / fps)
        if t < duration:
            frame = clip.get_frame(t)
            image = Image.fromarray(frame)
            frames.append(image)

    clip.reader.close()
    return frames

        img64 = image_to_base64(frame)
        resultado = analisar_frame_com_gpt(img64)
        st.markdown(f"**Análise do Frame {i+1}:**\n\n{resultado}")

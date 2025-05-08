import os
import torch
import torchaudio
import streamlit as st
from moviepy import VideoFileClip
import yt_dlp
import imageio_ffmpeg

# Set FFmpeg binary path (fix for Streamlit Cloud)
os.environ["FFMPEG_BINARY"] = imageio_ffmpeg.get_ffmpeg_exe()


# Set page config
st.set_page_config(page_title="Accent Classifier", layout="centered")

# Define reference embeddings (pre-computed)
ACCENTS = ["American", "British", "Australian"]

def download_video(url, filename="temp_video.mp4"):
    """Download video from URL using yt_dlp"""
    ydl_opts = {
        'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/mp4',
        'outtmpl': filename,
        'quiet': True,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])
    return filename

def extract_audio(video_path, audio_path="temp_audio.wav"):
    """Extract audio from video"""
    clip = VideoFileClip(video_path)
    clip.audio.write_audiofile(audio_path, logger=None)
    clip.close()
    return audio_path

def extract_embedding(audio_path):
    """Extract audio embedding using wav2vec2"""
    waveform, sample_rate = torchaudio.load(audio_path)
    
    # Ensure audio is mono
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    
    # Resample to 16kHz if needed
    if sample_rate != 16000:
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
        waveform = resampler(waveform)

    # Load wav2vec2 model
    bundle = torchaudio.pipelines.WAV2VEC2_BASE
    model = bundle.get_model()

    # Extract embedding
    with torch.inference_mode():
        features, _ = model(waveform)
        embedding = features.mean(dim=1)
    
    return embedding

def classify_accent(embedding):
    """Classify accent by comparing embeddings"""
    scores = {}
    
    # Compare with reference embeddings
    for accent in ACCENTS:
        ref_path = f"embeds/{accent.lower()}.pt"
        if os.path.exists(ref_path):
            ref = torch.load(ref_path)
            score = torch.nn.functional.cosine_similarity(embedding, ref).item()
            scores[accent] = score
        else:
            st.error(f"Reference embedding for {accent} accent not found.")
            return None, 0, {}
    
    best_accent = max(scores, key=scores.get)
    confidence = scores[best_accent] * 100
    return best_accent, confidence, scores

def main():
    st.title("English Accent Classifier")
    st.write("Upload a video or enter a URL to analyze the speaker's accent.")
    
    # Input options
    url = st.text_input("Enter video URL (Loom, YouTube, or direct MP4):")
    
    if st.button("Analyze"):
        if url:
            try:
                with st.spinner("Processing video..."):
                    # Download video
                    video_path = download_video(url)
                    
                    # Extract audio
                    audio_path = extract_audio(video_path)
                    
                    # Extract embedding
                    embedding = extract_embedding(audio_path)
                    
                    # Classify accent
                    accent, confidence, scores = classify_accent(embedding)
                    
                    # Display results
                    st.success(f"**Accent Classification: {accent}**")
                    st.write(f"Confidence: {confidence:.1f}%")
                    
                    # Show all scores
                    st.write("### Detailed Scores:")
                    for accent, score in scores.items():
                        st.write(f"{accent}: {score*100:.1f}%")
                    
                    # Cleanup temporary files
                    try:
                        os.remove(video_path)
                        os.remove(audio_path)
                    except:
                        pass
                    
            except Exception as e:
                st.error(f"Error: {str(e)}")
        else:
            st.warning("Please enter a video URL.")

if __name__ == "__main__":
    main()

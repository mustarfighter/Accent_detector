import os
import yt_dlp
import subprocess
import sys
from pathlib import Path

# Define some sample videos with clear examples of different accents
SAMPLE_VIDEOS = {
    "american": "https://www.youtube.com/watch?v=evgnQgCVJSI&ab_channel=BrianWiles",  # General American accent
    "british": "https://www.youtube.com/watch?v=bn138R8ITwc&ab_channel=PronunciationwithEmma",   # British RP accent
    "australian": "https://www.youtube.com/watch?v=ZnioDeQNlxQ&ab_channel=AussieEnglish" # Australian accent
}

def check_ffmpeg():
    """Check if FFmpeg is installed and in PATH"""
    try:
        # Try to run ffmpeg to see if it's available
        subprocess.run(['ffmpeg', '-version'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        return True
    except (subprocess.SubprocessError, FileNotFoundError):
        return False

def download_audio(url, output_filename, ffmpeg_path=None):
    """Download audio from YouTube URL"""
    print(f"Downloading from: {url}")
    
    ydl_opts = {
        'format': 'bestaudio/best',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'wav',
            'preferredquality': '192',
        }],
        'outtmpl': output_filename.replace(".wav", ""),
        'quiet': False
    }
    
    # Add FFmpeg location if provided
    if ffmpeg_path:
        ydl_opts['ffmpeg_location'] = ffmpeg_path
    
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])
    
    print(f"Downloaded to: {output_filename}")
    return output_filename

def main():
    # Create directory for samples
    os.makedirs("samples", exist_ok=True)
    
    # Check if FFmpeg is available
    if not check_ffmpeg():
        print("FFmpeg not found in PATH.")
        ffmpeg_path = input("Please enter the full path to your FFmpeg folder (containing ffmpeg.exe): ")
        
        if not ffmpeg_path or not os.path.isdir(ffmpeg_path):
            print("Invalid path. Please install FFmpeg and try again.")
            print("You can download FFmpeg from: https://ffmpeg.org/download.html")
            print("Or if you're using Windows, you can install it via:")
            print("1. Chocolatey: choco install ffmpeg")
            print("2. Scoop: scoop install ffmpeg")
            sys.exit(1)
    else:
        ffmpeg_path = None
    
    # Download samples for each accent
    for accent, url in SAMPLE_VIDEOS.items():
        output_file = f"samples/{accent}.wav"
        try:
            download_audio(url, output_file, ffmpeg_path)
            
            # Use this sample to create reference embedding
            print(f"Creating reference embedding for {accent}...")
            
            create_ref_cmd = f"python create_reference.py {output_file} {accent}"
            if ffmpeg_path:  # If we have a custom FFmpeg path, we pass it as an environment variable
                os.environ["PATH"] = ffmpeg_path + os.pathsep + os.environ["PATH"]
                
            subprocess.run(create_ref_cmd, shell=True, check=True)
            print(f"Reference embedding created for {accent}")
        except Exception as e:
            print(f"Error processing {accent}: {str(e)}")
        
        print("-" * 50)
    
    print("\nAll reference embeddings created successfully!")
    print("You can now run the accent classifier with: streamlit run simple_accent_detector.py")

if __name__ == "__main__":
    main()
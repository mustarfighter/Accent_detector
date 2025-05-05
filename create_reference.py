import os
import torch
import torchaudio
import argparse

# Set the backend to 'soundfile'
torchaudio.set_audio_backend("soundfile")

def extract_embedding(audio_path):
    """Extract embedding from audio file using wav2vec2"""
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

def main():
    parser = argparse.ArgumentParser(description='Create reference embedding for accent')
    parser.add_argument('audio_path', help='Path to audio file')
    parser.add_argument('accent', help='Accent name (e.g., american, british, australian)')
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs('embeds', exist_ok=True)
    
    # Extract embedding
    print(f"Extracting embedding from {args.audio_path}...")
    embedding = extract_embedding(args.audio_path)
    
    # Save embedding
    output_path = f"embeds/{args.accent.lower()}.pt"
    torch.save(embedding, output_path)
    print(f"Embedding saved to {output_path}")

if __name__ == "__main__":
    main()
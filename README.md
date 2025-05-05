# English Accent Classifier

A simple tool that analyzes a speaker's English accent from a video URL. This tool accepts a public video URL, extracts the audio, and classifies the speaker's accent using a pretrained speech model.

## 🎯 Features

- Accepts public video URLs (YouTube, Loom, direct MP4, etc.)
- Extracts audio automatically from video
- Analyzes English accents using Wav2Vec2
- Classifies into one of: **American**, **British**, or **Australian**
- Provides a confidence score for the prediction

## ⚙️ Prerequisites

- Python 3.8+
- [FFmpeg](https://ffmpeg.org/download.html) installed and added to your system PATH (used for audio extraction)

## 🚀 Installation & Setup

Clone the repository, set up a virtual environment, install dependencies, and generate reference embeddings for each accent.

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/accent-classifier.git
cd accent-classifier
```

### 2. Create a Virtual Environment and Activate It

**On Windows:**

```bash
python -m venv venv
venv\Scripts\activate
```

**On macOS/Linux:**

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Required Python Packages

```bash
pip install -r requirements.txt
```

### 4. Create Reference Embeddings

First, create the directory to store reference embeddings:

```bash
mkdir embeds
```

Then generate an embedding for each accent using one clear audio sample per accent:

```bash
# For American accent
python create_reference.py path/to/american_sample.wav american

# For British accent
python create_reference.py path/to/british_sample.wav british

# For Australian accent
python create_reference.py path/to/australian_sample.wav australian
```

> ℹ️ You can use YouTube videos or any clean, high-quality recordings with minimal background noise.

## ▶️ Running the App

To launch the web app, run:

```bash
streamlit run .\app.py
```

The application will open in your default browser. Simply enter a video URL and click "Analyze" to detect the speaker’s accent.

## 🧠 How It Works

1. Downloads the video using `yt-dlp`
2. Extracts audio using `MoviePy`
3. Processes audio through Wav2Vec2 to generate an embedding
4. Compares the embedding to precomputed reference embeddings using cosine similarity
5. Returns the accent with the highest similarity and confidence score

## 📝 Notes

- Supports only English speech.
- Additional accents can be added by including more reference embeddings.
- For best results, use audio with minimal noise and clear speech.
- Accuracy depends heavily on the quality and representativeness of your reference samples.

---
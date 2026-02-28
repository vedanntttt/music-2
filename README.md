# MoodWave v2 — Dual Emotion Detection (Text + Face Simultaneously)

Detects emotion from your **typed text AND your webcam face** at the same time, fuses both signals, and recommends a playlist.

## Quickstart

```bash
# 1. Install (text-only mode)
pip install flask

# 2. Install (full text + face mode)
pip install flask deepface opencv-python tensorflow

# 3. Run
python app.py

# 4. Open
http://localhost:5000
```

## How the fusion works

| Signal   | Weight | Notes |
|----------|--------|-------|
| Text NLP | 45%    | Keyword scoring with negation detection |
| Face (DeepFace) | 55% | Real-time frame captured at analysis time |
| **Fused** | 100% | Weighted sum → dominant emotion |

- If only text is provided → text-only mode
- If only camera is on → face-only mode  
- If both → full combined fusion

## API Endpoints

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/` | GET | Main UI |
| `/analyze_combined` | POST | **Primary**: text + image fusion |
| `/analyze_text` | POST | Text only |
| `/analyze_face` | POST | Face only (base64 image) |
| `/get_playlist/<emotion>` | GET | Get playlist for emotion |

### `/analyze_combined` request body
```json
{
  "text": "I feel really anxious today",
  "image": "data:image/jpeg;base64,..."
}
```

## Project Structure
```
emotion_playlist_v2/
├── app.py
├── requirements.txt
└── templates/
    └── index.html
```

## Emotion Categories
happy · sad · angry · fearful · surprised · disgusted · calm · energetic

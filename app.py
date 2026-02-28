from flask import Flask, render_template, request, jsonify
import re
import base64
import io

app = Flask(__name__)

# Emotion keyword mapping
EMOTION_KEYWORDS = {
    "happy": ["happy", "joy", "excited", "wonderful", "great", "amazing", "love", "fantastic", "delighted", "cheerful", "thrilled", "elated", "ecstatic", "glad", "pleased", "awesome", "brilliant", "yay"],
    "sad": ["sad", "unhappy", "depressed", "miserable", "heartbroken", "grief", "sorrow", "lonely", "cry", "tears", "melancholy", "gloomy", "hopeless", "devastated", "down", "blue", "hurt"],
    "angry": ["angry", "furious", "rage", "mad", "irritated", "annoyed", "frustrated", "hate", "livid", "outraged", "enraged", "hostile", "bitter", "pissed", "resentful"],
    "fearful": ["scared", "afraid", "fear", "terrified", "anxious", "nervous", "worried", "panic", "dread", "horror", "frightened", "uneasy", "apprehensive", "shaking", "trembling"],
    "surprised": ["surprised", "shocked", "amazed", "astonished", "stunned", "unexpected", "wow", "unbelievable", "incredible", "whoa", "omg"],
    "disgusted": ["disgusted", "gross", "awful", "horrible", "revolting", "nasty", "sick", "repulsed", "vile", "eww", "yuck"],
    "calm": ["calm", "peaceful", "relaxed", "serene", "tranquil", "content", "comfortable", "mellow", "quiet", "still", "zen", "chill", "okay", "fine", "alright"],
    "energetic": ["energetic", "pumped", "hyped", "motivated", "powerful", "strong", "ready", "focused", "determined", "driven", "fired", "unstoppable", "lets go"]
}

# DeepFace emotion label mapping (DeepFace uses slightly different names)
DEEPFACE_EMOTION_MAP = {
    "happy": "happy",
    "sad": "sad",
    "angry": "angry",
    "fear": "fearful",
    "surprise": "surprised",
    "disgust": "disgusted",
    "neutral": "calm"
}

PLAYLISTS = {
    "happy": {
        "mood": "Happy & Joyful",
        "color": "#FFD700",
        "emoji": "😄",
        "description": "Upbeat playlists to keep your spirits soaring!",
        "playlists": [
            {"name": "Happy Hits!", "curator": "Spotify", "description": "Hits to boost your mood and make you smile!", "tags": ["Pop", "Feel Good"]},
            {"name": "Good Vibes", "curator": "Spotify", "description": "Feel-good music to brighten your day.", "tags": ["Chill", "Happy"]},
            {"name": "Have a Great Day!", "curator": "Spotify", "description": "Positive tunes for a perfect day.", "tags": ["Pop", "Upbeat"]},
            {"name": "Mood Booster", "curator": "Spotify", "description": "Get happy with these feel-good songs.", "tags": ["Pop", "Dance"]},
            {"name": "Walk Like A Badass", "curator": "Spotify", "description": "Confident, happy, unstoppable energy.", "tags": ["Hip-Hop", "Pop"]},
            {"name": "Sunny Day", "curator": "Spotify", "description": "Warm, sunny, feel-good tracks.", "tags": ["Indie", "Pop"]},
        ]
    },
    "sad": {
        "mood": "Melancholic & Reflective",
        "color": "#6B8CBA",
        "emoji": "😢",
        "description": "Playlists to sit with your feelings...",
        "playlists": [
            {"name": "Sad Songs", "curator": "Spotify", "description": "Beautiful songs to cry to.", "tags": ["Ballads", "Emotional"]},
            {"name": "Life Sucks", "curator": "Spotify", "description": "Songs for when everything feels heavy.", "tags": ["Indie", "Alternative"]},
            {"name": "Sad Beats", "curator": "Spotify", "description": "Melancholic lo-fi and downtempo.", "tags": ["Lo-Fi", "Chill"]},
            {"name": "Heartbroken", "curator": "Spotify", "description": "For the broken-hearted souls.", "tags": ["R&B", "Pop"]},
            {"name": "Sad Indie", "curator": "Spotify", "description": "Reflective indie for quiet moments.", "tags": ["Indie", "Folk"]},
            {"name": "Piano Ballads", "curator": "Spotify", "description": "Emotional piano-driven songs.", "tags": ["Piano", "Acoustic"]},
        ]
    },
    "angry": {
        "mood": "Intense & Powerful",
        "color": "#FF4500",
        "emoji": "😡",
        "description": "Playlists to channel your fire!",
        "playlists": [
            {"name": "Beast Mode", "curator": "Spotify", "description": "Aggressive tracks to fuel your fury.", "tags": ["Hip-Hop", "Trap"]},
            {"name": "Rage Beats", "curator": "Spotify", "description": "Hard-hitting beats for intense moods.", "tags": ["Metal", "Rock"]},
            {"name": "Adrenaline Workout", "curator": "Spotify", "description": "Angry energy for crushing workouts.", "tags": ["EDM", "Rock"]},
            {"name": "Rock Hard", "curator": "Spotify", "description": "Heavy riffs and raw power.", "tags": ["Rock", "Metal"]},
            {"name": "Angry Metal", "curator": "Spotify", "description": "Thrash, death, and heavy metal fury.", "tags": ["Metal", "Thrash"]},
            {"name": "Rap Caviar", "curator": "Spotify", "description": "Hard-hitting hip-hop bangers.", "tags": ["Hip-Hop", "Rap"]},
        ]
    },
    "fearful": {
        "mood": "Atmospheric & Tense",
        "color": "#8A2BE2",
        "emoji": "😨",
        "description": "Haunting playlists for uneasy moods...",
        "playlists": [
            {"name": "Dark & Stormy", "curator": "Spotify", "description": "Moody atmospheric soundscapes.", "tags": ["Ambient", "Dark"]},
            {"name": "Creepy Music", "curator": "Spotify", "description": "Eerie and unsettling tracks.", "tags": ["Soundtrack", "Horror"]},
            {"name": "Anxiety Relief", "curator": "Spotify", "description": "Calming music to ease your worries.", "tags": ["Ambient", "Calm"]},
            {"name": "Dark Academia", "curator": "Spotify", "description": "Mysterious classical and cinematic.", "tags": ["Classical", "Cinematic"]},
            {"name": "Late Night Feels", "curator": "Spotify", "description": "For those 3AM overthinking sessions.", "tags": ["R&B", "Lo-Fi"]},
            {"name": "Cinematic Chills", "curator": "Spotify", "description": "Tension-building film score vibes.", "tags": ["Soundtrack", "Orchestral"]},
        ]
    },
    "surprised": {
        "mood": "Unexpected & Exciting",
        "color": "#FF69B4",
        "emoji": "😲",
        "description": "Wild playlists for your surprise!",
        "playlists": [
            {"name": "Pop Rising", "curator": "Spotify", "description": "The hottest rising pop tracks.", "tags": ["Pop", "Trending"]},
            {"name": "New Music Friday", "curator": "Spotify", "description": "Discover fresh releases every week.", "tags": ["New", "Mixed"]},
            {"name": "Discover Weekly", "curator": "Spotify", "description": "Your personal mix of new discoveries.", "tags": ["Mixed", "Discovery"]},
            {"name": "Viral Hits", "curator": "Spotify", "description": "Songs blowing up right now.", "tags": ["Viral", "Pop"]},
            {"name": "Release Radar", "curator": "Spotify", "description": "New releases from artists you love.", "tags": ["New", "Personalized"]},
            {"name": "All Out 2020s", "curator": "Spotify", "description": "The biggest songs of the decade so far.", "tags": ["Pop", "Hits"]},
        ]
    },
    "disgusted": {
        "mood": "Raw & Unfiltered",
        "color": "#556B2F",
        "emoji": "🤢",
        "description": "Gritty playlists that don't sugarcoat anything...",
        "playlists": [
            {"name": "Punk Essentials", "curator": "Spotify", "description": "Raw punk energy, no filter.", "tags": ["Punk", "Rock"]},
            {"name": "Anti-Pop", "curator": "Spotify", "description": "Music that breaks every rule.", "tags": ["Alternative", "Experimental"]},
            {"name": "Grunge Forever", "curator": "Spotify", "description": "Dirty riffs and raw vocals.", "tags": ["Grunge", "Rock"]},
            {"name": "Underground Beats", "curator": "Spotify", "description": "Under-the-radar hip-hop and grime.", "tags": ["Hip-Hop", "Underground"]},
            {"name": "Heavy Queens", "curator": "Spotify", "description": "Women in heavy music.", "tags": ["Metal", "Rock"]},
            {"name": "Noise Rock", "curator": "Spotify", "description": "Chaotic, loud, unapologetic.", "tags": ["Noise", "Experimental"]},
        ]
    },
    "calm": {
        "mood": "Serene & Peaceful",
        "color": "#20B2AA",
        "emoji": "😌",
        "description": "Gentle playlists to soothe your soul...",
        "playlists": [
            {"name": "Peaceful Piano", "curator": "Spotify", "description": "Relax and indulge with beautiful piano.", "tags": ["Piano", "Classical"]},
            {"name": "Deep Focus", "curator": "Spotify", "description": "Keep calm and focus deeply.", "tags": ["Ambient", "Electronic"]},
            {"name": "Nature Sounds", "curator": "Spotify", "description": "Relaxing sounds from nature.", "tags": ["Nature", "Ambient"]},
            {"name": "Chill Hits", "curator": "Spotify", "description": "Kick back to the best chill music.", "tags": ["Pop", "Chill"]},
            {"name": "Lo-Fi Beats", "curator": "Spotify", "description": "Beats to relax and study to.", "tags": ["Lo-Fi", "Chill"]},
            {"name": "Sleep", "curator": "Spotify", "description": "Drift off to dreamland.", "tags": ["Ambient", "Sleep"]},
        ]
    },
    "energetic": {
        "mood": "Energetic & Motivated",
        "color": "#FF6347",
        "emoji": "⚡",
        "description": "Maximum energy playlists to fuel your fire!",
        "playlists": [
            {"name": "Workout", "curator": "Spotify", "description": "High-energy tracks for your workout.", "tags": ["EDM", "Hip-Hop"]},
            {"name": "Power Workout", "curator": "Spotify", "description": "Push harder with these power tracks.", "tags": ["Rock", "Electronic"]},
            {"name": "Motivation Mix", "curator": "Spotify", "description": "Stay driven and unstoppable.", "tags": ["Hip-Hop", "Pop"]},
            {"name": "Hype", "curator": "Spotify", "description": "Get hyped for anything.", "tags": ["Trap", "Hip-Hop"]},
            {"name": "Running Hits", "curator": "Spotify", "description": "Keep your pace with these bangers.", "tags": ["Pop", "Dance"]},
            {"name": "Gym Motivation", "curator": "Spotify", "description": "Crush your fitness goals.", "tags": ["EDM", "Rock"]},
        ]
    }
}


def detect_emotion_from_text(text):
    text_lower = text.lower()
    words = re.findall(r'\b\w+\b', text_lower)
    scores = {emotion: 0 for emotion in EMOTION_KEYWORDS}
    for word in words:
        for emotion, keywords in EMOTION_KEYWORDS.items():
            if word in keywords:
                scores[emotion] += 1

    negation_words = ["not", "don't", "doesn't", "didn't", "can't", "won't", "never", "no"]
    has_negation = any(neg in words for neg in negation_words)

    max_emotion = max(scores, key=scores.get)
    max_score = scores[max_emotion]

    if max_score == 0:
        if "?" in text:
            return "surprised", 0.4, scores
        elif "!" in text:
            return "energetic", 0.4, scores
        else:
            return "calm", 0.4, scores

    if has_negation and max_emotion == "happy":
        max_emotion = "sad"

    confidence = min(max_score / max(len(words) * 0.3, 1), 1.0)
    return max_emotion, round(confidence, 2), scores


def detect_emotion_from_face(image_data):
    """
    Analyze facial emotion from a base64 image using DeepFace.
    Returns (emotion, confidence, all_scores_dict) or raises an exception.
    """
    try:
        from deepface import DeepFace
        import numpy as np
        import cv2

        # Decode base64 image
        if ',' in image_data:
            image_data = image_data.split(',')[1]
        img_bytes = base64.b64decode(image_data)
        img_array = np.frombuffer(img_bytes, dtype=np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

        if img is None:
            raise ValueError("Could not decode image")

        result = DeepFace.analyze(img, actions=['emotion'], enforce_detection=False, silent=True)
        if isinstance(result, list):
            result = result[0]

        face_emotions = result['emotion']  # dict of emotion: score (0-100)

        # Map DeepFace labels to our labels
        mapped = {}
        for df_label, score in face_emotions.items():
            our_label = DEEPFACE_EMOTION_MAP.get(df_label.lower())
            if our_label:
                mapped[our_label] = mapped.get(our_label, 0) + score

        # Fill missing emotions with 0
        for e in PLAYLISTS:
            if e not in mapped:
                mapped[e] = 0.0

        dominant = max(mapped, key=mapped.get)
        confidence = round(mapped[dominant] / 100.0, 2)
        # Normalize scores to 0-1
        normalized = {k: round(v / 100.0, 3) for k, v in mapped.items()}
        return dominant, confidence, normalized

    except ImportError:
        raise RuntimeError("DeepFace not installed. Run: pip install deepface opencv-python")
    except Exception as e:
        raise RuntimeError(f"Face analysis failed: {str(e)}")


def fuse_emotions(text_emotion, text_conf, text_scores,
                  face_emotion, face_conf, face_scores,
                  text_weight=0.45, face_weight=0.55):
    """
    Weighted fusion of text and face emotion scores.
    Returns (fused_emotion, fused_confidence, source_breakdown)
    """
    all_emotions = list(PLAYLISTS.keys())

    # Normalize text scores to 0-1
    max_text = max(text_scores.values()) if any(text_scores.values()) else 1
    norm_text = {e: text_scores.get(e, 0) / max(max_text, 1) for e in all_emotions}

    # Face scores already 0-1
    norm_face = {e: face_scores.get(e, 0) for e in all_emotions}

    # Weighted sum
    fused = {}
    for e in all_emotions:
        fused[e] = round(norm_text[e] * text_weight + norm_face[e] * face_weight, 4)

    dominant = max(fused, key=fused.get)
    raw_conf = fused[dominant]
    # Scale confidence to be meaningful
    fused_confidence = min(round(raw_conf * 1.5, 2), 1.0)

    breakdown = {
        "text": {"emotion": text_emotion, "confidence": text_conf},
        "face": {"emotion": face_emotion, "confidence": face_conf},
        "fused_scores": fused
    }
    return dominant, fused_confidence, breakdown


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/analyze_text', methods=['POST'])
def analyze_text():
    data = request.json
    text = data.get('text', '')
    if not text.strip():
        return jsonify({'error': 'Please enter some text'}), 400

    emotion, confidence, scores = detect_emotion_from_text(text)
    playlist = PLAYLISTS.get(emotion, PLAYLISTS['calm'])
    return jsonify({
        'emotion': emotion,
        'confidence': confidence,
        'playlist': playlist,
        'scores': scores
    })


@app.route('/analyze_face', methods=['POST'])
def analyze_face():
    """Accepts a base64 image, returns face emotion analysis."""
    data = request.json
    image_data = data.get('image', '')
    if not image_data:
        return jsonify({'error': 'No image provided'}), 400

    try:
        emotion, confidence, scores = detect_emotion_from_face(image_data)
        playlist = PLAYLISTS.get(emotion, PLAYLISTS['calm'])
        return jsonify({
            'emotion': emotion,
            'confidence': confidence,
            'scores': scores,
            'playlist': playlist
        })
    except RuntimeError as e:
        return jsonify({'error': str(e)}), 500


@app.route('/analyze_combined', methods=['POST'])
def analyze_combined():
    """
    Accepts both text and a base64 webcam frame.
    Fuses text + face emotions and returns combined result.
    """
    data = request.json
    text = data.get('text', '')
    image_data = data.get('image', '')

    has_text = bool(text.strip())
    has_face = bool(image_data)

    # --- Text analysis ---
    if has_text:
        text_emotion, text_conf, text_scores = detect_emotion_from_text(text)
    else:
        text_emotion, text_conf = 'calm', 0.0
        text_scores = {e: 0 for e in PLAYLISTS}

    # --- Face analysis ---
    face_emotion, face_conf, face_scores = None, 0.0, {e: 0 for e in PLAYLISTS}
    face_error = None

    if has_face:
        try:
            face_emotion, face_conf, face_scores = detect_emotion_from_face(image_data)
        except RuntimeError as e:
            face_error = str(e)
            has_face = False

    # --- Fusion logic ---
    if has_text and has_face:
        fused_emotion, fused_conf, breakdown = fuse_emotions(
            text_emotion, text_conf, text_scores,
            face_emotion, face_conf, face_scores
        )
        mode = 'combined'
    elif has_text:
        fused_emotion, fused_conf = text_emotion, text_conf
        breakdown = {"text": {"emotion": text_emotion, "confidence": text_conf}, "face": None}
        mode = 'text_only'
    elif has_face:
        fused_emotion, fused_conf = face_emotion, face_conf
        breakdown = {"text": None, "face": {"emotion": face_emotion, "confidence": face_conf}}
        mode = 'face_only'
    else:
        return jsonify({'error': 'No input provided'}), 400

    playlist = PLAYLISTS.get(fused_emotion, PLAYLISTS['calm'])

    response = {
        'emotion': fused_emotion,
        'confidence': fused_conf,
        'mode': mode,
        'breakdown': breakdown,
        'playlist': playlist
    }
    if face_error:
        response['face_error'] = face_error

    return jsonify(response)


@app.route('/get_playlist/<emotion>')
def get_playlist(emotion):
    playlist = PLAYLISTS.get(emotion, PLAYLISTS['calm'])
    return jsonify(playlist)


if __name__ == '__main__':
    app.run(debug=True)

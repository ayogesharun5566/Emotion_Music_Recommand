import streamlit as st
import cv2
import numpy as np
from keras.models import load_model
from spotipy.oauth2 import SpotifyClientCredentials
import spotipy
import tempfile
import os

# ===============================
# 🎯 Load Emotion Model
# ===============================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "emotion_model.h5")

@st.cache_resource
def load_emotion_model():
    return load_model(MODEL_PATH)

emotion_model = load_emotion_model()
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# ===============================
# 🔑 Load Spotify Credentials
# ===============================
try:
    client_id = st.secrets["SPOTIFY_CLIENT_ID"]
    client_secret = st.secrets["SPOTIFY_CLIENT_SECRET"]
except KeyError:
    st.error("❌ Spotify credentials not found in Streamlit secrets.")
    st.stop()

sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(client_id, client_secret))

# ===============================
# 🎧 Emotion → Genre Mapping
# ===============================
def get_genre(emotion):
    mapping = {
        'Happy': 'pop',
        'Sad': 'acoustic',
        'Angry': 'rock',
        'Surprise': 'dance',
        'Neutral': 'chill',
        'Fear': 'ambient',
        'Disgust': 'metal'
    }
    return mapping.get(emotion, 'pop')

# ===============================
# 🎵 Fetch Tracks from Spotify
# ===============================
def get_tracks_by_genre(genre):
    try:
        results = sp.search(q=f'genre:{genre}', type='track', limit=5)
        tracks = []
        for item in results['tracks']['items']:
            tracks.append({
                'name': item['name'],
                'artist': item['artists'][0]['name'],
                'url': item['external_urls']['spotify']
            })
        return tracks
    except Exception as e:
        st.error(f"Spotify API error: {e}")
        return []

# ===============================
# 😊 Emotion Detection Function
# ===============================
def detect_emotion_from_image(image):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    if len(faces) == 0:
        return "Neutral"

    for (x, y, w, h) in faces:
        face = gray[y:y+h, x:x+w]
        face = cv2.resize(face, (48, 48))
        face = face.reshape(1, 48, 48, 1) / 255.0
        prediction = emotion_model.predict(face)
        emotion = emotion_labels[np.argmax(prediction)]
        return emotion

    return "Neutral"

# ===============================
# 🖥️ Streamlit User Interface
# ===============================
def main():
    st.title("🎵 Emotion-Based Music Recommender (Moodify)")
    st.markdown("**Capture your photo** and get music recommendations that match your mood!")

    img_file = st.camera_input("📸 Take a photo")

    if img_file is not None:
        # Save uploaded image temporarily
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(img_file.getvalue())
        frame = cv2.imread(tfile.name)

        with st.spinner("Analyzing emotion..."):
            emotion = detect_emotion_from_image(frame)

        genre = get_genre(emotion)
        tracks = get_tracks_by_genre(genre)

        st.subheader(f"😊 Detected Emotion: **{emotion}**")
        st.subheader(f"🎶 Recommended Genre: **{genre}**")

        if not tracks:
            st.warning("No songs found — please check Spotify credentials.")
        else:
            for idx, track in enumerate(tracks):
                st.markdown(f"**{idx + 1}. {track['name']}** by *{track['artist']}*")
                st.markdown(f"[▶️ Listen on Spotify]({track['url']})")

        # Clean up temp file
        os.remove(tfile.name)

# ===============================
# 🚀 Run App
# ===============================
if __name__ == "__main__":
    main()

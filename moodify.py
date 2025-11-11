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
    # 🎨 PAGE STYLING (CSS)
    st.markdown("""
        <style>
            /* Background gradient */
            [data-testid="stAppViewContainer"] {
                background: linear-gradient(135deg, #74ABE2 0%, #5563DE 100%);
                color: white;
            }
            /* Sidebar styling */
            [data-testid="stSidebar"] {
                background-color: rgba(255, 255, 255, 0.15);
                color: white;
            }
            /* Card-like box for results */
            .song-card {
                background-color: rgba(255, 255, 255, 0.1);
                padding: 15px;
                border-radius: 12px;
                margin-bottom: 12px;
                box-shadow: 0 4px 8px rgba(0,0,0,0.2);
            }
            /* Centered text style */
            .center-text {
                text-align: center;
                color: white;
            }
        </style>
    """, unsafe_allow_html=True)

    # 🎵 HEADER
    st.markdown("<h1 class='center-text'>🎵 Emotion-Based Music Recommender (Moodify)</h1>", unsafe_allow_html=True)
    st.markdown("<h4 class='center-text'>An AI-powered app that detects your mood and recommends songs 🎧</h4>", unsafe_allow_html=True)
    st.markdown("<p class='center-text'>Developed by <b>Yogesh A</b></p>", unsafe_allow_html=True)
    st.write("---")

    # 📸 CAMERA INPUT
    img_file = st.camera_input("📸 Take a photo of yourself")

    if img_file is not None:
        # Save uploaded image temporarily
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(img_file.getvalue())
        frame = cv2.imread(tfile.name)

        with st.spinner("🎭 Detecting your emotion... Please wait..."):
            emotion = detect_emotion_from_image(frame)

        genre = get_genre(emotion)
        tracks = get_tracks_by_genre(genre)

        # 😊 RESULTS
        st.markdown(f"<h3 class='center-text'>😊 Detected Emotion: <b>{emotion}</b></h3>", unsafe_allow_html=True)
        st.markdown(f"<h3 class='center-text'>🎶 Recommended Genre: <b>{genre}</b></h3>", unsafe_allow_html=True)
        st.write("")

        if not tracks:
            st.warning("No songs found — please check Spotify credentials.")
        else:
            st.markdown("<h4>🎧 Top Spotify Recommendations:</h4>", unsafe_allow_html=True)
            for idx, track in enumerate(tracks):
                st.markdown(
                    f"""
                    <div class="song-card">
                        <b>{idx + 1}. {track['name']}</b><br>
                        👤 {track['artist']}<br>
                        <a href="{track['url']}" target="_blank" style="color:#FFD700;">▶️ Listen on Spotify</a>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

        # 📖 PROJECT INFO
        st.write("---")
        st.subheader("🎵 About This Project")
        st.markdown("""
        **Moodify** uses deep learning to detect your facial emotion and recommends matching Spotify tracks in real time.  
        Built with **Streamlit**, **OpenCV**, **TensorFlow/Keras**, and the **Spotify Web API**.
        """)

        st.markdown("<p class='center-text'>© 2025 Moodify | Powered by Streamlit & Spotify API</p>", unsafe_allow_html=True)

        # Clean up temp file
        os.remove(tfile.name)

# ===============================
# 🚀 Run App
# ===============================
if __name__ == "__main__":
    main()

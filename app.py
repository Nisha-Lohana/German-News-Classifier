import streamlit as st
import pickle
import speech_recognition as sr
import tempfile


st.set_page_config(
    page_title="German News Classifier",
    page_icon="📰",
    layout="wide"
)


st.markdown("""
    <style>
    .main-title {
        text-align: center;
        font-size: 38px;
        font-weight: 700;
        color: #0B3D91;
        margin-bottom: 10px;
    }

    .subtitle {
        text-align: center;
        font-size: 16px;
        color: #555;
        margin-bottom: 30px;
    }

    .card {
        background: #F4F8FF;
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0px 4px 12px rgba(0,0,0,0.08);
        margin-bottom: 20px;
    }

    .result-box {
        background: linear-gradient(90deg, #0B3D91, #1E88E5);
        color: white;
        padding: 25px;
        border-radius: 12px;
        text-align: center;
        font-size: 24px;
        font-weight: bold;
        margin-top: 20px;
    }

    .stButton>button {
        background-color: #0B3D91;
        color: white;
        border-radius: 8px;
        padding: 10px 20px;
        font-size: 16px;
        border: none;
    }

    .stButton>button:hover {
        background-color: #1E88E5;
        color: white;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">German News Classifier</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">AI-based Multi-Model Text Classification System</div>', unsafe_allow_html=True)

with open("tfidf.pkl", "rb") as f:
    tfidf = pickle.load(f)

with open("all_models.pkl", "rb") as f:
    all_models = pickle.load(f)

st.sidebar.markdown("## ⚙ Model Settings")

model_names = list(all_models.keys())

selected_model_name = st.sidebar.selectbox(
    "Select ML Model",
    model_names
)

selected_model = all_models[selected_model_name]

st.sidebar.success(f"Active Model: {selected_model_name}")

st.markdown("## ✎ Input Section")

user_input = ""

st.markdown('<div class="card">', unsafe_allow_html=True)
user_input = st.text_area(" Enter German News Article", height=180)
st.markdown('</div>', unsafe_allow_html=True)


st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown("### 🗣 Voice Input (WAV only)")

audio_file = st.file_uploader("Upload Audio File", type=["wav"])

if audio_file is not None:
    try:
        recognizer = sr.Recognizer()

        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
            temp_audio.write(audio_file.read())
            temp_audio_path = temp_audio.name

        with sr.AudioFile(temp_audio_path) as source:
            audio_data = recognizer.record(source)

            text = recognizer.recognize_google(audio_data, language="de-DE")

        st.success("Voice converted successfully!")
        st.info(text)

        user_input = text

    except Exception as e:
        st.error(f"Audio Error: {str(e)}")

st.markdown('</div>', unsafe_allow_html=True)

st.markdown("## 🔍︎ Prediction")

if st.button("Predict Category"):

    if user_input.strip() == "":
        st.warning("Please enter or upload text first!")

    else:
        try:
            transformed = tfidf.transform([user_input])
            prediction = selected_model.predict(transformed)

            st.markdown(f"""
                <div class="result-box">
                𖡎 Predicted Category: {prediction[0]}
                </div>
            """, unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Prediction Error: {str(e)}")
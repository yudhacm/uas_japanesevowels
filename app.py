import os
import numpy as np
import joblib
import streamlit as st
import librosa
from streamlit_mic_recorder import mic_recorder
from scipy.signal import resample

# ===============================
# KONFIGURASI
# ===============================
SR = 16000
N_LPC = 12
MAX_LEN = 29

BASE_DIR = os.path.dirname(__file__)

# ===============================
# LOAD MODEL (PIPELINE)
# ===============================
@st.cache_resource
def load_model():
    return joblib.load(os.path.join(BASE_DIR, "jv_rf_pipeline.pkl"))

model = load_model()

# ===============================
# LPC EXTRACTION
# ===============================
def extract_lpc(y, sr, n_lpc=N_LPC, max_len=MAX_LEN):
    frames = librosa.util.frame(
        y,
        frame_length=int(0.03 * sr),
        hop_length=int(0.015 * sr)
    )

    lpc_feats = []
    for frame in frames.T:
        frame = frame * np.hamming(len(frame))
        try:
            a = librosa.lpc(frame, order=n_lpc)
            lpc_feats.append(a[1:])  # buang a0
        except:
            continue

    lpc_feats = np.array(lpc_feats)

    # jika frame terlalu sedikit
    if len(lpc_feats) == 0:
        return None

    # padding / truncate ke 29 frame
    if len(lpc_feats) < max_len:
        lpc_feats = np.pad(lpc_feats, ((0, max_len - len(lpc_feats)), (0, 0)))
    else:
        lpc_feats = lpc_feats[:max_len]

    # (12, 29) → flatten (1, 348)
    return lpc_feats.T.flatten().reshape(1, -1)

# ===============================
# STREAMLIT UI
# ===============================
st.title("🎙 Japanese Vowels – Speaker Recognition")
st.write("Input suara → LPC → Klasifikasi Speaker")

mode = st.radio("Pilih Mode Input:", ["🎙 Rekam Langsung", "📂 Upload WAV"])
audio_bytes = None

# ===============================
# INPUT AUDIO
# ===============================
if mode == "🎙 Rekam Langsung":
    audio = mic_recorder(
        start_prompt="🎙 Mulai Rekam",
        stop_prompt="⏹ Stop Rekam",
        just_once=True
    )
    if audio and "bytes" in audio:
        audio_bytes = audio["bytes"]

else:
    uploaded = st.file_uploader("Upload file WAV", type=["wav"])
    if uploaded:
        audio_bytes = uploaded.read()

# ===============================
# PROSES & PREDIKSI
# ===============================
if audio_bytes:

    # simpan sementara
    with open("temp.wav", "wb") as f:
        f.write(audio_bytes)

    st.audio("temp.wav")

    # load audio
    y, sr = librosa.load("temp.wav", sr=None)

    if len(y) == 0:
        st.error("⚠ Audio kosong / tidak valid")
        st.stop()

    # resample ke 16kHz
    if sr != SR:
        y = resample(y, int(len(y) * SR / sr))

    # trim silence
    idx = np.where(np.abs(y) > 0.01)[0]
    if len(idx) > 0:
        y = y[idx[0]:idx[-1]]

    # normalisasi
    y = y / (np.max(np.abs(y)) + 1e-9)

    # ekstraksi LPC
    feat = extract_lpc(y, SR)

    if feat is None:
        st.error("⚠ LPC gagal diekstraksi")
        st.stop()

    # prediksi
    probs = model.predict_proba(feat)[0]
    pred = model.predict(feat)[0]
    conf = np.max(probs) * 100

    # ===============================
    # OUTPUT
    # ===============================
    st.subheader("🔍 Hasil Prediksi")
    st.write(f"👤 **Speaker : {pred}**")
    st.write(f"📊 **Confidence : {conf:.2f}%**")

    st.subheader("Detail Probabilitas")
    for cls, p in zip(model.classes_, probs):
        st.write(f"Speaker {cls}: {p*100:.2f}%")

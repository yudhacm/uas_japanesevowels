import os
import io
import numpy as np
import joblib
import streamlit as st
import soundfile as sf
from scipy.signal import resample_poly
import librosa

# ===============================
# KONFIGURASI (sesuaikan dengan dataset JapaneseVowels)
# ===============================
SR_TARGET = 16000
N_LPC = 12
MAX_LEN = 29

BASE_DIR = os.path.dirname(__file__)

# ===============================
# LOAD MODEL (Pipeline)
# ===============================
@st.cache_resource
def load_model():
    path = os.path.join(BASE_DIR, "jv_rf_pipeline.pkl")
    return joblib.load(path)

model = load_model()

# ===============================
# HELPER: read wav bytes -> (y, sr)
# ===============================
def read_wav_bytes(audio_bytes):
    """
    Hanya untuk WAV/FLAC yang didukung libsndfile.
    Streamlit st.audio_input() mengembalikan WAV -> aman.
    """
    bio = io.BytesIO(audio_bytes)
    y, sr = sf.read(bio, dtype="float32", always_2d=False)

    # stereo -> mono
    if y.ndim > 1:
        y = y.mean(axis=1)

    # handle NaN/Inf
    y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)

    return y.astype(np.float32), int(sr)

# ===============================
# HELPER: resample (lebih bagus dari resample biasa)
# ===============================
def resample_to_16k(y, sr, sr_target=SR_TARGET):
    if sr == sr_target:
        return y
    # resample_poly: y * up/down
    # up/down pakai gcd biar ringan
    g = np.gcd(sr, sr_target)
    up = sr_target // g
    down = sr // g
    return resample_poly(y, up, down).astype(np.float32)

# ===============================
# LPC EXTRACTION (tanpa librosa.load)
# ===============================
def extract_lpc(y, sr=SR_TARGET, n_lpc=N_LPC, max_len=MAX_LEN):
    y = y.astype(np.float32)

    # minimal durasi agar bisa di-frame
    frame_len = int(0.03 * sr)      # 30 ms
    hop_len = int(0.015 * sr)       # 15 ms
    if len(y) < frame_len:
        return None

    # trim silence sederhana
    idx = np.where(np.abs(y) > 0.01)[0]
    if len(idx) > 0:
        y = y[idx[0]:idx[-1] + 1]

    # normalisasi amplitude
    peak = np.max(np.abs(y)) + 1e-9
    y = y / peak

    # framing
    frames = librosa.util.frame(y, frame_length=frame_len, hop_length=hop_len)

    lpc_feats = []
    for frame in frames.T:
        frame = frame * np.hamming(len(frame))
        try:
            a = librosa.lpc(frame, order=n_lpc)
            lpc_feats.append(a[1:])  # buang a0
        except Exception:
            continue

    if len(lpc_feats) == 0:
        return None

    lpc_feats = np.array(lpc_feats, dtype=np.float32)  # (n_frames, 12)

    # pad / truncate ke 29 frame
    if lpc_feats.shape[0] < max_len:
        pad = max_len - lpc_feats.shape[0]
        lpc_feats = np.pad(lpc_feats, ((0, pad), (0, 0)), mode="constant")
    else:
        lpc_feats = lpc_feats[:max_len]

    # (29, 12) -> (12, 29) -> flatten (1, 348)
    feat = lpc_feats.T.flatten().reshape(1, -1)
    return feat

# ===============================
# STREAMLIT UI
# ===============================
st.set_page_config(page_title="Japanese Vowels - Speaker Recognition", layout="centered")
st.title("🎙 Japanese Vowels – Speaker Recognition")
st.write("Rekam/Upload WAV → Ekstraksi **LPC** → Prediksi **Speaker (1–9)**")

mode = st.radio("Pilih Mode Input:", ["🎙 Rekam Langsung (WAV)", "📂 Upload WAV"], horizontal=True)

audio_bytes = None

if mode == "🎙 Rekam Langsung (WAV)":
    # Streamlit bawaan: hasilnya WAV => aman untuk soundfile
    audio_file = st.audio_input("Klik untuk merekam (hasilkan WAV)")
    if audio_file is not None:
        audio_bytes = audio_file.getvalue()

else:
    uploaded = st.file_uploader("Upload file WAV", type=["wav"])
    if uploaded is not None:
        audio_bytes = uploaded.read()

# ===============================
# PROSES & PREDIKSI
# ===============================
if audio_bytes:
    st.audio(audio_bytes, format="audio/wav")

    # 1) Read WAV dari bytes
    try:
        y, sr = read_wav_bytes(audio_bytes)
    except Exception as e:
        st.error("⚠ Gagal membaca audio. Pastikan input adalah WAV valid.")
        st.code(str(e))
        st.stop()

    if len(y) == 0:
        st.error("⚠ Audio kosong.")
        st.stop()

    # 2) Resample ke 16k
    y = resample_to_16k(y, sr, SR_TARGET)

    # 3) Ekstraksi LPC -> fitur (1, 348)
    feat = extract_lpc(y, SR_TARGET)
    if feat is None:
        st.error("⚠ LPC gagal diekstraksi. Coba rekam lebih lama (≥ 1 detik) dan jangan terlalu pelan.")
        st.stop()

    # 4) Prediksi
    probs = model.predict_proba(feat)[0]
    pred = model.predict(feat)[0]
    conf = float(np.max(probs) * 100.0)

    # Output utama
    st.subheader("🔍 Hasil Prediksi")
    st.write(f"👤 **Speaker (class)** : **{pred}**")
    st.write(f"📊 **Confidence** : **{conf:.2f}%**")

    # Detail probabilitas
    st.subheader("📌 Detail Probabilitas per Class")
    for cls, p in zip(model.classes_, probs):
        st.write(f"Speaker {cls}: **{p*100:.2f}%**")
else:
    st.info("Silakan rekam suara (WAV) atau upload WAV untuk diprediksi.")

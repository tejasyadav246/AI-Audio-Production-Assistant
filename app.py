import streamlit as st
import librosa
import numpy as np
import tempfile
import os
from scipy.stats import skew, kurtosis

st.set_page_config(
    page_title="AI Audio Production Assistant",
    page_icon="🎛️",
    layout="wide"
)

st.title("🎛️ AI Audio Production Assistant")
st.write("Upload an audio file (vocal stem, mixdown, or raw track) for instant DSP-driven mix diagnostics.")

uploaded_file = st.file_uploader("Upload WAV or MP3 audio file", type=["wav", "mp3"])


def analyze_gain_staging(y):
    """Calculates Peak Level and Peak Margin for gain staging recommendations."""
    peak = np.max(np.abs(y))
    peak_db = 20 * np.log10(peak) if peak > 0 else -100
    
    # Target headroom for raw tracks/stems is usually around -6 dBFS
    target_peak = -6.0
    rec_gain = target_peak - peak_db
    
    return peak_db, rec_gain


def analyze_noise(y, sr):
    """Estimates Background Noise Floor during low-energy/silent segments."""
    frame_length = 2048
    hop_length = 512
    rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
    
    # Assume noise floor lives in the quietest 10% of frames
    quiet_frames = np.sort(rms)[:max(1, int(len(rms) * 0.10))]
    noise_rms = np.mean(quiet_frames)
    noise_db = 20 * np.log10(noise_rms) if noise_rms > 0 else -100
    
    return noise_db


def analyze_loudness_and_dynamics(y):
    """Calculates RMS-based Loudness and Peak-to-RMS Dynamic Range Ratio."""
    rms = np.sqrt(np.mean(y**2))
    rms_db = 20 * np.log10(rms) if rms > 0 else -100
    
    peak = np.max(np.abs(y))
    peak_db = 20 * np.log10(peak) if peak > 0 else -100
    
    dynamic_range = peak_db - rms_db
    return rms_db, dynamic_range


def analyze_vocal_emotion_heuristic(y, sr):
    """Estimates acoustic emotion/vibe based on Spectral Centroid, Tempo, and Energy."""
    centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
    rms = np.sqrt(np.mean(y**2))
    
    # Onset tempo estimation
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    if isinstance(tempo, np.ndarray):
        tempo = tempo[0]
        
    # Acoustic Heuristics
    if centroid > 3000 and rms > 0.08:
        emotion = "Energetic / Aggressive ⚡"
        desc = "High spectral brightness combined with strong energy."
    elif centroid > 2200 and tempo > 115:
        emotion = "Uplifting / Bright ✨"
        desc = "Fast tempo paired with clean high-frequency presence."
    elif centroid < 1500 and rms < 0.05:
        emotion = "Intimate / Melancholic 🌙"
        desc = "Warm low-end profile with subdued dynamic intensity."
    else:
        emotion = "Neutral / Balanced 🎤"
        desc = "Balanced frequency distribution and moderate dynamics."
        
    return emotion, desc, tempo


def calculate_mix_score(peak_db, noise_db, dynamic_range):
    """Scores the technical quality out of 100 based on standard audio standards."""
    score = 100
    feedback = []

    # Gain Staging Check
    if peak_db > -0.1:
        score -= 25
        feedback.append("❌ Digital clipping detected! Reduce track gain.")
    elif peak_db < -12.0:
        score -= 10
        feedback.append("⚠️ Track level is low; boost gain for optimal resolution.")
    else:
        feedback.append("✅ Peak level gain staging is within a healthy range.")

    # Noise Floor Check
    if noise_db > -45.0:
        score -= 20
        feedback.append("❌ High noise floor detected. Consider applying a noise gate or de-noiser.")
    else:
        feedback.append("✅ Clean background noise floor.")

    # Dynamic Range Check
    if dynamic_range < 6.0:
        score -= 20
        feedback.append("⚠️ Over-compressed dynamic range (Loudness War territory).")
    elif dynamic_range > 20.0:
        score -= 10
        feedback.append("ℹ️ Wide dynamic range; automated leveling or light compression recommended.")
    else:
        feedback.append("✅ Healthy balance of dynamics and loudness consistency.")

    return max(0, score), feedback


if uploaded_file is not None:
    st.audio(uploaded_file, format="audio/wav")

    with st.spinner("Analyzing DSP signals and acoustic properties..."):
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name

        try:
            y, sr = librosa.load(tmp_path, sr=None, mono=True)
            duration = librosa.get_duration(y=y, sr=sr)

            # DSP Computations
            peak_db, rec_gain = analyze_gain_staging(y)
            noise_db = analyze_noise(y, sr)
            rms_db, dynamic_range = analyze_loudness_and_dynamics(y)
            emotion, emotion_desc, tempo = analyze_vocal_emotion_heuristic(y, sr)
            mix_score, mix_feedback = calculate_mix_score(peak_db, noise_db, dynamic_range)

            st.markdown("---")
            
            # --- OVERALL MIX SCORE ---
            col_score, col_status = st.columns([1, 2])
            with col_score:
                st.metric("Technical Mix Quality Score", f"{mix_score} / 100")
            with col_status:
                st.subheader("Diagnostic Summary")
                for item in mix_feedback:
                    st.write(item)

            st.markdown("---")

            # --- DETAILED METRICS GRID ---
            st.subheader("📊 Detailed Audio Signal Analysis")
            
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Peak Level", f"{peak_db:.1f} dBFS", delta=f"{rec_gain:+.1f} dB Rec")
            col2.metric("Loudness (RMS)", f"{rms_db:.1f} dBFS")
            col3.metric("Estimated Noise Floor", f"{noise_db:.1f} dBFS")
            col4.metric("Dynamic Range", f"{dynamic_range:.1f} dB")

            st.markdown("---")

            # --- EMOTION & CHARACTERISTICS ---
            st.subheader("🎭 Acoustic & Aesthetic Analysis")
            col_emo, col_bpm = st.columns(2)
            with col_emo:
                st.info(f"**Vibe / Emotion Profile:** {emotion}\n\n*{emotion_desc}*")
            with col_bpm:
                st.success(f"**Detected Tempo:** {tempo:.1f} BPM\n\n**Duration:** {duration:.1f} seconds")

        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

"""
Voice Analysis Pro - Professional Voice & Singing Analysis Tool
===============================================================
Single-file Streamlit app for comprehensive vocal analysis.

Features: Pitch detection, range analysis, voice type classification,
timbre profiling, and personalized song recommendations.

Author: Claude | Version: 1.1 | Python: 3.10+
"""

import streamlit as st
import numpy as np
import pandas as pd
import librosa
import soundfile as sf
from scipy import signal
import json
import io
import os
import logging
from typing import Dict, Tuple
import warnings
warnings.filterwarnings('ignore')

import plotly.graph_objects as go

# Setup logging
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

# Optional dependencies
try:
    import torch
    import torchcrepe
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch/torchcrepe not available, using librosa YIN")

try:
    from streamlit_webrtc import webrtc_streamer
    WEBRTC_AVAILABLE = True
except ImportError:
    WEBRTC_AVAILABLE = False
    logger.warning("streamlit-webrtc not available, microphone input disabled")

# Constants
SAMPLE_RATE = 22050
HOP_LENGTH = 512
MIN_FREQUENCY = 65
MAX_FREQUENCY = 2093

VOICE_TYPES = {
    'Bass': {'min': 40, 'max': 64, 'tessitura': (45, 57)},
    'Baritone': {'min': 45, 'max': 69, 'tessitura': (50, 62)},
    'Tenor': {'min': 48, 'max': 72, 'tessitura': (55, 67)},
    'Alto': {'min': 53, 'max': 77, 'tessitura': (60, 72)},
    'Mezzo-Soprano': {'min': 57, 'max': 81, 'tessitura': (64, 76)},
    'Soprano': {'min': 60, 'max': 84, 'tessitura': (67, 79)}
}

CATALOG_FILE = "song_catalog.csv"

# Utility Functions
def hz_to_midi(hz: float) -> float:
    """Convert Hz to MIDI note number."""
    if hz <= 0:
        return 0
    return 12 * np.log2(hz / 440.0) + 69

def midi_to_hz(midi: float) -> float:
    """Convert MIDI to Hz."""
    return 440.0 * (2.0 ** ((midi - 69) / 12.0))

def midi_to_note_name(midi: float) -> str:
    """Convert MIDI to note name with octave."""
    if midi <= 0:
        return "N/A"
    note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    note_num = int(round(midi))
    octave = (note_num // 12) - 1
    note = note_names[note_num % 12]
    return f"{note}{octave}"

def cents_from_midi(hz: float, ref_midi: float) -> float:
    """Calculate cents deviation from reference."""
    if hz <= 0:
        return 0
    return (hz_to_midi(hz) - ref_midi) * 100

def smooth_pitch(pitch: np.ndarray, confidence: np.ndarray, window_size: int = 5, conf_threshold: float = 0.3) -> np.ndarray:
    """Smooth pitch with median filter."""
    pitch_masked = pitch.copy()
    pitch_masked[confidence < conf_threshold] = 0
    smoothed = signal.medfilt(pitch_masked, kernel_size=window_size)
    return smoothed

def compute_spectral_features(y: np.ndarray, sr: int) -> Dict:
    """Compute spectral features for timbre analysis."""
    D = np.abs(librosa.stft(y, hop_length=HOP_LENGTH))
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=HOP_LENGTH)[0]
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, hop_length=HOP_LENGTH)[0]
    
    freqs = librosa.fft_frequencies(sr=sr)
    low_band = (freqs < 300)
    mid_band = (freqs >= 300) & (freqs < 3000)
    high_band = (freqs >= 3000)
    
    low_energy = np.mean(D[low_band, :])
    mid_energy = np.mean(D[mid_band, :])
    high_energy = np.mean(D[high_band, :])
    total_energy = low_energy + mid_energy + high_energy
    
    return {
        'centroid_mean': float(np.mean(centroid)),
        'rolloff_mean': float(np.mean(rolloff)),
        'low_energy_ratio': float(low_energy / total_energy if total_energy > 0 else 0),
        'mid_energy_ratio': float(mid_energy / total_energy if total_energy > 0 else 0),
        'high_energy_ratio': float(high_energy / total_energy if total_energy > 0 else 0)
    }

def classify_timbre(spectral_features: Dict) -> str:
    """Classify timbre."""
    low = spectral_features['low_energy_ratio']
    mid = spectral_features['mid_energy_ratio']
    high = spectral_features['high_energy_ratio']
    
    if low > 0.4:
        return "Bass-heavy"
    elif high > 0.35:
        return "Treble-bright"
    elif mid > 0.5:
        return "Mid-forward"
    else:
        return "Balanced"

def detect_pitch_crepe(y: np.ndarray, sr: int) -> Tuple[np.ndarray, np.ndarray]:
    """Detect pitch using CREPE."""
    if not TORCH_AVAILABLE:
        raise ImportError("CREPE not available")
    
    y_16k = librosa.resample(y, orig_sr=sr, target_sr=16000) if sr != 16000 else y
    audio_tensor = torch.from_numpy(y_16k).float().unsqueeze(0)
    
    with torch.no_grad():
        pitch, confidence = torchcrepe.predict(
            audio_tensor, 16000, hop_length=160,
            fmin=MIN_FREQUENCY, fmax=MAX_FREQUENCY,
            model='tiny', device='cpu', return_periodicity=True
        )
    
    return pitch.squeeze().numpy(), confidence.squeeze().numpy()

def detect_pitch_yin(y: np.ndarray, sr: int) -> Tuple[np.ndarray, np.ndarray]:
    """Detect pitch using YIN."""
    pitch = librosa.yin(y, fmin=MIN_FREQUENCY, fmax=MAX_FREQUENCY, sr=sr, hop_length=HOP_LENGTH)
    rms = librosa.feature.rms(y=y, hop_length=HOP_LENGTH)[0]
    confidence = np.clip(rms / np.max(rms) if np.max(rms) > 0 else rms, 0, 1)
    min_len = min(len(pitch), len(confidence))
    return pitch[:min_len], confidence[:min_len]

def classify_voice_type(midi_notes: np.ndarray, tessitura_range: Tuple[float, float]) -> Dict:
    """Classify voice type."""
    min_note = float(np.min(midi_notes))
    max_note = float(np.max(midi_notes))
    tess_low, tess_high = float(tessitura_range[0]), float(tessitura_range[1])
    
    scores = {}
    for voice_type, ranges in VOICE_TYPES.items():
        range_overlap = (min(max_note, ranges['max']) - max(min_note, ranges['min'])) / (ranges['max'] - ranges['min'])
        tess_overlap = (min(tess_high, ranges['tessitura'][1]) - max(tess_low, ranges['tessitura'][0])) / (ranges['tessitura'][1] - ranges['tessitura'][0])
        scores[voice_type] = float(range_overlap * 0.4 + tess_overlap * 0.6)
    
    sorted_types = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    best_type, best_score = sorted_types[0]
    is_borderline = len(sorted_types) > 1 and sorted_types[1][1] > best_score * 0.85
    
    return {
        'type': best_type,
        'confidence': float(best_score),
        'is_borderline': bool(is_borderline),
        'alternative': sorted_types[1][0] if is_borderline else None,
        'all_scores': {k: float(v) for k, v in sorted_types}
    }

def analyze_audio(y: np.ndarray, sr: int, progress_callback=None, voicing_threshold: float = 0.3) -> Dict:
    """Main analysis pipeline."""
    results = {}
    
    if progress_callback:
        progress_callback("Preprocessing...", 0.1)
    
    # Preprocess
    if sr != SAMPLE_RATE:
        y = librosa.resample(y, orig_sr=sr, target_sr=SAMPLE_RATE)
        sr = SAMPLE_RATE
    
    if len(y.shape) > 1:
        y = librosa.to_mono(y)
    
    y = librosa.util.normalize(y)
    results['duration'] = float(len(y) / sr)
    results['sample_rate'] = int(sr)
    
    if progress_callback:
        progress_callback("Detecting pitch...", 0.3)
    
    # Pitch detection
    try:
        if TORCH_AVAILABLE:
            pitch, confidence = detect_pitch_crepe(y, sr)
            results['pitch_method'] = 'CREPE'
        else:
            pitch, confidence = detect_pitch_yin(y, sr)
            results['pitch_method'] = 'YIN'
    except Exception as e:
        logger.warning(f"torchcrepe failed, falling back to librosa: {e}")
        pitch, confidence = detect_pitch_yin(y, sr)
        results['pitch_method'] = 'YIN (fallback)'
    
    pitch_smoothed = smooth_pitch(pitch, confidence, conf_threshold=voicing_threshold)
    voiced_mask = (pitch_smoothed > 0) & (confidence > voicing_threshold)
    voiced_pitch = pitch_smoothed[voiced_mask]
    
    if len(voiced_pitch) < 10:
        return {
            'error': f'Insufficient voice detected ({len(voiced_pitch)} frames). Try lowering voicing threshold to {voicing_threshold-0.1:.1f}',
            'voiced_ratio': float(np.sum(voiced_mask) / len(voiced_mask) if len(voiced_mask) > 0 else 0)
        }
    
    results['voiced_ratio'] = float(np.sum(voiced_mask) / len(voiced_mask))
    
    if progress_callback:
        progress_callback("Analyzing range...", 0.5)
    
    # Convert to MIDI
    midi_notes = np.array([hz_to_midi(f) for f in voiced_pitch if f > 0])
    
    if len(midi_notes) == 0:
        return {'error': 'No valid pitches detected'}
    
    results['min_note_midi'] = float(np.min(midi_notes))
    results['max_note_midi'] = float(np.max(midi_notes))
    results['min_note'] = midi_to_note_name(results['min_note_midi'])
    results['max_note'] = midi_to_note_name(results['max_note_midi'])
    
    tess_low = float(np.percentile(midi_notes, 25))
    tess_high = float(np.percentile(midi_notes, 75))
    results['tessitura_low_midi'] = tess_low
    results['tessitura_high_midi'] = tess_high
    results['tessitura_low'] = midi_to_note_name(tess_low)
    results['tessitura_high'] = midi_to_note_name(tess_high)
    
    if progress_callback:
        progress_callback("Classifying voice...", 0.6)
    
    results['voice_type'] = classify_voice_type(midi_notes, (tess_low, tess_high))
    
    if progress_callback:
        progress_callback("Pitch accuracy...", 0.7)
    
    # Pitch accuracy
    cents_errors = [cents_from_midi(hz, round(hz_to_midi(hz))) for hz in voiced_pitch]
    
    results['pitch_accuracy'] = {
        'mean_cents_error': float(np.mean(np.abs(cents_errors))),
        'median_cents_error': float(np.median(cents_errors)),
        'std_cents_error': float(np.std(cents_errors)),
        'intonation_score': float(max(0, 100 - np.mean(np.abs(cents_errors))))
    }
    
    if progress_callback:
        progress_callback("Timbre analysis...", 0.8)
    
    spectral_features = compute_spectral_features(y, sr)
    results['spectral_features'] = spectral_features
    results['timbre_classification'] = classify_timbre(spectral_features)
    
    # Note distribution (limit size!)
    note_histogram = {}
    for midi in midi_notes[::10]:  # Sample every 10th note
        note_name = midi_to_note_name(midi)
        note_histogram[note_name] = note_histogram.get(note_name, 0) + 1
    
    results['note_distribution'] = note_histogram
    
    # Store DOWNSAMPLED contour to avoid huge data
    downsample = max(1, len(pitch_smoothed) // 1000)  # Max 1000 points
    results['pitch_contour'] = {
        'times': [float(x) for x in (np.arange(len(pitch_smoothed))[::downsample] * HOP_LENGTH / sr).tolist()],
        'pitch_hz': [float(x) for x in pitch_smoothed[::downsample].tolist()],
        'confidence': [float(x) for x in confidence[::downsample].tolist()],
        'voiced_mask': [bool(x) for x in voiced_mask[::downsample].tolist()]
    }
    
    if progress_callback:
        progress_callback("Complete!", 1.0)
    
    return results

# Song Functions
def initialize_song_catalog():
    """Initialize catalog."""
    if not os.path.exists(CATALOG_FILE):
        demo_songs = [
            {'title': 'Amazing Grace', 'artist': 'Traditional', 'key': 'G', 'typical_low': 'G3', 'typical_high': 'D5', 'tags': 'hymn,mid-forward', 'difficulty': 'beginner'},
            {'title': 'Happy Birthday', 'artist': 'Traditional', 'key': 'F', 'typical_low': 'F3', 'typical_high': 'F4', 'tags': 'celebration,balanced', 'difficulty': 'beginner'},
            {'title': 'Hallelujah', 'artist': 'Leonard Cohen', 'key': 'C', 'typical_low': 'C3', 'typical_high': 'C5', 'tags': 'ballad,mid-forward', 'difficulty': 'intermediate'},
            {'title': 'Somewhere Over Rainbow', 'artist': 'Harold Arlen', 'key': 'Eb', 'typical_low': 'Eb3', 'typical_high': 'Bb4', 'tags': 'classic,balanced', 'difficulty': 'intermediate'},
            {'title': 'Ave Maria', 'artist': 'Schubert', 'key': 'Bb', 'typical_low': 'F3', 'typical_high': 'Ab5', 'tags': 'classical,treble-bright', 'difficulty': 'advanced'},
            {'title': 'Danny Boy', 'artist': 'Traditional', 'key': 'C', 'typical_low': 'C3', 'typical_high': 'D5', 'tags': 'folk,balanced', 'difficulty': 'intermediate'},
            {'title': 'Edelweiss', 'artist': 'R&H', 'key': 'Bb', 'typical_low': 'Bb3', 'typical_high': 'Eb4', 'tags': 'gentle,mid-forward', 'difficulty': 'beginner'},
            {'title': 'What a Wonderful World', 'artist': 'Armstrong', 'key': 'F', 'typical_low': 'F3', 'typical_high': 'F4', 'tags': 'jazz,bass-heavy', 'difficulty': 'beginner'}
        ]
        pd.DataFrame(demo_songs).to_csv(CATALOG_FILE, index=False)
    return pd.read_csv(CATALOG_FILE)

def recommend_songs(analysis_results: Dict, catalog: pd.DataFrame) -> Dict:
    """Recommend songs."""
    if 'error' in analysis_results:
        return {'fit': [], 'stretch': [], 'avoid': []}
    
    user_tess_low = analysis_results['tessitura_low_midi']
    user_tess_high = analysis_results['tessitura_high_midi']
    user_timbre = analysis_results['timbre_classification'].lower()
    
    fit_songs, stretch_songs, avoid_songs = [], [], []
    
    for _, song in catalog.iterrows():
        try:
            song_low = hz_to_midi(librosa.note_to_hz(song['typical_low']))
            song_high = hz_to_midi(librosa.note_to_hz(song['typical_high']))
            
            rec = {
                'title': song['title'],
                'artist': song['artist'],
                'key': song['key'],
                'range': f"{song['typical_low']}-{song['typical_high']}",
                'difficulty': song['difficulty'],
                'reasons': []
            }
            
            if song_low >= user_tess_low - 1 and song_high <= user_tess_high + 1:
                rec['reasons'].append(f"Fits your tessitura")
                if user_timbre in song['tags'].lower():
                    rec['reasons'].append(f"Matches {user_timbre} timbre")
                fit_songs.append(rec)
            elif song_high <= user_tess_high + 4 and song_low >= user_tess_low - 4:
                rec['reasons'].append("Safe stretch for growth")
                stretch_songs.append(rec)
            else:
                if song_high > user_tess_high + 4:
                    rec['reasons'].append("Too high for comfort")
                if song_low < user_tess_low - 4:
                    rec['reasons'].append("Too low for comfort")
                avoid_songs.append(rec)
        except:
            continue
    
    return {'fit': fit_songs[:10], 'stretch': stretch_songs[:5], 'avoid': avoid_songs[:5]}

# Visualization
def plot_pitch_contour(analysis_results: Dict) -> go.Figure:
    """Plot pitch contour."""
    contour = analysis_results['pitch_contour']
    times = np.array(contour['times'])
    pitch = np.array(contour['pitch_hz'])
    voiced = np.array(contour['voiced_mask'])
    
    voiced_times = times[voiced]
    voiced_pitch = pitch[voiced]
    
    if len(voiced_pitch) == 0:
        return None
    
    voiced_midi = [hz_to_midi(f) for f in voiced_pitch]
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=voiced_times, y=voiced_pitch, mode='lines',
        name='Pitch', line=dict(color='#3b82f6', width=2),
        hovertemplate='<b>Time:</b> %{x:.2f}s<br><b>Hz:</b> %{y:.1f}<br><b>Note:</b> %{text}<extra></extra>',
        text=[midi_to_note_name(m) for m in voiced_midi]
    ))
    
    tess_low_hz = midi_to_hz(analysis_results['tessitura_low_midi'])
    tess_high_hz = midi_to_hz(analysis_results['tessitura_high_midi'])
    
    fig.add_hrect(y0=tess_low_hz, y1=tess_high_hz, fillcolor='green', opacity=0.1, line_width=0)
    fig.update_layout(title='Pitch Over Time', xaxis_title='Time (s)', yaxis_title='Hz', height=400, template='plotly_white')
    
    return fig

def plot_note_distribution(analysis_results: Dict) -> go.Figure:
    """Plot note distribution."""
    note_dist = analysis_results['note_distribution']
    notes, counts = list(note_dist.keys()), list(note_dist.values())
    
    fig = go.Figure([go.Bar(x=notes, y=counts, marker_color='#8b5cf6')])
    fig.update_layout(title='Note Distribution', xaxis_title='Note', yaxis_title='Count', height=350, template='plotly_white')
    return fig

def plot_spectral_profile(analysis_results: Dict) -> go.Figure:
    """Plot spectral profile."""
    features = analysis_results['spectral_features']
    bands = ['Low', 'Mid', 'High']
    energies = [features['low_energy_ratio']*100, features['mid_energy_ratio']*100, features['high_energy_ratio']*100]
    colors = ['#ef4444', '#f59e0b', '#3b82f6']
    
    fig = go.Figure([go.Bar(x=bands, y=energies, marker_color=colors, text=[f'{e:.1f}%' for e in energies], textposition='auto')])
    fig.update_layout(title='Timbre Profile', yaxis_title='Energy %', height=350, template='plotly_white', showlegend=False)
    return fig

# UI
def apply_custom_css():
    """Apply styling."""
    st.markdown("""
    <style>
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem; border-radius: 12px; color: white;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1); margin: 0.5rem 0;
    }
    .info-card {
        background: #f8fafc; padding: 1.5rem; border-radius: 12px;
        border-left: 4px solid #3b82f6; margin: 1rem 0;
    }
    h1 {
        background: linear-gradient(120deg, #667eea, #764ba2);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    }
    </style>
    """, unsafe_allow_html=True)

def main():
    """Main app."""
    st.set_page_config(page_title="Voice Analysis Pro", page_icon="🎤", layout="wide")
    
    apply_custom_css()
    
    if 'analysis_results' not in st.session_state:
        st.session_state.analysis_results = None
    if 'audio_data' not in st.session_state:
        st.session_state.audio_data = None
    if 'catalog' not in st.session_state:
        st.session_state.catalog = initialize_song_catalog()
    
    st.markdown("<div style='text-align:center; padding:2rem 0;'><h1 style='font-size:3rem;'>🎤 Voice Analysis Pro</h1><p style='font-size:1.2rem;color:#64748b;'>Professional vocal analysis • 100% Local</p></div>", unsafe_allow_html=True)
    
    with st.sidebar:
        st.markdown("### ⚙️ Settings")
        st.markdown("**Voicing Sensitivity**")
        voicing_threshold = st.slider("Threshold", 0.0, 1.0, 0.25, 0.05, help="Lower = more sensitive", label_visibility="collapsed")
        
        if voicing_threshold < 0.3:
            st.caption("🔊 High sensitivity")
        elif voicing_threshold > 0.6:
            st.caption("🔇 Low sensitivity")
        else:
            st.caption("✓ Balanced")
        
        st.markdown("---")
        st.info("""
        **Voice Analysis Pro**
        - 🎵 Vocal range
        - 🎭 Voice type
        - 🎯 Pitch accuracy
        - 🎨 Timbre profile
        - 📊 Song recommendations
        
        **100% Private & Local**
        """)
        
        with st.expander("System Info"):
            st.text(f"Pitch: {'CREPE' if TORCH_AVAILABLE else 'YIN'}\nWebRTC: {'Yes' if WEBRTC_AVAILABLE else 'No'}")
    
    tab1, tab2, tab3, tab4 = st.tabs(["🎙️ Record/Upload", "🔍 Analysis", "📈 Charts", "🎵 Recommendations"])
    
    with tab1:
        st.markdown("## Input Your Voice")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 📤 Upload Audio")
            uploaded_file = st.file_uploader("Choose audio file", type=['wav', 'mp3', 'm4a', 'flac'])
            
            if uploaded_file:
                try:
                    audio_bytes = uploaded_file.read()
                    y, sr = librosa.load(io.BytesIO(audio_bytes), sr=None)
                    st.session_state.audio_data = (y, sr)
                    st.success(f"✅ Loaded: {uploaded_file.name}")
                    st.info(f"Duration: {len(y)/sr:.2f}s | SR: {sr}Hz")
                except Exception as e:
                    logger.error(f"Error loading audio: {e}")
                    st.error(f"❌ Error: {e}")
        
        with col2:
            st.markdown("### 🎙️ Live Recording")
            if not WEBRTC_AVAILABLE:
                st.warning("⚠️ Install `streamlit-webrtc` for live recording")
            else:
                st.info("🎤 WebRTC recording available")
        
        st.markdown("---")
        col_analyze, col_demo = st.columns([3, 1])
        
        with col_analyze:
            if st.button("🔬 Analyze Voice", type="primary", use_container_width=True):
                if not st.session_state.audio_data:
                    st.error("❌ Upload audio first!")
                else:
                    y, sr = st.session_state.audio_data
                    progress_bar = st.progress(0)
                    status = st.empty()
                    
                    def update_progress(msg, val):
                        status.text(msg)
                        progress_bar.progress(val)
                    
                    try:
                        results = analyze_audio(y, sr, update_progress, voicing_threshold)
                        
                        if 'error' in results:
                            st.error(f"❌ {results['error']}")
                            st.info(f"💡 Voiced: {results.get('voiced_ratio', 0)*100:.1f}%. Lower threshold in sidebar.")
                        else:
                            st.session_state.analysis_results = results
                            st.success("✅ Analysis complete!")
                            st.balloons()
                    except Exception as e:
                        logger.error(f"Analysis failed: {e}")
                        st.error(f"❌ Failed: {e}")
                    finally:
                        progress_bar.empty()
                        status.empty()
        
        with col_demo:
            if st.button("🧪 Demo", use_container_width=True):
                sr = 22050
                t = np.linspace(0, 3, sr*3)
                freqs = [261.63, 293.66, 329.63, 349.23, 392.00, 440.00, 493.88, 523.25]
                y = np.concatenate([0.3 * np.sin(2*np.pi*f*np.linspace(0, 0.3, int(sr*0.3))) * np.exp(-np.linspace(0, 0.3, int(sr*0.3))*2) for f in freqs])
                st.session_state.audio_data = (y, sr)
                st.success("✅ Demo audio ready!")
    
    with tab2:
        st.markdown("## Analysis Results")
        
        if not st.session_state.analysis_results:
            st.info("👆 Analyze audio first")
        else:
            results = st.session_state.analysis_results
            
            if 'error' in results:
                st.error(results['error'])
            else:
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.markdown(f"<div class='metric-card'><h3 style='color:white;margin:0;'>🎵 Range</h3><p style='font-size:1.8rem;font-weight:bold;margin:0.5rem 0;'>{results['min_note']}-{results['max_note']}</p><p style='margin:0;opacity:0.9;'>{int(results['max_note_midi']-results['min_note_midi'])} semitones</p></div>", unsafe_allow_html=True)
                
                with col2:
                    voice = results['voice_type']

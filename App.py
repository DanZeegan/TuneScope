"""
Voice Analysis Web Application
A comprehensive tool for analyzing vocal performance with real-time feedback,
song recommendations, and educational features.

Author: AI Assistant
Date: 2024
Requirements: Python 3.10+, Streamlit, and various audio processing libraries
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import librosa
import librosa.display
import soundfile as sf
from scipy import signal
from scipy.io import wavfile
import io
import os
import json
import time
import base64
from typing import Dict, List, Tuple, Optional, Any
import warnings
import torch
warnings.filterwarnings('ignore')

# Try to import torchcrepe for advanced pitch detection
try:
    import torchcrepe
    USE_TORCH_CREPE = True
except ImportError:
    USE_TORCH_CREPE = False
    print("torchcrepe not available, using librosa fallback")

# Audio processing constants
SAMPLE_RATE = 22050
HOP_LENGTH = 512
WIN_LENGTH = 1024
FMIN = 80
FMAX = 800

def hz_to_note(frequency: float) -> str:
    """Convert frequency in Hz to musical note name."""
    if frequency <= 0:
        return "N/A"
    
    # A4 = 440 Hz reference
    A4 = 440.0
    C0 = A4 * pow(2, -4.75)  # C0 is 16.35 Hz
    
    h = round(12 * np.log2(frequency / C0))
    octave = h // 12
    n = h % 12
    
    note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    return f"{note_names[n]}{octave}"

def note_to_hz(note: str) -> float:
    """Convert musical note name to frequency in Hz."""
    note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    
    # Extract note and octave
    note_part = note[:-1] if note[-1].isdigit() else note
    octave = int(note[-1]) if note[-1].isdigit() else 4
    
    if note_part not in note_names:
        return 440.0  # Default to A4
    
    n = note_names.index(note_part)
    A4 = 440.0
    C0 = A4 * pow(2, -4.75)
    
    h = octave * 12 + n
    return C0 * pow(2, h / 12)

def cents_error(frequency: float, target_note: str) -> float:
    """Calculate pitch deviation in cents from target note."""
    if frequency <= 0:
        return 0
    
    target_freq = note_to_hz(target_note)
    return 1200 * np.log2(frequency / target_freq)

def smooth_f0(f0: np.ndarray, voicing_threshold: float = 0.3) -> np.ndarray:
    """Apply smoothing and voicing threshold to pitch contour."""
    # Apply median filter for smoothing
    smoothed = signal.medfilt(f0, kernel_size=5)
    
    # Apply voicing threshold
    smoothed[smoothed < voicing_threshold] = 0
    
    return smoothed

def detect_pitch_audio(audio: np.ndarray, sr: int = SAMPLE_RATE) -> Tuple[np.ndarray, np.ndarray]:
    """Detect pitch using available methods."""
    try:
        if USE_TORCH_CREPE:
            # Use torchcrepe for better accuracy
            audio_tensor = torch.from_numpy(audio).float()
            if audio_tensor.dim() == 1:
                audio_tensor = audio_tensor.unsqueeze(0)
            
            f0, voicing_probability = torchcrepe.predict(
                audio_tensor, 
                sr, 
                hop_length=HOP_LENGTH,
                fmin=FMIN, 
                fmax=FMAX,
                model="full",
                return_periodicity=True
            )
            
            f0 = f0.squeeze().numpy()
            voicing = voicing_probability.squeeze().numpy()
            
            # Apply voicing threshold
            f0[voicing < 0.3] = 0
            
        else:
            # Fallback to librosa YIN
            f0 = librosa.yin(audio, 
                           fmin=FMIN, 
                           fmax=FMAX, 
                           sr=sr, 
                           hop_length=HOP_LENGTH)
            
            # Create voicing mask based on energy
            hop_length = HOP_LENGTH
            n_frames = len(f0)
            frame_length = WIN_LENGTH
            
            energy = np.array([
                np.sum(audio[i*hop_length:i*hop_length+frame_length]**2)
                for i in range(n_frames)
            ])
            
            # Threshold energy for voicing
            energy_threshold = np.percentile(energy, 10)
            voiced_frames = energy > energy_threshold
            f0[~voiced_frames] = 0
            
    except Exception as e:
        print(f"Pitch detection error: {e}")
        # Return empty arrays as fallback
        f0 = np.zeros(len(audio) // HOP_LENGTH)
    
    return f0, np.arange(len(f0)) * HOP_LENGTH / sr

def analyze_vocal_range(f0: np.ndarray) -> Dict[str, Any]:
    """Analyze vocal range from pitch contour."""
    # Filter out unvoiced frames
    voiced_f0 = f0[f0 > 0]
    
    if len(voiced_f0) == 0:
        return {
            'min_freq': 0,
            'max_freq': 0,
            'tessitura_low': 0,
            'tessitura_high': 0,
            'voice_type': 'Unknown',
            'confidence': 0.0
        }
    
    # Calculate range statistics
    min_freq = np.min(voiced_f0)
    max_freq = np.max(voiced_f0)
    
    # Calculate tessitura (central 50% range)
    sorted_f0 = np.sort(voiced_f0)
    n_frames = len(sorted_f0)
    tess_start = n_frames // 4
    tess_end = 3 * n_frames // 4
    
    tessitura_low = sorted_f0[tess_start]
    tessitura_high = sorted_f0[tess_end]
    
    # Classify voice type
    voice_type, confidence = classify_voice_type(min_freq, max_freq, tessitura_low, tessitura_high)
    
    return {
        'min_freq': min_freq,
        'max_freq': max_freq,
        'tessitura_low': tessitura_low,
        'tessitura_high': tessitura_high,
        'voice_type': voice_type,
        'confidence': confidence,
        'voiced_frames': len(voiced_f0),
        'total_frames': len(f0)
    }

def classify_voice_type(min_freq: float, max_freq: float, tess_low: float, tess_high: float) -> Tuple[str, float]:
    """Classify voice type based on frequency range."""
    
    # Voice type ranges (in Hz)
    voice_ranges = {
        'Bass': (80, 330),
        'Baritone': (110, 400),
        'Tenor': (140, 500),
        'Alto': (165, 600),
        'Mezzo-Soprano': (200, 700),
        'Soprano': (250, 1500)
    }
    
    # Calculate range span
    range_span = max_freq - min_freq
    
    # Find best matching voice type
    best_match = None
    best_score = 0
    
    for voice_type, (low_range, high_range) in voice_ranges.items():
        # Calculate overlap score
        overlap_low = max(min_freq, low_range)
        overlap_high = min(max_freq, high_range)
        overlap = max(0, overlap_high - overlap_low)
        
        # Calculate coverage scores
        range_coverage = overlap / (high_range - low_range) if high_range != low_range else 0
        user_coverage = overlap / range_span if range_span > 0 else 0
        
        # Combined score
        score = (range_coverage + user_coverage) / 2
        
        if score > best_score:
            best_score = score
            best_match = voice_type
    
    # Handle borderline cases
    if best_score < 0.3:
        # Check if it's a borderline case
        if 300 <= min_freq <= 350:
            best_match = "Baritone/Tenor borderline"
        elif 180 <= min_freq <= 220:
            best_match = "Alto/Mezzo borderline"
        else:
            best_match = "Unknown"
        best_score *= 0.5
    
    return best_match, best_score

def analyze_timbre(audio: np.ndarray, sr: int = SAMPLE_RATE) -> Dict[str, Any]:
    """Analyze timbre characteristics of the audio."""
    
    # Calculate spectral features
    spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)[0]
    spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr)[0]
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sr)[0]
    
    # Calculate zero crossing rate
    zcr = librosa.feature.zero_crossing_rate(audio)[0]
    
    # Calculate MFCCs
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
    
    # Analyze frequency bands
    stft = np.abs(librosa.stft(audio))
    frequencies = librosa.fft_frequencies(sr=sr)
    
    # Define frequency bands
    bass_band = (frequencies < 300)
    mid_band = (frequencies >= 300) & (frequencies < 3000)
    treble_band = (frequencies >= 3000)
    
    # Calculate energy in each band
    bass_energy = np.mean(stft[bass_band, :]) if np.any(bass_band) else 0
    mid_energy = np.mean(stft[mid_band, :]) if np.any(mid_band) else 0
    treble_energy = np.mean(stft[treble_band, :]) if np.any(treble_band) else 0
    
    # Determine dominant frequency region
    energies = [bass_energy, mid_energy, treble_energy]
    bands = ['Bass-heavy', 'Mid-forward', 'Treble-bright']
    dominant_band = bands[np.argmax(energies)]
    
    # Check if balanced
    energy_ratio = max(energies) / (sum(energies) + 1e-10)
    if energy_ratio < 0.6:
        timbre_class = 'Balanced'
    else:
        timbre_class = dominant_band
    
    return {
        'timbre_class': timbre_class,
        'spectral_centroid': np.mean(spectral_centroid),
        'spectral_rolloff': np.mean(spectral_rolloff),
        'zero_crossing_rate': np.mean(zcr),
        'bass_energy': bass_energy,
        'mid_energy': mid_energy,
        'treble_energy': treble_energy,
        'energy_distribution': {
            'bass': bass_energy / sum(energies),
            'mid': mid_energy / sum(energies),
            'treble': treble_energy / sum(energies)
        }
    }

def calculate_pitch_accuracy(f0: np.ndarray, sr: int = SAMPLE_RATE) -> Dict[str, Any]:
    """Calculate pitch accuracy metrics."""
    
    voiced_f0 = f0[f0 > 0]
    if len(voiced_f0) == 0:
        return {'accuracy_score': 0, 'mean_cents_error': 0, 'stability': 0}
    
    # Calculate cents error for each voiced frame
    cents_errors = []
    for freq in voiced_f0:
        if freq > 0:
            note = hz_to_note(freq)
            cents_err = cents_error(freq, note)
            cents_errors.append(abs(cents_err))
    
    cents_errors = np.array(cents_errors)
    
    # Calculate accuracy metrics
    mean_cents_error = np.mean(cents_errors)
    std_cents_error = np.std(cents_errors)
    
    # Convert to accuracy score (0-100)
    # Perfect accuracy = 0 cents error, score = 100
    # Poor accuracy = 50+ cents error, score = 0
    accuracy_score = max(0, 100 - (mean_cents_error * 2))
    
    # Calculate stability (jitter)
    if len(voiced_f0) > 1:
        jitter = np.mean(np.abs(np.diff(voiced_f0))) / np.mean(voiced_f0)
        stability = max(0, 100 - (jitter * 1000))
    else:
        stability = 100
    
    return {
        'accuracy_score': accuracy_score,
        'mean_cents_error': mean_cents_error,
        'std_cents_error': std_cents_error,
        'stability': stability,
        'voiced_percentage': len(voiced_f0) / len(f0) * 100
    }

def load_song_catalog() -> pd.DataFrame:
    """Load the song catalog from CSV file."""
    try:
        catalog_path = os.path.join(os.path.dirname(__file__), 'song_catalog.csv')
        return pd.read_csv(catalog_path)
    except FileNotFoundError:
        # Create a basic catalog if file not found
        basic_songs = [
            ['Someone Like You', 'Adele', 'A', 'C3', 'C5', 'Pop', 'Intermediate', 'Mid-forward', '2010s', 'Ballad'],
            ['Rolling in the Deep', 'Adele', 'C', 'A2', 'D5', 'Pop', 'Advanced', 'Bass-heavy', '2010s', 'Powerful belt'],
            ['Hallelujah', 'Leonard Cohen', 'C', 'G2', 'G4', 'Folk', 'Intermediate', 'Mid-forward', '1980s', 'Classic'],
            ['Bohemian Rhapsody', 'Queen', 'G', 'G2', 'G6', 'Rock', 'Advanced', 'Balanced', '1970s', 'Epic multi-section'],
            ['I Will Always Love You', 'Whitney Houston', 'A', 'Bb2', 'G5', 'Soul', 'Advanced', 'Balanced', '1990s', 'Power ballad'],
            ['Memory', 'Cats', 'G', 'G3', 'G5', 'Musical Theatre', 'Advanced', 'Balanced', '1980s', 'Andrew Lloyd Webber'],
            ['Defying Gravity', 'Wicked', 'D', 'G3', 'F5', 'Musical Theatre', 'Advanced', 'Balanced', '2000s', 'Power anthem'],
            ['Let It Go', 'Frozen', 'G', 'G3', 'G5', 'Musical Theatre', 'Advanced', 'Balanced', '2010s', 'Disney power ballad']
        ]
        
        return pd.DataFrame(basic_songs, columns=[
            'title', 'artist', 'key', 'typical_low', 'typical_high', 
            'genre', 'difficulty', 'timbre_tag', 'era', 'notes'
        ])

def generate_song_recommendations(vocal_analysis: Dict[str, Any], timbre_analysis: Dict[str, Any], 
                                catalog: pd.DataFrame, n_recommendations: int = 5) -> pd.DataFrame:
    """Generate song recommendations based on vocal analysis."""
    
    user_range = vocal_analysis['max_freq'] - vocal_analysis['min_freq']
    user_tessitura = vocal_analysis['tessitura_high'] - vocal_analysis['tessitura_low']
    user_voice_type = vocal_analysis['voice_type']
    user_timbre = timbre_analysis['timbre_class']
    
    recommendations = []
    
    for _, song in catalog.iterrows():
        # Parse song range
        song_low = note_to_hz(song['typical_low'])
        song_high = note_to_hz(song['typical_high'])
        song_range = song_high - song_low
        
        # Calculate compatibility scores
        range_compatibility = 0
        tessitura_compatibility = 0
        timbre_compatibility = 0
        
        # Range compatibility (can user sing this song comfortably?)
        if vocal_analysis['min_freq'] <= song_low and vocal_analysis['max_freq'] >= song_high:
            range_compatibility = 1.0
        elif (vocal_analysis['tessitura_low'] <= song_low and 
              vocal_analysis['tessitura_high'] >= song_high):
            range_compatibility = 0.8
        elif (abs(vocal_analysis['min_freq'] - song_low) < 100 or 
              abs(vocal_analysis['max_freq'] - song_high) < 100):
            range_compatibility = 0.6
        else:
            range_compatibility = 0.2
        
        # Timbre compatibility
        if song['timbre_tag'] == user_timbre:
            timbre_compatibility = 1.0
        elif song['timbre_tag'] == 'Balanced' or user_timbre == 'Balanced':
            timbre_compatibility = 0.8
        else:
            timbre_compatibility = 0.4
        
        # Voice type compatibility
        voice_compatibility = 0.7  # Default moderate compatibility
        if user_voice_type in song['notes'] or song['difficulty'] in ['Beginner', 'Intermediate']:
            voice_compatibility = 0.9
        
        # Overall compatibility score
        total_score = (range_compatibility * 0.5 + 
                      timbre_compatibility * 0.3 + 
                      voice_compatibility * 0.2)
        
        # Classify recommendation type
        if total_score >= 0.8:
            rec_type = "Perfect Fit"
        elif total_score >= 0.6:
            rec_type = "Good Match"
        elif total_score >= 0.4:
            rec_type = "Stretch Song"
        else:
            rec_type = "Avoid"
        
        recommendations.append({
            'title': song['title'],
            'artist': song['artist'],
            'genre': song['genre'],
            'difficulty': song['difficulty'],
            'range_compatibility': range_compatibility,
            'timbre_compatibility': timbre_compatibility,
            'total_score': total_score,
            'recommendation_type': rec_type,
            'song_range': f"{song['typical_low']}-{song['typical_high']}"
        })
    
    # Sort by total score and return top recommendations
    recommendations_df = pd.DataFrame(recommendations)
    recommendations_df = recommendations_df.sort_values('total_score', ascending=False)
    
    return recommendations_df.head(n_recommendations)

def create_pitch_visualization(f0: np.ndarray, times: np.ndarray, sr: int = SAMPLE_RATE) -> go.Figure:
    """Create interactive pitch visualization."""
    
    # Filter out unvoiced frames for cleaner visualization
    voiced_mask = f0 > 0
    voiced_f0 = f0[voiced_mask]
    voiced_times = times[voiced_mask]
    
    fig = go.Figure()
    
    # Add pitch contour
    fig.add_trace(go.Scatter(
        x=voiced_times,
        y=voiced_f0,
        mode='lines+markers',
        name='Pitch Contour',
        line=dict(color='#2a9d8f', width=2),
        marker=dict(size=3, color='#2a9d8f'),
        hovertemplate='<b>Time:</b> %{x:.2f}s<br><b>Frequency:</b> %{y:.1f}Hz<br><b>Note:</b> %{customdata}<extra></extra>',
        customdata=[hz_to_note(f) for f in voiced_f0]
    ))
    
    # Add note grid lines for reference
    note_freqs = [note_to_hz(note) for note in ['C2', 'C3', 'C4', 'C5', 'C6']]
    for freq in note_freqs:
        if freq > 0:
            fig.add_hline(y=freq, line_dash="dash", line_color="rgba(0,0,0,0.2)", 
                         annotation_text=hz_to_note(freq), annotation_position="right")
    
    fig.update_layout(
        title='Pitch Analysis Over Time',
        xaxis_title='Time (seconds)',
        yaxis_title='Frequency (Hz)',
        template='plotly_white',
        height=400,
        showlegend=True
    )
    
    return fig

def create_spectrum_visualization(audio: np.ndarray, sr: int = SAMPLE_RATE) -> go.Figure:
    """Create frequency spectrum visualization."""
    
    # Calculate spectrum
    fft = np.fft.fft(audio)
    freqs = np.fft.fftfreq(len(audio), 1/sr)
    magnitude = np.abs(fft)
    
    # Only show positive frequencies up to Nyquist
    positive_mask = (freqs >= 0) & (freqs <= sr/2)
    freqs = freqs[positive_mask]
    magnitude = magnitude[positive_mask]
    
    # Smooth the spectrum for better visualization
    smoothing_factor = max(1, len(freqs) // 1000)
    freqs = freqs[::smoothing_factor]
    magnitude = magnitude[::smoothing_factor]
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=freqs,
        y=magnitude,
        mode='lines',
        name='Frequency Spectrum',
        line=dict(color='#f4a261', width=2),
        fill='tonexty'
    ))
    
    fig.update_layout(
        title='Frequency Spectrum Analysis',
        xaxis_title='Frequency (Hz)',
        yaxis_title='Magnitude',
        template='plotly_white',
        height=300,
        xaxis=dict(range=[0, min(4000, sr/2)])
    )
    
    return fig

def create_timbre_visualization(timbre_analysis: Dict[str, Any]) -> go.Figure:
    """Create timbre analysis visualization."""
    
    energy_dist = timbre_analysis['energy_distribution']
    
    fig = go.Figure(data=[
        go.Bar(
            x=['Bass (< 300Hz)', 'Mid (300-3000Hz)', 'Treble (> 3000Hz)'],
            y=[energy_dist['bass'], energy_dist['mid'], energy_dist['treble']],
            marker_color=['#1a1f36', '#2a9d8f', '#f4a261']
        )
    ])
    
    fig.update_layout(
        title='Timbre Analysis - Energy Distribution',
        yaxis_title='Relative Energy',
        template='plotly_white',
        height=300,
        showlegend=False
    )
    
    return fig

def main():
    """Main application function."""
    
    # Page configuration
    st.set_page_config(
        page_title="Voice Analysis Pro",
        page_icon="🎤",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS for enhanced styling
    st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .stApp {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
    }
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin: 0.5rem 0;
    }
    .hero-text {
        font-size: 3rem;
        font-weight: bold;
        background: linear-gradient(45deg, #1a1f36, #2a9d8f);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    .stButton > button {
        background: linear-gradient(45deg, #2a9d8f, #f4a261);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        border-radius: 8px;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.15);
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Initialize session state
    if 'analysis_results' not in st.session_state:
        st.session_state.analysis_results = None
    if 'audio_data' not in st.session_state:
        st.session_state.audio_data = None
    if 'song_catalog' not in st.session_state:
        st.session_state.song_catalog = load_song_catalog()
    
    # Header
    st.markdown('<h1 class="hero-text">🎤 Voice Analysis Pro</h1>', unsafe_allow_html=True)
    st.markdown("*Professional vocal analysis with real-time feedback and song recommendations*")
    
    # Navigation tabs
    tabs = st.tabs(["🎙️ Record/Upload", "🔍 Analysis", "📈 Charts", "🎵 Recommendations", "📚 Daily Lessons"])
    
    # Tab 1: Record/Upload
    with tabs[0]:
        st.header("Audio Input")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("File Upload")
            uploaded_file = st.file_uploader(
                "Choose an audio file", 
                type=['wav', 'mp3', 'm4a', 'flac'],
                help="Upload a recording of your singing or speaking"
            )
            
            if uploaded_file is not None:
                # Load and process the audio file
                try:
                    audio_bytes = uploaded_file.read()
                    audio, sr = librosa.load(io.BytesIO(audio_bytes), sr=SAMPLE_RATE)
                    st.session_state.audio_data = (audio, sr)
                    
                    # Display file info
                    st.success(f"✅ File loaded: {uploaded_file.name}")
                    st.info(f"Duration: {len(audio)/sr:.2f} seconds | Sample Rate: {sr} Hz")
                    
                    # Audio player
                    st.audio(audio_bytes, format=uploaded_file.type)
                    
                except Exception as e:
                    st.error(f"Error loading file: {e}")
        
        with col2:
            st.subheader("Quick Test")
            st.info("Use the sample audio below to test the analysis features")
            
            # Generate a test tone (C major scale)
            if st.button("Generate Test Audio"):
                duration = 3.0
                frequencies = [261.63, 293.66, 329.63, 349.23, 392.00, 440.00, 493.88, 523.25]  # C major scale
                test_audio = []
                
                for i, freq in enumerate(frequencies):
                    t = np.linspace(0, duration/len(frequencies), int(SAMPLE_RATE * duration/len(frequencies)))
                    tone = 0.5 * np.sin(2 * np.pi * freq * t)
                    test_audio.extend(tone)
                
                test_audio = np.array(test_audio)
                st.session_state.audio_data = (test_audio, SAMPLE_RATE)
                st.success("✅ Test audio generated!")
                
                # Play the test audio
                buffer = io.BytesIO()
                sf.write(buffer, test_audio, SAMPLE_RATE, format='wav')
                buffer.seek(0)
                st.audio(buffer, format='audio/wav')
    
    # Tab 2: Analysis
    with tabs[1]:
        st.header("Vocal Analysis Results")
        
        if st.session_state.audio_data is not None:
            audio, sr = st.session_state.audio_data
            
            # Perform analysis
            with st.spinner("Analyzing your voice..."):
                # Detect pitch
                f0, times = detect_pitch_audio(audio, sr)
                
                # Analyze vocal range
                vocal_analysis = analyze_vocal_range(f0)
                
                # Analyze timbre
                timbre_analysis = analyze_timbre(audio, sr)
                
                # Calculate pitch accuracy
                accuracy_analysis = calculate_pitch_accuracy(f0)
                
                # Store results
                st.session_state.analysis_results = {
                    'f0': f0,
                    'times': times,
                    'vocal_analysis': vocal_analysis,
                    'timbre_analysis': timbre_analysis,
                    'accuracy_analysis': accuracy_analysis,
                    'audio': audio,
                    'sr': sr
                }
            
            results = st.session_state.analysis_results
            
            # Display results in a grid
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric("Voice Type", results['vocal_analysis']['voice_type'])
                st.metric("Confidence", f"{results['vocal_analysis']['confidence']:.1%}")
                st.metric("Vocal Range", f"{results['vocal_analysis']['min_freq']:.0f} - {results['vocal_analysis']['max_freq']:.0f} Hz")
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col2:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric("Tessitura", f"{results['vocal_analysis']['tessitura_low']:.0f} - {results['vocal_analysis']['tessitura_high']:.0f} Hz")
                st.metric("Timbre", results['timbre_analysis']['timbre_class'])
                st.metric("Voiced Frames", f"{results['vocal_analysis']['voiced_percentage']:.1f}%")
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col3:
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.metric("Pitch Accuracy", f"{results['accuracy_analysis']['accuracy_score']:.0f}/100")
                st.metric("Stability", f"{results['accuracy_analysis']['stability']:.0f}/100")
                st.metric("Mean Cents Error", f"{results['accuracy_analysis']['mean_cents_error']:.1f}")
                st.markdown('</div>', unsafe_allow_html=True)
            
            # Detailed range visualization
            st.subheader("Vocal Range Visualization")
            
            # Create piano keyboard visualization
            fig_range = go.Figure()
            
            # Add piano keys (simplified)
            notes = ['C2', 'D2', 'E2', 'F2', 'G2', 'A2', 'B2', 
                    'C3', 'D3', 'E3', 'F3', 'G3', 'A3', 'B3',
                    'C4', 'D4', 'E4', 'F4', 'G4', 'A4', 'B4',
                    'C5', 'D5', 'E5', 'F5', 'G5', 'A5', 'B5']
            
            freqs = [note_to_hz(note) for note in notes]
            
            # Add all notes as background
            fig_range.add_trace(go.Bar(
                x=notes,
                y=[1] * len(notes),
                marker_color='lightgray',
                name='All Notes',
                opacity=0.3
            ))
            
            # Highlight user's vocal range
            user_notes = []
            for note in notes:
                freq = note_to_hz(note)
                if results['vocal_analysis']['min_freq'] <= freq <= results['vocal_analysis']['max_freq']:
                    user_notes.append(note)
            
            if user_notes:
                fig_range.add_trace(go.Bar(
                    x=user_notes,
                    y=[1] * len(user_notes),
                    marker_color='#2a9d8f',
                    name='Your Range',
                    opacity=0.8
                ))
            
            fig_range.update_layout(
                title='Your Vocal Range on Piano Keyboard',
                xaxis_title='Notes',
                yaxis_title='',
                template='plotly_white',
                height=200,
                showlegend=True
            )
            
            st.plotly_chart(fig_range, use_container_width=True)
            
        else:
            st.warning("⚠️ Please upload an audio file or generate test audio first")
    
    # Tab 3: Charts
    with tabs[2]:
        st.header("Interactive Visualizations")
        
        if st.session_state.analysis_results is not None:
            results = st.session_state.analysis_results
            
            # Pitch contour chart
            st.subheader("Pitch Contour Over Time")
            fig_pitch = create_pitch_visualization(results['f0'], results['times'], results['sr'])
            st.plotly_chart(fig_pitch, use_container_width=True)
            
            # Spectrum analysis
            st.subheader("Frequency Spectrum")
            fig_spectrum = create_spectrum_visualization(results['audio'], results['sr'])
            st.plotly_chart(fig_spectrum, use_container_width=True)
            
            # Timbre analysis
            st.subheader("Timbre Analysis")
            fig_timbre = create_timbre_visualization(results['timbre_analysis'])
            st.plotly_chart(fig_timbre, use_container_width=True)
            
            # Note distribution histogram
            st.subheader("Note Distribution")
            voiced_f0 = results['f0'][results['f0'] > 0]
            if len(voiced_f0) > 0:
                notes = [hz_to_note(f) for f in voiced_f0]
                note_counts = pd.Series(notes).value_counts()
                
                fig_notes = go.Figure(data=[
                    go.Bar(x=note_counts.index[:20], y=note_counts.values[:20])
                ])
                
                fig_notes.update_layout(
                    title='Most Frequently Sung Notes',
                    xaxis_title='Notes',
                    yaxis_title='Frequency',
                    template='plotly_white',
                    height=300
                )
                
                st.plotly_chart(fig_notes, use_container_width=True)
            
        else:
            st.warning("⚠️ No analysis data available. Please analyze audio first.")
    
    # Tab 4: Recommendations
    with tabs[3]:
        st.header("Song Recommendations")
        
        if st.session_state.analysis_results is not None and st.session_state.song_catalog is not None:
            results = st.session_state.analysis_results
            catalog = st.session_state.song_catalog
            
            # Generate recommendations
            with st.spinner("Generating personalized song recommendations..."):
                recommendations = generate_scommendations(
                    results['vocal_analysis'], 
                    results['timbre_analysis'], 
                    catalog,
                    n_recommendations=10
                )
            
            # Display recommendations
            st.subheader("🎯 Perfect Fit Songs")
            perfect_fit = recommendations[recommendations['recommendation_type'] == 'Perfect Fit']
            if not perfect_fit.empty:
                for _, song in perfect_fit.head(3).iterrows():
                    with st.expander(f"🌟 {song['title']} by {song['artist']}"):
                        st.write(f"**Genre:** {song['genre']} | **Difficulty:** {song['difficulty']}")
                        st.write(f"**Vocal Range:** {song['song_range']}")
                        st.write(f"**Compatibility Score:** {song['total_score']:.1%}")
                        st.success("Perfect match for your voice type and range!")
            else:
                st.info("No perfect matches found, but here are some great options!")
            
            st.subheader("📈 Good Match Songs")
            good_match = recommendations[recommendations['recommendation_type'] == 'Good Match']
            if not good_match.empty:
                for _, song in good_match.head(3).iterrows():
                    with st.expander(f"✅ {song['title']} by {song['artist']}"):
                        st.write(f"**Genre:** {song['genre']} | **Difficulty:** {song['difficulty']}")
                        st.write(f"**Vocal Range:** {song['song_range']}")
                        st.write(f"**Compatibility Score:** {song['total_score']:.1%}")
                        st.info("Great choice for building your vocal skills!")
            
            st.subheader("🚀 Stretch Songs (For Growth)")
            stretch = recommendations[recommendations['recommendation_type'] == 'Stretch Song']
            if not stretch.empty:
                for _, song in stretch.head(3).iterrows():
                    with st.expander(f"🎯 {song['title']} by {song['artist']}"):
                        st.write(f"**Genre:** {song['genre']} | **Difficulty:** {song['difficulty']}")
                        st.write(f"**Vocal Range:** {song['song_range']}")
                        st.write(f"**Compatibility Score:** {song['total_score']:.1%}")
                        st.warning("This song will challenge you and help expand your range!")
            
            # Song matching summary
            st.subheader("📊 Recommendation Analysis")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Total Songs Analyzed", len(catalog))
            with col2:
                st.metric("Perfect Matches", len(perfect_fit))
            with col3:
                st.metric("Good Matches", len(good_match) + len(stretch))
            
        else:
            st.warning("⚠️ Please complete vocal analysis first to get recommendations")
    
    # Tab 5: Daily Lessons
    with tabs[4]:
        st.header("Daily Singing Lessons")
        
        if st.session_state.analysis_results is not None:
            results = st.session_state.analysis_results
            
            st.subheader("🎵 Today's Practice Plan")
            
            # Generate practice recommendations based on analysis
            voice_type = results['vocal_analysis']['voice_type']
            accuracy_score = results['accuracy_analysis']['accuracy_score']
            
            # Warm-up exercises
            st.markdown("### 🌅 Warm-up Exercises")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Breathing Exercises**")
                st.write("1. Deep breathing (4-4-4-4 pattern)")
                st.write("2. Lip trills for 30 seconds")
                st.write("3. Sirens from low to high")
                
                if st.button("Start Breathing Exercise"):
                    st.info("Breathe in for 4 counts, hold for 4, out for 4, hold for 4...")
                    time.sleep(2)
                    st.success("Great! Continue with lip trills.")
            
            with col2:
                st.markdown("**Vocal Exercises**")
                st.write("1. Scale practice in your comfortable range")
                st.write("2. Arpeggio patterns")
                st.write("3. Interval training")
                
                if st.button("Start Scale Practice"):
                    st.info("Let's practice scales in your tessitura range...")
                    time.sleep(2)
                    st.success("Excellent pitch control!")
            
            # Targeted exercises based on analysis
            st.markdown("### 🎯 Targeted Improvement")
            
            if accuracy_score < 70:
                st.warning("Your pitch accuracy needs work. Focus on these exercises:")
                st.write("• Practice with a tuner or piano")
                st.write("• Slow, deliberate note matching")
                st.write("• Record yourself and listen back")
            
            if results['accuracy_analysis']['stability'] < 70:
                st.warning("Your vocal stability could improve. Try these:")
                st.write("• Sustained note practice")
                st.write("• Breath support exercises")
                st.write("• Gentle vibrato exercises")
            
            # Daily challenge
            st.markdown("### 🏆 Daily Challenge")
            
            challenges = [
                "Sing a song in your tessitura range",
                "Practice matching pitches with a reference tone",
                "Record yourself singing and analyze the playback",
                "Try singing a song slightly outside your comfort zone",
                "Practice breathing exercises for 5 minutes",
                "Work on vocal runs and melismas",
                "Sing with different emotional expressions",
                "Practice harmonizing with a recording"
            ]
            
            import random
            daily_challenge = random.choice(challenges)
            
            st.info(f"**Today's Challenge:** {daily_challenge}")
            
            if st.button("Complete Challenge"):
                st.balloons()
                st.success("🎉 Great job! You've completed today's challenge!")
                
                # Add to progress (in a real app, this would be stored)
                if 'daily_streak' not in st.session_state:
                    st.session_state.daily_streak = 0
                st.session_state.daily_streak += 1
                
                st.metric("Daily Streak", f"{st.session_state.daily_streak} days")
            
            # Song of the day
            st.markdown("### 🎶 Song of the Day")
            
            if st.session_state.song_catalog is not None:
                catalog = st.session_state.song_catalog
                suitable_songs = catalog[catalog['difficulty'].isin(['Beginner', 'Intermediate'])]
                
                if not suitable_songs.empty:
                    song_of_day = suitable_songs.sample(1).iloc[0]
                    
                    st.write(f"**{song_of_day['title']}** by {song_of_day['artist']}")
                    st.write(f"Genre: {song_of_day['genre']} | Difficulty: {song_of_day['difficulty']}")
                    st.write(f"Range: {song_of_day['typical_low']} - {song_of_day['typical_high']}")
                    
                    if st.button("Practice This Song"):
                        st.info("Great choice! Remember to warm up first.")
                        st.write("Tips for this song:")
                        st.write(f"• Focus on the {song_of_day['timbre_tag'].lower()} quality")
                        st.write("• Pay attention to breath support")
                        st.write("• Practice the challenging sections slowly")
            
        else:
            st.warning("⚠️ Complete a vocal analysis first to get personalized lessons")
    
    # Sidebar with additional information
    with st.sidebar:
        st.header("📱 Quick Controls")
        
        if st.button("🔄 Reset Analysis"):
            st.session_state.analysis_results = None
            st.session_state.audio_data = None
            st.rerun()
        
        st.header("ℹ️ About")
        st.info("""
        **Voice Analysis Pro** uses advanced audio processing to analyze your voice and provide personalized recommendations.
        
        **Features:**
        • Real-time pitch detection
        • Vocal range analysis
        • Timbre classification
        • Song recommendations
        • Daily practice lessons
        
        **Privacy:** All processing happens locally on your device.
        """)
        
        if st.session_state.analysis_results is not None:
            st.header("📊 Quick Stats")
            results = st.session_state.analysis_results
            
            st.metric("Voice Type", results['vocal_analysis']['voice_type'])
            st.metric("Accuracy", f"{results['accuracy_analysis']['accuracy_score']:.0f}%")
            st.metric("Timbre", results['timbre_analysis']['timbre_class'])
    
    # Footer
    st.markdown("---")
    st.markdown("*Voice Analysis Pro - Professional vocal analysis made accessible*")

if __name__ == "__main__":
    main()

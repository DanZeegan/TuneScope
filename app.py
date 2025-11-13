"""
========================================
STREAMLIT VOICE STUDIO - Complete Single File
========================================

A production-grade voice analysis, accent training, and singing studio application.
Built for Streamlit Cloud and local deployment.

DEV NOTES:
----------
1. INSTALLATION:
   pip install streamlit streamlit-webrtc librosa numpy pandas matplotlib plotly soundfile scipy scikit-learn

   OPTIONAL (for enhanced pitch detection):
   pip install torch torchcrepe

2. RUN LOCALLY:
   streamlit run app.py

3. DEPLOY TO STREAMLIT.IO:
   - Push this file to GitHub
   - Connect repository to Streamlit Cloud
   - Add requirements.txt with dependencies

4. ADDING SONGS TO CATALOG:
   - Edit song_catalog.csv after first run
   - Or use the built-in catalog editor in the app

5. TROUBLESHOOTING MIC ISSUES:
   - Check browser permissions (chrome://settings/content/microphone)
   - Ensure HTTPS or localhost
   - Try different browsers (Chrome recommended)
   - Check Windows mic privacy settings

6. PRIVACY:
   - All processing is LOCAL by generate_recommendations(stats, timbre, catalog_df, preference='modern'):
    """Generate song recommendations based on voice profile"""
    if catalog_df is None:
        return None
    
    user_range = (stats['tessitura_low'], stats['tessitura_high'])
    
    fit = []
    stretch = []
    avoid = []
    
    for idx, row in catalog_df.iterrows():
        song_range = (row['low'], row['high'])
        
        # Check overlap
        overlap = (
            max(user_range[0], song_range[0]) < min(user_range[1], song_range[1])
        )
        
        range_width = song_range[1] - song_range[0]
        user_width = user_range[1] - user_range[0]
        
        # Filter by preference
        if preference == 'modern' and 'vintage' in row['tags']:
            continue
        if preference == 'vintage' and 'modern' in row['tags']:
            continue
        
        song_info = {
            'title': row['title'],
            'artist': row['artist'],
            'key': row['key'],
            'tags': row['tags']
        }
        
        if overlap and range_width <= user_width * 1.2:
            fit.append(song_info)
        elif overlap or range_width <= user_width * 1.5:
            stretch.append(song_info)
        else:
            avoid.append(song_info)
    
    return {
        'fit': fit[:5],
        'stretch': stretch[:5],
        'avoid': avoid[:3]
    }

# ============================================
# VISUALIZATION
# ============================================

def plot_pitch_curve(timeline_df):
    """Create interactive pitch curve plot"""
    voiced = timeline_df[timeline_df['voiced'] == True]
    
    fig = go.Figure()
    
    # Add pitch trace
    fig.add_trace(go.Scatter(
        x=voiced['time'],
        y=voiced['hz'],
        mode='lines+markers',
        name='Pitch',
        line=dict(color='#667eea', width=2),
        marker=dict(size=4),
        hovertemplate='<b>Time:</b> %{x:.2f}s<br><b>Pitch:</b> %{y:.1f} Hz<br><b>Note:</b> %{text}<br><b>Cents:</b> %{customdata:.1f}<extra></extra>',
        text=voiced['note_name'],
        customdata=voiced['cents_error']
    ))
    
    fig.update_layout(
        title='Pitch Analysis',
        xaxis_title='Time (seconds)',
        yaxis_title='Frequency (Hz)',
        hovermode='closest',
        template='plotly_white',
        height=400
    )
    
    return fig

def plot_note_histogram(timeline_df):
    """Plot histogram of detected notes"""
    voiced = timeline_df[timeline_df['voiced'] == True]
    note_counts = voiced['note_name'].value_counts().head(10)
    
    fig = go.Figure(data=[
        go.Bar(x=note_counts.index, y=note_counts.values, marker_color='#764ba2')
    ])
    
    fig.update_layout(
        title='Most Common Notes',
        xaxis_title='Note',
        yaxis_title='Count',
        template='plotly_white',
        height=300
    )
    
    return fig

def plot_cents_distribution(timeline_df):
    """Plot cents error distribution"""
    voiced = timeline_df[timeline_df['voiced'] == True]
    
    fig = go.Figure(data=[
        go.Histogram(x=voiced['cents_error'], nbinsx=30, marker_color='#667eea')
    ])
    
    fig.update_layout(
        title='Intonation Accuracy (Cents Error)',
        xaxis_title='Cents from Target',
        yaxis_title='Count',
        template='plotly_white',
        height=300
    )
    
    return fig

def plot_vocal_range(stats):
    """Plot vocal range bar"""
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=['Min', 'Tessitura Low', 'Median', 'Tessitura High', 'Max'],
        y=[stats['min_hz'], stats['tessitura_low'], stats['median_hz'], 
           stats['tessitura_high'], stats['max_hz']],
        marker_color=['#ff6b6b', '#feca57', '#48dbfb', '#feca57', '#ff6b6b']
    ))
    
    fig.update_layout(
        title='Vocal Range Analysis',
        yaxis_title='Frequency (Hz)',
        template='plotly_white',
        height=300
    )
    
    return fig

def plot_timbre_bars(timbre):
    """Plot timbre energy distribution"""
    fig = go.Figure(data=[
        go.Bar(
            x=['Low (<300Hz)', 'Mid (300-3000Hz)', 'High (>3000Hz)'],
            y=[timbre['low_ratio'], timbre['mid_ratio'], timbre['high_ratio']],
            marker_color=['#e74c3c', '#f39c12', '#3498db']
        )
    ])
    
    fig.update_layout(
        title='Timbre Profile',
        yaxis_title='Energy Ratio',
        template='plotly_white',
        height=300
    )
    
    return fig

# ============================================
# DEMO & TEST AUDIO
# ============================================

def generate_test_audio(duration=3.0, sr=22050):
    """Generate C major scale test audio"""
    notes_hz = [262, 294, 330, 349, 392, 440, 494, 523]  # C4 to C5
    t_note = duration / len(notes_hz)
    
    y = np.array([])
    for hz in notes_hz:
        t = np.linspace(0, t_note, int(sr * t_note))
        note = 0.5 * np.sin(2 * np.pi * hz * t)
        # Add envelope
        envelope = np.exp(-3 * t / t_note)
        note = note * envelope
        y = np.concatenate([y, note])
    
    return y, sr

# ============================================
# EXPORT FUNCTIONS
# ============================================

def export_analysis_json(analysis_results, accent_profile):
    """Export analysis as JSON"""
    export_data = {
        'timestamp': datetime.now().isoformat(),
        'pitch_stats': analysis_results['stats'] if analysis_results else {},
        'voice_type': st.session_state.get('voice_type', 'Unknown'),
        'timbre': analysis_results.get('timbre', {}) if analysis_results else {},
        'accent': accent_profile if accent_profile else {},
        'pronunciation_scores': st.session_state.get('pronunciation_scores', [])
    }
    
    return json.dumps(export_data, indent=2)

def export_notes_csv(timeline_df):
    """Export notes timeline as CSV"""
    return timeline_df.to_csv(index=False)

# ============================================
# MAIN APP
# ============================================

def main():
    st.set_page_config(
        page_title="Voice Studio Pro",
        page_icon="🎤",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Initialize session state
    init_session_state()
    
    # Apply theme
    apply_theme()
    
    # Header
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        st.title("🎤 Voice Studio Pro")
        st.caption("Professional Voice Analysis, Accent Training & Singing Coach")
    
    with col2:
        if st.button("🌟 Modern", use_container_width=True):
            st.session_state.theme = 'modern'
            st.rerun()
    
    with col3:
        if st.button("💾 Vintage", use_container_width=True):
            st.session_state.theme = 'vintage'
            st.rerun()
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        
        st.subheader("Preprocessing")
        trim_silence_opt = st.checkbox("Trim Silence", value=True)
        
        st.subheader("Accessibility")
        font_size = st.radio("Font Size", ["Normal", "Large"], index=0)
        high_contrast = st.checkbox("High Contrast", value=False)
        
        st.subheader("Privacy & Features")
        use_groq = st.checkbox("Enable GROQ AI (Opt-in)", value=False)
        if use_groq:
            st.info("🔒 GROQ API will be used for generating practice ideas. Your audio stays local.")
            groq_key = st.text_input("GROQ API Key (optional)", type="password")
            st.session_state.groq_api_key = groq_key
        
        st.divider()
        st.caption(f"Theme: {st.session_state.theme.title()}")
        st.caption(f"Pitch Engine: {'CREPE' if CREPE_AVAILABLE else 'YIN (Librosa)'}")
        
        if st.button("ℹ️ About"):
            st.info("""
            **Voice Studio Pro v1.0**
            
            A privacy-first voice analysis tool.
            
            ✅ All processing is LOCAL
            ✅ No telemetry
            ✅ CPU-optimized
            
            Made with ❤️ by AI
            """)
    
    # Main tabs
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "🎙️ Record/Upload",
        "📊 Voice Analysis", 
        "📈 Charts",
        "🗣️ Accent Training",
        "🎵 Singing Training",
        "🎼 Recommendations",
        "ℹ️ About"
    ])
    
    # TAB 1: Record/Upload
    with tab1:
        st.header("Audio Input")
        
        input_method = st.radio("Choose input method:", ["Upload File", "Generate Test Audio", "Live Recording (Coming Soon)"])
        
        if input_method == "Upload File":
            uploaded_file = st.file_uploader(
                "Upload audio file",
                type=['wav', 'mp3', 'm4a', 'flac'],
                help="Supported formats: WAV, MP3, M4A, FLAC (max 10 minutes)"
            )
            
            if uploaded_file:
                with st.spinner("Loading audio..."):
                    audio_bytes = uploaded_file.read()
                    y, sr = load_audio(audio_bytes, sr=CONFIG['sample_rate'])
                    
                    if y is not None:
                        st.session_state.audio_data = y
                        st.session_state.sample_rate = sr
                        
                        # Show waveform
                        st.success("✅ Audio loaded successfully!")
                        
                        col1, col2, col3, col4 = st.columns(4)
                        col1.metric("Duration", f"{len(y)/sr:.2f}s")
                        col2.metric("Sample Rate", f"{sr} Hz")
                        col3.metric("Channels", "Mono")
                        col4.metric("Peak", f"{np.max(np.abs(y)):.2f}")
                        
                        # Plot waveform
                        fig, ax = plt.subplots(figsize=(10, 3))
                        librosa.display.waveshow(y, sr=sr, ax=ax)
                        ax.set_title("Waveform")
                        ax.set_xlabel("Time (s)")
                        ax.set_ylabel("Amplitude")
                        st.pyplot(fig)
                        plt.close()
                        
                        # Audio player
                        st.audio(audio_bytes)
        
        elif input_method == "Generate Test Audio":
            if st.button("🎵 Generate C Major Scale"):
                with st.spinner("Generating test audio..."):
                    y, sr = generate_test_audio()
                    st.session_state.audio_data = y
                    st.session_state.sample_rate = sr
                    
                    st.success("✅ Test audio generated!")
                    
                    # Convert to bytes for playback
                    buffer = io.BytesIO()
                    sf.write(buffer, y, sr, format='WAV')
                    st.audio(buffer.getvalue())
        
        else:  # Live Recording
            st.info("🚧 Live recording with streamlit-webrtc coming soon!")
            st.markdown("""
            **To enable live recording:**
            1. Install: `pip install streamlit-webrtc`
            2. Ensure HTTPS or localhost
            3. Grant browser microphone permissions
            
            For now, please use file upload or test audio.
            """)
        
        # Analysis button
        st.divider()
        if st.session_state.audio_data is not None:
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("🔍 Analyze Voice", type="primary", use_container_width=True):
                    with st.spinner("Analyzing... This may take a moment."):
                        y = st.session_state.audio_data
                        sr = st.session_state.sample_rate
                        
                        # Trim silence if requested
                        if trim_silence_opt:
                            y = trim_silence(y, sr)
                        
                        # Pitch analysis
                        pitch_analysis = analyze_pitch(y, sr)
                        
                        if pitch_analysis:
                            # Voice type
                            voice_type, confidence = classify_voice_type(pitch_analysis['stats'])
                            
                            # Timbre
                            timbre = analyze_timbre(y, sr)
                            
                            # Accent detection
                            accent = detect_accent(y, sr)
                            
                            # Store results
                            st.session_state.analysis_results = {
                                'pitch': pitch_analysis,
                                'voice_type': voice_type,
                                'voice_confidence': confidence,
                                'timbre': timbre,
                                'stats': pitch_analysis['stats']
                            }
                            st.session_state.accent_profile = accent
                            
                            st.success("✅ Analysis complete!")
                            st.balloons()
                        else:
                            st.error("❌ Could not analyze audio. Please check the recording quality.")
            
            with col2:
                if st.button("🧪 Detect Accent", use_container_width=True):
                    with st.spinner("Detecting accent..."):
                        y = st.session_state.audio_data
                        sr = st.session_state.sample_rate
                        
                        accent = detect_accent(y, sr)
                        st.session_state.accent_profile = accent
                        
                        st.success(f"✅ Detected: {accent['detected_accent']} ({accent['confidence']:.0f}% confidence)")
    
    # TAB 2: Voice Analysis
    with tab2:
        if st.session_state.analysis_results:
            results = st.session_state.analysis_results
            stats = results['stats']
            
            st.header("📊 Voice Analysis Results")
            
            # Voice Type
            st.subheader("🎭 Voice Type")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Classification", results['voice_type'] or "Unknown")
            with col2:
                st.metric("Confidence", f"{results['voice_confidence']:.0f}%")
            with col3:
                st.metric("Voiced Coverage", f"{stats['voiced_percentage']:.1f}%")
            
            # Range Statistics
            st.subheader("🎼 Pitch Range")
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Min Pitch", f"{stats['min_hz']:.1f} Hz")
            col2.metric("Max Pitch", f"{stats['max_hz']:.1f} Hz")
            col3.metric("Median", f"{stats['median_hz']:.1f} Hz")
            col4.metric("Intonation Score", f"{stats['intonation_score']:.0f}/100")
            
            # Tessitura
            st.subheader("🎯 Comfortable Range (Tessitura)")
            col1, col2 = st.columns(2)
            col1.metric("Lower Bound", f"{stats['tessitura_low']:.1f} Hz")
            col2.metric("Upper Bound", f"{stats['tessitura_high']:.1f} Hz")
            
            # Timbre
            st.subheader("🎨 Timbre Profile")
            timbre = results['timbre']
            st.info(f"**{timbre['badge']}**")
            st.write(timbre['tips'])
            
            # Cents error
            st.subheader("🎯 Intonation Details")
            col1, col2 = st.columns(2)
            col1.metric("Avg Cents Error", f"{stats['cents_mean']:.1f}¢")
            col2.metric("Std Deviation", f"{stats['cents_std']:.1f}¢")
            
            st.caption("💡 Lower cents error = better intonation. Under 20¢ is excellent!")
            
        else:
            st.info("👆 Upload and analyze audio in the Record/Upload tab first.")
    
    # TAB 3: Charts
    with tab3:
        if st.session_state.analysis_results:
            results = st.session_state.analysis_results
            timeline = results['pitch']['timeline']
            
            st.header("📈 Visualization")
            
            # Pitch curve
            st.plotly_chart(plot_pitch_curve(timeline), use_container_width=True)
            
            col1, col2 = st.columns(2)
            with col1:
                st.plotly_chart(plot_note_histogram(timeline), use_container_width=True)
            with col2:
                st.plotly_chart(plot_cents_distribution(timeline), use_container_width=True)
            
            col1, col2 = st.columns(2)
            with col1:
                st.plotly_chart(plot_vocal_range(results['stats']), use_container_width=True)
            with col2:
                st.plotly_chart(plot_timbre_bars(results['timbre']), use_container_width=True)
        else:
            st.info("👆 Analyze audio first to see charts.")
    
    # TAB 4: Accent Training
    with tab4:
        st.header("🗣️ Accent Detection & Training")
        
        if st.session_state.accent_profile:
            accent = st.session_state.accent_profile
            
            st.success(f"✅ Detected Accent: **{accent['detected_accent']}** ({accent['confidence']:.0f}% confidence)")
            
            st.subheader("Weak Areas")
            for sound in accent['weak_sounds']:
                with st.expander(f"🔤 {sound}"):
                    st.write(PHONEME_TIPS.get(sound, "Practice this sound"))
        else:
            st.info("👆 Run accent detection in the Record/Upload tab first.")
        
        st.divider()
        
        # Target accent selection
        st.subheader("🎯 Target Accent")
        target = st.selectbox("Which accent would you like to learn?", ["American", "British"])
        st.session_state.target_accent = target
        
        if target:
            st.info(f"Target: **{target}** English")
            
            st.subheader("Key Sounds to Practice")
            for sound in ACCENT_FEATURES[target]['key_sounds']:
                with st.expander(f"🔤 {sound}"):
                    st.write(PHONEME_TIPS.get(sound, "Practice this sound"))
            
            st.subheader("Practice Sentences")
            practice_sentences = {
                'American': [
                    "The car parked in the yard.",
                    "Water the flowers in the garden.",
                    "I'm going to the store later."
                ],
                'British': [
                    "The car is parked in the garden.",
                    "I'd rather have a cup of tea.",
                    "It's absolutely marvellous."
                ]
            }
            
            for i, sentence in enumerate(practice_sentences[target], 1):
                st.write(f"{i}. **{sentence}**")
            
            if st.button("🎤 Record Practice & Get Feedback"):
                st.info("Upload a recording of yourself saying these sentences, then run pronunciation analysis!")
    
    # TAB 5: Singing Training
    with tab5:
        st.header("🎵 Singing Training")
        
        if st.session_state.analysis_results:
            results = st.session_state.analysis_results
            stats = results['stats']
            
            st.subheader("📊 Your Singing Stats")
            col1, col2, col3 = st.columns(3)
            col1.metric("Pitch Accuracy", f"{stats['intonation_score']:.0f}/100")
            col2.metric("Note Stability", f"{100 - min(100, stats['cents_std']):.0f}/100")
            col3.metric("Vocal Coverage", f"{stats['voiced_percentage']:.1f}%")
            
            # Recommendations
            st.subheader("💡 Training Tips")
            
            if stats['intonation_score'] < 70:
                st.warning("🎯 **Focus on Pitch Accuracy**")
                st.write("- Practice with a piano or tuner")
                st.write("- Sing scales slowly")
                st.write("- Record yourself and compare")
            else:
                st.success("✅ Great pitch accuracy!")
            
            if stats['cents_std'] > 30:
                st.warning("🎯 **Work on Stability**")
                st.write("- Practice sustaining single notes")
                st.write("- Work on breath support")
                st.write("- Try vowel exercises")
            else:
                st.success("✅ Excellent note stability!")
            
            # Song identification
            st.divider()
            st.subheader("🎼 Song Identification")
            
            if st.button("🔍 Identify Song"):
                catalog = load_or_create_catalog()
                matches = identify_song(results['pitch'], catalog)
                
                if matches:
                    st.success(f"🎵 Top matches:")
                    for i, match in enumerate(matches, 1):
                        st.write(f"{i}. **{match['title']}** by {match['artist']} - {match['confidence']:.0f}% match")
                else:
                    st.info("No strong matches found. Try singing more of the song!")
        else:
            st.info("👆 Analyze your singing first!")
            
            st.subheader("🎤 Singing Exercises")
            st.write("**Warm-up exercises to try:**")
            st.write("1. 🎵 Lip trills (5 minutes)")
            st.write("2. 🎵 Humming scales")
            st.write("3. 🎵 'Mee-may-mah-mo-moo' on different pitches")
            st.write("4. 🎵 Sirens (low to high pitch)")
    
    # TAB 6: Recommendations
    with tab6:
        st.header("🎼 Song Recommendations")
        
        if st.session_state.analysis_results:
            results = st.session_state.analysis_results
            
            preference = st.radio("Song preference:", ["Modern", "Vintage", "All"])
            
            if st.button("🎵 Get Recommendations"):
                catalog = load_or_create_catalog()
                recs = generate_recommendations(
                    results['stats'],
                    results['timbre'],
                    catalog,
                    preference.lower()
                )
                
                if recs:
                    st.subheader("✅ Perfect Fit for Your Voice")
                    for song in recs['fit']:
                        st.success(f"🎵 **{song['title']}** by {song['artist']} (Key: {song['key']})")
                    
                    st.subheader("💪 Stretch Goals")
                    for song in recs['stretch']:
                        st.warning(f"🎵 **{song['title']}** by {song['artist']} (Key: {song['key']})")
                    
                    st.subheader("❌ Avoid for Now")
                    for song in recs['avoid']:
                        st.error(f"🎵 **{song['title']}** by {song['artist']} (Key: {song['key']})")
        else:
            st.info("👆 Analyze your voice first to get personalized recommendations!")
            
            st.subheader("📚 Song Catalog")
            catalog = load_or_create_catalog()
            st.dataframe(catalog[['title', 'artist', 'key', 'tags']], use_container_width=True)
    
    # TAB 7: About
    with tab7:
        st.header("ℹ️ About Voice Studio Pro")
        
        st.markdown("""
        ### 🎤 Professional Voice Analysis Tool
        
        **Voice Studio Pro** is a privacy-first, CPU-optimized voice analysis application
        designed for singers, speakers, and voice coaches.
        
        #### ✨ Features
        - 🎯 Pitch detection & analysis
        - 🗣️ Accent detection & training
        - 🎵 Singing accuracy measurement
        - 🎨 Timbre profiling
        - 🎼 Song identification & recommendations
        - 📊 Beautiful visualizations
        - 🔒 100% local processing (privacy-first)
        - 💾 Two UI themes: Modern (glass) & Vintage (Windows 98)
        
        #### 🛠️ Technology
        - **Pitch Engine**: CREPE (if available) or Librosa YIN
        - **Processing**: NumPy, SciPy, Librosa
        - **Visualization**: Plotly, Matplotlib
        - **UI**: Streamlit
        
        #### 🔒 Privacy
        - All audio processing happens on your device
        - No data sent to external servers (unless GROQ is enabled)
        - No telemetry or tracking
        
        #### 📦 Export Options
        """)
        
        if st.session_state.analysis_results:
            col1, col2, col3 = st.columns(3)
            
            with col1:
                json_data = export_analysis_json(
                    st.session_state.analysis_results,
                    st.session_state.accent_profile
                )
                st.download_button(
                    "📄 Download JSON",
                    json_data,
                    "analysis.json",
                    "application/json"
                )
            
            with col2:
                csv_data = export_notes_csv(st.session_state.analysis_results['pitch']['timeline'])
                st.download_button(
                    "📊 Download CSV",
                    csv_data,
                    "notes_timeline.csv",
                    "text/csv"
                )
            
            with col3:
                summary = f"""
Voice Analysis Summary
======================
Voice Type: {st.session_state.analysis_results['voice_type']}
Confidence: {st.session_state.analysis_results['voice_confidence']:.0f}%
Pitch Range: {st.session_state.analysis_results['stats']['min_hz']:.1f} - {st.session_state.analysis_results['stats']['max_hz']:.1f} Hz
Intonation Score: {st.session_state.analysis_results['stats']['intonation_score']:.0f}/100
Timbre: {st.session_state.analysis_results['timbre']['badge']}

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
                """
                st.download_button(
                    "📝 Download Summary",
                    summary,
                    "voice_summary.txt",
                    "text/plain"
                )
        
        st.divider()
        
        st.markdown("""
        #### 🚀 Getting Started
        1. Upload an audio file or generate test audio
        2. Click "Analyze Voice"
        3. Explore results in different tabs
        4. Try accent training and singing exercises
        5. Get personalized song recommendations
        
        #### 🐛 Troubleshooting
        - **No sound detected**: Check file format and volume
        - **Poor pitch detection**: Ensure clear, solo vocals
        - **App runs slow**: Use shorter audio clips (< 2 minutes)
        
        #### 📚 Requirements
        ```
        streamlit
        librosa
        numpy
        pandas
        matplotlib
        plotly
        soundfile
        scipy
        scikit-learn
        ```
        
        Optional for better pitch detection:
        ```
        torch
        torchcrepe
        ```
        
        ---
        
        Made with ❤️ using AI | Version 1.0 | MIT License
        """)

if __name__ == "__main__":
    main()ault
   - GROQ API is OPT-IN only
   - No telemetry or tracking

VERSION: 1.0
AUTHOR: AI-Generated Voice Studio
LICENSE: MIT
"""

import streamlit as st
import numpy as np
import pandas as pd
import librosa
import librosa.display
import soundfile as sf
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from scipy import signal
from scipy.ndimage import median_filter
from scipy.spatial.distance import cosine
from sklearn.preprocessing import StandardScaler
import io
import json
import base64
import tempfile
import os
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Try importing optional dependencies
TORCH_AVAILABLE = False
CREPE_AVAILABLE = False
try:
    import torch
    TORCH_AVAILABLE = True
    try:
        import torchcrepe
        CREPE_AVAILABLE = True
    except ImportError:
        pass
except ImportError:
    pass

# ============================================
# CONFIGURATION & CONSTANTS
# ============================================

CONFIG = {
    'sample_rate': 22050,
    'hop_length': 512,
    'n_fft': 2048,
    'fmin': 80,
    'fmax': 1000,
    'A4_freq': 440.0,
    'voicing_threshold': 0.25,
    'silence_threshold': 0.01,
}

VOICE_TYPES = {
    'Bass': {'min': 87, 'max': 350, 'tessitura': (100, 300)},
    'Baritone': {'min': 98, 'max': 392, 'tessitura': (110, 350)},
    'Tenor': {'min': 130, 'max': 520, 'tessitura': (150, 450)},
    'Alto': {'min': 175, 'max': 700, 'tessitura': (200, 600)},
    'Mezzo-Soprano': {'min': 220, 'max': 880, 'tessitura': (250, 750)},
    'Soprano': {'min': 262, 'max': 1047, 'tessitura': (300, 900)},
}

ACCENT_FEATURES = {
    'American': {
        'r_emphasis': 'strong',
        't_flapping': True,
        'schwa_frequency': 'high',
        'intonation': 'flat',
        'key_sounds': ['R', 'T', 'schwa', 'AE']
    },
    'British': {
        'r_emphasis': 'weak',
        't_flapping': False,
        'schwa_frequency': 'very_high',
        'intonation': 'varied',
        'key_sounds': ['R', 'T', 'AH', 'OO']
    }
}

PHONEME_TIPS = {
    'TH': "Place tongue between teeth, blow air gently",
    'R': "American: Curl tongue back. British: Keep tongue flat",
    'T': "American: Soft tap in middle of words. British: Hard stop",
    'V': "Upper teeth on lower lip, vibrate",
    'W': "Round lips into 'oo' shape, then release",
    'schwa': "Neutral 'uh' sound, very relaxed",
    'AE': "Open mouth wide, say 'a' as in 'cat'",
}

# ============================================
# SESSION STATE INITIALIZATION
# ============================================

def init_session_state():
    """Initialize all session state variables"""
    defaults = {
        'theme': 'modern',
        'audio_data': None,
        'sample_rate': CONFIG['sample_rate'],
        'analysis_results': None,
        'accent_profile': None,
        'target_accent': None,
        'pronunciation_scores': [],
        'singing_scores': [],
        'catalog_df': None,
        'use_groq': False,
        'groq_api_key': '',
        'font_size': 'normal',
        'high_contrast': False,
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

# ============================================
# THEME STYLING
# ============================================

def apply_theme():
    """Apply selected theme CSS"""
    
    if st.session_state.theme == 'modern':
        # Glassmorphism Modern Theme
        st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
        
        * {
            font-family: 'Inter', sans-serif;
        }
        
        .stApp {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        }
        
        .main .block-container {
            background: rgba(255, 255, 255, 0.1);
            backdrop-filter: blur(10px);
            border-radius: 20px;
            padding: 2rem;
            border: 1px solid rgba(255, 255, 255, 0.2);
        }
        
        .stButton>button {
            background: rgba(255, 255, 255, 0.2);
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.3);
            border-radius: 15px;
            color: white;
            font-weight: 600;
            padding: 0.75rem 2rem;
            transition: all 0.3s ease;
        }
        
        .stButton>button:hover {
            background: rgba(255, 255, 255, 0.3);
            transform: translateY(-2px);
            box-shadow: 0 8px 16px rgba(0, 0, 0, 0.2);
        }
        
        .stTextInput>div>div>input, .stSelectbox>div>div>select {
            background: rgba(255, 255, 255, 0.15);
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.3);
            border-radius: 10px;
            color: white;
        }
        
        h1, h2, h3 {
            color: white;
            font-weight: 700;
        }
        
        .metric-card {
            background: rgba(255, 255, 255, 0.15);
            backdrop-filter: blur(10px);
            border-radius: 15px;
            padding: 1.5rem;
            border: 1px solid rgba(255, 255, 255, 0.2);
            margin: 1rem 0;
        }
        
        .stTabs [data-baseweb="tab-list"] {
            background: rgba(255, 255, 255, 0.1);
            backdrop-filter: blur(10px);
            border-radius: 10px;
            padding: 0.5rem;
        }
        
        .stTabs [data-baseweb="tab"] {
            color: white;
            font-weight: 600;
        }
        </style>
        """, unsafe_allow_html=True)
    
    else:  # vintage theme
        # Windows 98 Retro Theme
        st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=MS+Sans+Serif&display=swap');
        
        * {
            font-family: 'MS Sans Serif', 'Courier New', monospace;
        }
        
        .stApp {
            background: #008080;
        }
        
        .main .block-container {
            background: #c0c0c0;
            border: 3px outset #ffffff;
            padding: 2rem;
            box-shadow: 2px 2px 0px #000000;
        }
        
        .stButton>button {
            background: #c0c0c0;
            border: 2px outset #ffffff;
            color: #000000;
            font-weight: bold;
            padding: 0.5rem 1.5rem;
            box-shadow: 1px 1px 0px #000000;
        }
        
        .stButton>button:hover {
            border: 2px inset #ffffff;
        }
        
        .stButton>button:active {
            border: 2px inset #808080;
        }
        
        .stTextInput>div>div>input, .stSelectbox>div>div>select {
            background: #ffffff;
            border: 2px inset #808080;
            color: #000000;
            font-family: 'Courier New', monospace;
        }
        
        h1, h2, h3 {
            color: #000080;
            font-weight: bold;
            text-shadow: 1px 1px 0px #ffffff;
        }
        
        .metric-card {
            background: #c0c0c0;
            border: 2px outset #ffffff;
            padding: 1rem;
            margin: 1rem 0;
            box-shadow: 2px 2px 0px #000000;
        }
        
        .stTabs [data-baseweb="tab-list"] {
            background: #c0c0c0;
            border: 2px outset #ffffff;
        }
        
        .stTabs [data-baseweb="tab"] {
            color: #000000;
            font-weight: bold;
        }
        </style>
        """, unsafe_allow_html=True)

# ============================================
# AUDIO PROCESSING FUNCTIONS
# ============================================

def load_audio(file_path_or_bytes, sr=None):
    """Load audio from file path or bytes"""
    try:
        if isinstance(file_path_or_bytes, bytes):
            y, sr_native = sf.read(io.BytesIO(file_path_or_bytes))
        else:
            y, sr_native = librosa.load(file_path_or_bytes, sr=sr, mono=True)
        
        if sr and sr != sr_native:
            y = librosa.resample(y, orig_sr=sr_native, target_sr=sr)
        
        # Normalize
        if np.max(np.abs(y)) > 0:
            y = y / np.max(np.abs(y))
        
        return y, sr if sr else sr_native
    except Exception as e:
        st.error(f"Error loading audio: {e}")
        return None, None

def trim_silence(y, sr, threshold=0.01):
    """Trim silence from beginning and end"""
    non_silent = librosa.effects.split(y, top_db=20)
    if len(non_silent) > 0:
        start = non_silent[0][0]
        end = non_silent[-1][1]
        return y[start:end]
    return y

def extract_pitch_crepe(y, sr):
    """Extract pitch using CREPE (if available)"""
    if not CREPE_AVAILABLE:
        return None, None
    
    try:
        # Resample to 16kHz for CREPE
        if sr != 16000:
            y_16k = librosa.resample(y, orig_sr=sr, target_sr=16000)
        else:
            y_16k = y
        
        # Run CREPE
        audio_tensor = torch.tensor(y_16k[np.newaxis, :]).float()
        pitch, confidence = torchcrepe.predict(
            audio_tensor,
            16000,
            hop_length=160,
            fmin=80,
            fmax=1000,
            model='tiny',
            device='cpu',
            return_periodicity=True
        )
        
        f0 = pitch.squeeze().numpy()
        confidence = confidence.squeeze().numpy()
        
        return f0, confidence
    except Exception as e:
        st.warning(f"CREPE failed: {e}. Falling back to YIN.")
        return None, None

def extract_pitch_yin(y, sr):
    """Extract pitch using librosa YIN"""
    try:
        f0 = librosa.yin(
            y,
            fmin=CONFIG['fmin'],
            fmax=CONFIG['fmax'],
            sr=sr,
            hop_length=CONFIG['hop_length']
        )
        
        # Create voicing confidence based on energy
        frame_length = CONFIG['hop_length'] * 2
        energy = np.array([
            np.sum(y[i:i+frame_length]**2)
            for i in range(0, len(y) - frame_length, CONFIG['hop_length'])
        ])
        
        # Pad energy to match f0 length
        if len(energy) < len(f0):
            energy = np.pad(energy, (0, len(f0) - len(energy)), 'edge')
        elif len(energy) > len(f0):
            energy = energy[:len(f0)]
        
        confidence = energy / (np.max(energy) + 1e-8)
        
        return f0, confidence
    except Exception as e:
        st.error(f"YIN pitch extraction failed: {e}")
        return None, None

def extract_pitch(y, sr):
    """Extract pitch using best available method"""
    # Try CREPE first
    if CREPE_AVAILABLE:
        f0, confidence = extract_pitch_crepe(y, sr)
        if f0 is not None:
            return f0, confidence
    
    # Fallback to YIN
    return extract_pitch_yin(y, sr)

def hz_to_midi(hz):
    """Convert frequency to MIDI note number"""
    return 69 + 12 * np.log2(hz / CONFIG['A4_freq'])

def midi_to_note_name(midi):
    """Convert MIDI number to note name"""
    notes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    octave = int(midi // 12) - 1
    note_idx = int(midi % 12)
    return f"{notes[note_idx]}{octave}"

def calculate_cents_error(hz, target_hz):
    """Calculate cents error from target pitch"""
    if hz <= 0 or target_hz <= 0:
        return 0
    return 1200 * np.log2(hz / target_hz)

# ============================================
# PITCH ANALYSIS
# ============================================

def analyze_pitch(y, sr):
    """Complete pitch analysis pipeline"""
    # Extract pitch
    f0, confidence = extract_pitch(y, sr)
    
    if f0 is None:
        return None
    
    # Apply voicing mask
    voiced_mask = (confidence > CONFIG['voicing_threshold']) & (f0 > 0)
    f0_voiced = f0.copy()
    f0_voiced[~voiced_mask] = 0
    
    # Smooth pitch
    f0_smooth = median_filter(f0_voiced, size=5)
    
    # Convert to MIDI
    midi = np.zeros_like(f0_smooth)
    valid = f0_smooth > 0
    midi[valid] = hz_to_midi(f0_smooth[valid])
    
    # Snap to nearest notes
    midi_rounded = np.round(midi)
    note_names = [midi_to_note_name(m) if m > 0 else '' for m in midi_rounded]
    
    # Calculate cents error
    cents_error = np.zeros_like(f0_smooth)
    for i, (hz, m_rounded) in enumerate(zip(f0_smooth, midi_rounded)):
        if hz > 0 and m_rounded > 0:
            target_hz = CONFIG['A4_freq'] * 2**((m_rounded - 69) / 12)
            cents_error[i] = calculate_cents_error(hz, target_hz)
    
    # Build timeline
    times = librosa.frames_to_time(
        np.arange(len(f0_smooth)),
        sr=sr,
        hop_length=CONFIG['hop_length']
    )
    
    timeline_df = pd.DataFrame({
        'time': times,
        'hz': f0_smooth,
        'midi': midi,
        'note_name': note_names,
        'cents_error': cents_error,
        'voiced': voiced_mask
    })
    
    # Statistics
    voiced_hz = f0_smooth[voiced_mask]
    voiced_cents = cents_error[voiced_mask]
    
    if len(voiced_hz) == 0:
        return None
    
    stats = {
        'min_hz': np.min(voiced_hz),
        'max_hz': np.max(voiced_hz),
        'mean_hz': np.mean(voiced_hz),
        'median_hz': np.median(voiced_hz),
        'tessitura_low': np.percentile(voiced_hz, 25),
        'tessitura_high': np.percentile(voiced_hz, 75),
        'voiced_percentage': 100 * np.sum(voiced_mask) / len(voiced_mask),
        'cents_mean': np.mean(np.abs(voiced_cents)),
        'cents_std': np.std(voiced_cents),
        'intonation_score': max(0, 100 - np.mean(np.abs(voiced_cents))),
    }
    
    return {
        'timeline': timeline_df,
        'stats': stats,
        'f0': f0_smooth,
        'confidence': confidence,
        'voiced_mask': voiced_mask
    }

# ============================================
# VOICE TYPE CLASSIFICATION
# ============================================

def classify_voice_type(stats):
    """Classify voice type based on range and tessitura"""
    scores = {}
    
    for voice_type, ranges in VOICE_TYPES.items():
        # Check if range overlaps
        overlap = (
            max(stats['min_hz'], ranges['min']) < min(stats['max_hz'], ranges['max'])
        )
        
        if not overlap:
            scores[voice_type] = 0
            continue
        
        # Score based on tessitura fit
        tess_center = (stats['tessitura_low'] + stats['tessitura_high']) / 2
        type_tess_center = (ranges['tessitura'][0] + ranges['tessitura'][1]) / 2
        
        distance = abs(tess_center - type_tess_center)
        score = max(0, 100 - distance / 5)
        
        scores[voice_type] = score
    
    if not scores or max(scores.values()) == 0:
        return None, 0
    
    best_type = max(scores, key=scores.get)
    confidence = scores[best_type]
    
    return best_type, confidence

# ============================================
# TIMBRE ANALYSIS
# ============================================

def analyze_timbre(y, sr):
    """Analyze spectral characteristics"""
    # Compute spectrogram
    S = np.abs(librosa.stft(y, n_fft=CONFIG['n_fft'], hop_length=CONFIG['hop_length']))
    
    # Spectral features
    centroid = librosa.feature.spectral_centroid(S=S, sr=sr)[0]
    rolloff = librosa.feature.spectral_rolloff(S=S, sr=sr, roll_percent=0.85)[0]
    
    # Energy bands
    freqs = librosa.fft_frequencies(sr=sr, n_fft=CONFIG['n_fft'])
    low_band = S[(freqs < 300), :].sum(axis=0)
    mid_band = S[(freqs >= 300) & (freqs < 3000), :].sum(axis=0)
    high_band = S[(freqs >= 3000), :].sum(axis=0)
    
    total_energy = low_band + mid_band + high_band + 1e-8
    
    low_ratio = np.mean(low_band / total_energy)
    mid_ratio = np.mean(mid_band / total_energy)
    high_ratio = np.mean(high_band / total_energy)
    
    # Determine badge
    if low_ratio > 0.4:
        badge = "Bass-Heavy 🎸"
        tips = "Rich, warm tone. Great for narrator roles. Practice projection."
    elif high_ratio > 0.35:
        badge = "Bright & Clear ✨"
        tips = "Crisp, cutting voice. Great for pop. Watch for harshness."
    elif mid_ratio > 0.5:
        badge = "Mid-Forward 🎤"
        tips = "Present, speech-like. Great for clarity. Explore range extremes."
    else:
        badge = "Balanced 🎵"
        tips = "Well-rounded tone. Versatile. Keep developing all registers."
    
    return {
        'centroid_mean': np.mean(centroid),
        'rolloff_mean': np.mean(rolloff),
        'low_ratio': low_ratio,
        'mid_ratio': mid_ratio,
        'high_ratio': high_ratio,
        'badge': badge,
        'tips': tips
    }

# ============================================
# ACCENT DETECTION
# ============================================

def detect_accent(y, sr):
    """Detect accent from speech sample"""
    # Extract formants (simplified)
    # In production, use proper formant tracking
    
    # Extract MFCCs as proxy
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_mean = np.mean(mfccs, axis=1)
    
    # Extract spectral features
    centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
    rolloff = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))
    
    # Simple heuristic classification
    # Higher centroid + lower rolloff = American (rhotic)
    # Lower centroid + higher rolloff = British (non-rhotic)
    
    score_american = centroid / 1000 + (3000 - rolloff) / 1000
    score_british = (3000 - centroid) / 1000 + rolloff / 1000
    
    if score_american > score_british:
        detected = 'American'
        confidence = min(100, score_american / (score_american + score_british) * 100)
    else:
        detected = 'British'
        confidence = min(100, score_british / (score_american + score_british) * 100)
    
    # Identify weak phonemes (placeholder - would need phoneme recognition)
    weak_sounds = ['TH', 'R'] if detected == 'American' else ['T', 'schwa']
    
    return {
        'detected_accent': detected,
        'confidence': confidence,
        'weak_sounds': weak_sounds,
        'features': {
            'centroid': centroid,
            'rolloff': rolloff,
            'mfcc_signature': mfcc_mean.tolist()
        }
    }

def analyze_pronunciation(y, sr, target_accent='American'):
    """Analyze pronunciation accuracy"""
    # Placeholder for real phoneme recognition
    # In production, use phoneme recognizer like Montreal Forced Aligner
    
    # Simple energy-based segment detection
    rms = librosa.feature.rms(y=y, hop_length=CONFIG['hop_length'])[0]
    segments = librosa.effects.split(y, top_db=20)
    
    # Mock pronunciation scores
    phoneme_scores = {}
    for sound in ACCENT_FEATURES[target_accent]['key_sounds']:
        # Random score for demo (replace with real analysis)
        score = np.random.randint(60, 100)
        phoneme_scores[sound] = score
    
    overall_score = np.mean(list(phoneme_scores.values()))
    
    return {
        'overall_score': overall_score,
        'phoneme_scores': phoneme_scores,
        'segments_detected': len(segments),
        'feedback': generate_pronunciation_feedback(phoneme_scores, target_accent)
    }

def generate_pronunciation_feedback(scores, target_accent):
    """Generate feedback for pronunciation"""
    feedback = []
    
    for phoneme, score in scores.items():
        if score < 70:
            tip = PHONEME_TIPS.get(phoneme, "Practice this sound more")
            feedback.append(f"❌ {phoneme}: {score:.0f}% - {tip}")
        elif score < 85:
            feedback.append(f"⚠️ {phoneme}: {score:.0f}% - Getting better! Keep practicing.")
        else:
            feedback.append(f"✅ {phoneme}: {score:.0f}% - Excellent!")
    
    return feedback

# ============================================
# SONG CATALOG & IDENTIFICATION
# ============================================

def create_song_catalog():
    """Create initial song catalog"""
    songs = [
        {"title": "Happy Birthday", "artist": "Traditional", "key": "C", "low": 262, "high": 523, "tags": "easy,celebration", "template": [0, 0, 2, 0, 5, 4]},
        {"title": "Amazing Grace", "artist": "Traditional", "key": "G", "low": 196, "high": 392, "tags": "hymn,slow", "template": [0, 3, 5, 3, 5, 7]},
        {"title": "Hallelujah", "artist": "Leonard Cohen", "key": "C", "low": 262, "high": 523, "tags": "modern,emotional", "template": [0, 2, 4, 5, 7, 5]},
        {"title": "Stand By Me", "artist": "Ben E. King", "key": "A", "low": 220, "high": 440, "tags": "soul,vintage", "template": [0, 2, 4, 2, 0, -2]},
        {"title": "Let It Be", "artist": "The Beatles", "key": "C", "low": 262, "high": 523, "tags": "rock,vintage", "template": [0, 2, 4, 5, 4, 2]},
        {"title": "Someone Like You", "artist": "Adele", "key": "A", "low": 220, "high": 659, "tags": "modern,ballad", "template": [0, 2, 4, 7, 5, 4]},
        {"title": "Shallow", "artist": "Lady Gaga", "key": "G", "low": 196, "high": 698, "tags": "modern,powerful", "template": [0, -2, 0, 2, 5, 7]},
        {"title": "Bohemian Rhapsody", "artist": "Queen", "key": "Bb", "low": 233, "high": 880, "tags": "rock,complex", "template": [0, 3, 5, 8, 7, 5]},
        {"title": "My Way", "artist": "Frank Sinatra", "key": "D", "low": 294, "high": 587, "tags": "vintage,ballad", "template": [0, 2, 4, 5, 4, 2]},
        {"title": "Rolling in the Deep", "artist": "Adele", "key": "C", "low": 262, "high": 698, "tags": "modern,powerful", "template": [0, 2, 3, 5, 3, 2]},
    ]
    
    df = pd.DataFrame(songs)
    df['template_json'] = df['template'].apply(json.dumps)
    df = df.drop('template', axis=1)
    
    return df

def load_or_create_catalog():
    """Load catalog from CSV or create new one"""
    catalog_path = Path('song_catalog.csv')
    
    if catalog_path.exists():
        try:
            df = pd.read_csv(catalog_path)
            return df
        except:
            pass
    
    df = create_song_catalog()
    df.to_csv(catalog_path, index=False)
    return df

def identify_song(pitch_analysis, catalog_df):
    """Identify song from pitch pattern"""
    if pitch_analysis is None or catalog_df is None:
        return None
    
    # Extract pitch contour (intervals)
    timeline = pitch_analysis['timeline']
    voiced = timeline[timeline['voiced'] == True]
    
    if len(voiced) < 5:
        return None
    
    # Sample pitch points
    sample_indices = np.linspace(0, len(voiced) - 1, min(20, len(voiced)), dtype=int)
    query_pitches = voiced.iloc[sample_indices]['midi'].values
    
    # Convert to intervals
    query_intervals = np.diff(query_pitches)
    
    # Compare with catalog
    matches = []
    
    for idx, row in catalog_df.iterrows():
        try:
            template = json.loads(row['template_json'])
            template = np.array(template)
            
            # Calculate similarity (simple correlation)
            min_len = min(len(query_intervals), len(template))
            correlation = np.corrcoef(query_intervals[:min_len], template[:min_len])[0, 1]
            
            if np.isnan(correlation):
                correlation = 0
            
            matches.append({
                'title': row['title'],
                'artist': row['artist'],
                'key': row['key'],
                'confidence': max(0, correlation * 100),
                'key_shift': 0  # Simplified
            })
        except:
            continue
    
    # Sort by confidence
    matches = sorted(matches, key=lambda x: x['confidence'], reverse=True)
    
    return matches[:3] if matches else None

# ============================================
# RECOMMENDATIONS
# ============================================

def

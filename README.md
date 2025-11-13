# Streamlit Voice Studio

A comprehensive voice analysis and training platform built with Streamlit.

## Features

- **Voice Analysis**: Pitch detection, voice type classification, timbre analysis
- **Accent Training**: American/British accent detection and training
- **Singing Training**: Real-time singing performance analysis
- **Song Recommendations**: Personalized song suggestions based on voice analysis
- **Dual Themes**: Modern (glassy) and Vintage (Windows 98) themes
- **Privacy First**: All processing happens locally on your device

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the application:
```bash
streamlit run app.py
```

3. Open your browser to the URL shown in the terminal (usually http://localhost:8501)

## Usage

1. **Record/Upload**: Use live recording or upload audio files
2. **Voice Analysis**: Get detailed analysis of your voice characteristics
3. **Accent Training**: Train American or British accent with real-time feedback
4. **Singing Training**: Practice singing with pitch and stability analysis
5. **Charts & Data**: View detailed visualizations of your voice data
6. **Recommendations**: Get personalized song recommendations

## System Requirements

- Python 3.10+
- Microphone for live recording
- Modern web browser with WebRTC support
- At least 4GB RAM recommended

## Notes

- The app works best with clear audio input
- For best results, use a quiet environment when recording
- All data is processed locally - no audio is sent to external servers
- Some features may require additional dependencies for full functionality

## Troubleshooting

If you encounter issues:
1. Ensure all dependencies are installed correctly
2. Check browser permissions for microphone access
3. Try refreshing the page if recording doesn't work
4. For WebRTC issues, ensure you're using HTTPS or localhost
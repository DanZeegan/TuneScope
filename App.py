st.markdown(f"""
                            <div class='warning-card'>
                                <b>🔧 Troubleshooting:</b><br>
                                • Detected {results.get('voiced_ratio', 0)*100:.1f}% voice<br>
                                • Lower threshold to {voicing_threshold-0.05:.2f} in sidebar<br>
                                • Sing/speak louder and longer<br>
                                • Check microphone is working
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            st.session_state.analysis_results = results
                            
                            # Log practice session
                            st.session_state.practice_log.append({
                                'date': datetime.datetime.now().strftime("%m/%d %H:%M"),
                                'score': int(results['singing_accuracy']['accuracy_score'])
                            })
                            
                            st.success("✅ **Analysis Complete!** Check other tabs for full results.")
                            st.balloons()
                            
                            # Quick preview
                            col_p1, col_p2, col_p3 = st.columns(3)
                            with col_p1:
                                st.metric("🎵 Voice Type", results['voice_type']['type'])
                            with col_p2:
                                st.metric("🎯 Accuracy", f"{results['singing_accuracy']['accuracy_score']:.0f}/100")
                            with col_p3:
                                st.metric("🎼 Range", f"{results['min_note']}-{results['max_note']}")
                    
                    except Exception as e:
                        st.error(f"❌ Analysis failed: {str(e)}")
                        st.info("💡 Try a different audio file or check format")
                    
                    finally:
                        progress_bar.empty()
                        status.empty()
        
        with col_demo:
            if st.button("🧪 Try Demo", use_container_width=True):
                sr = 22050
                freqs = [261.63, 293.66, 329.63, 349.23, 392.00, 440.00, 493.88, 523.25]
                y = np.concatenate([
                    0.3 * np.sin(2*np.pi*f*np.linspace(0, 0.35, int(sr*0.35))) * 
                    np.exp(-np.linspace(0, 0.35, int(sr*0.35))*2.5) 
                    for f in freqs
                ])
                st.session_state.audio_data = (y, sr)
                st.success("✅ Demo audio ready! Click Analyze.")
        
        with col_clear:
            if st.button("🗑️ Clear", use_container_width=True):
                st.session_state.audio_data = None
                st.session_state.analysis_results = None
                st.rerun()
    
    # TAB 2: Detailed Analysis
    with tab2:
        st.markdown("## 📊 Your Complete Voice Analysis")
        
        if not st.session_state.analysis_results:
            st.info("👆 Record and analyze your voice first in the **Record & Analyze** tab")
        else:
            results = st.session_state.analysis_results
            
            if 'error' in results:
                st.error(results['error'])
            else:
                # Key Metrics Cards
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.markdown(f"""
                    <div class='metric-card'>
                        <h3 style='color:white;margin:0;'>🎵 Vocal Range</h3>
                        <p style='font-size:1.8rem;font-weight:bold;margin:0.5rem 0;'>
                            {results['min_note']} - {results['max_note']}
                        </p>
                        <p style='margin:0;opacity:0.9;'>
                            {int(results['max_note_midi']-results['min_note_midi'])} semitones span
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    voice = results['voice_type']
                    st.markdown(f"""
                    <div class='metric-card'>
                        <h3 style='color:white;margin:0;'>🎭 Voice Type</h3>
                        <p style='font-size:1.8rem;font-weight:bold;margin:0.5rem 0;'>
                            {voice['type']}
                        </p>
                        <p style='margin:0;opacity:0.9;'>
                            {voice['confidence']*100:.0f}% confidence
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col3:
                    accuracy = results['singing_accuracy']['accuracy_score']
                    st.markdown(f"""
                    <div class='metric-card'>
                        <h3 style='color:white;margin:0;'>🎯 Singing Accuracy</h3>
                        <p style='font-size:1.8rem;font-weight:bold;margin:0.5rem 0;'>
                            {accuracy:.0f}/100
                        </p>
                        <p style='margin:0;opacity:0.9;'>
                            {'Excellent!' if accuracy >= 85 else 'Good!' if accuracy >= 70 else 'Keep practicing!'}
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col4:
                    st.markdown(f"""
                    <div class='metric-card'>
                        <h3 style='color:white;margin:0;'>🎨 Voice Tone</h3>
                        <p style='font-size:1.8rem;font-weight:bold;margin:0.5rem 0;'>
                            {results['timbre_classification']}
                        </p>
                        <p style='margin:0;opacity:0.9;'>
                            Spectral signature
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                
                st.markdown("---")
                
                # Detailed Breakdown
                col_left, col_right = st.columns(2)
                
                with col_left:
                    st.markdown("### 📏 Range Details")
                    
                    st.markdown(f"""
                    <div class='info-card'>
                        <b>🎼 Your Tessitura (Comfort Zone):</b><br>
                        {results['tessitura_low']} to {results['tessitura_high']}<br>
                        <small>This is where your voice sounds best and feels most comfortable</small>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.markdown(f"""
                    <div class='info-card'>
                        <b>🎭 Voice Type: {voice['type']}</b><br>
                        {voice['description']}<br>
                        <small>Confidence: {voice['confidence']*100:.0f}%</small>
                        {f"<br><small>Borderline with {voice['alternative']}</small>" if voice['is_borderline'] else ""}
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Voice type scores
                    st.markdown("**Voice Type Match Scores:**")
                    for vtype, score in list(voice['all_scores'].items())[:3]:
                        score_val = float(score)
                        st.progress(max(0.0, min(1.0, score_val)), text=f"{vtype}: {score_val*100:.0f}%")
                
                with col_right:
                    st.markdown("### 🎯 Singing Performance")
                    
                    sing_acc = results['singing_accuracy']
                    
                    if sing_acc['accuracy_score'] >= 85:
                        st.markdown("""
                        <div class='success-card'>
                            🌟 <b>Excellent Performance!</b><br>
                            Your pitch accuracy is outstanding. You're hitting notes consistently and accurately.
                        </div>
                        """, unsafe_allow_html=True)
                    elif sing_acc['accuracy_score'] >= 70:
                        st.markdown("""
                        <div class='info-card'>
                            👍 <b>Good Performance!</b><br>
                            You're doing well with pitch accuracy. Focus on consistency for improvement.
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown("""
                        <div class='warning-card'>
                            📚 <b>Room for Growth!</b><br>
                            Focus on ear training and pitch matching exercises. Practice with a piano or tuner.
                        </div>
                        """, unsafe_allow_html=True)
                    
                    st.markdown(f"""
                    <div class='info-card'>
                        <b>📊 Performance Metrics:</b><br>
                        • Average pitch error: ±{sing_acc['mean_cents_error']:.1f} cents<br>
                        • Consistency: {sing_acc['consistency']:.0f}/100<br>
                        • Stable notes: {sing_acc['stable_notes_percent']:.0f}%<br>
                        {'• ✨ Vibrato detected!' if sing_acc['has_vibrato'] else '• No vibrato detected'}
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.markdown("**💡 Personalized Feedback:**")
                    for feedback in sing_acc['feedback']:
                        st.info(feedback)
                
                st.markdown("---")
                
                # Visualizations
                st.markdown("### 📈 Visual Analysis")
                
                # Pitch contour
                fig_pitch = plot_pitch_contour(results)
                if fig_pitch:
                    st.plotly_chart(fig_pitch, use_container_width=True)
                
                col_v1, col_v2 = st.columns(2)
                
                with col_v1:
                    fig_notes = plot_note_distribution(results)
                    st.plotly_chart(fig_notes, use_container_width=True)
                
                with col_v2:
                    fig_spectral = plot_spectral_profile(results)
                    st.plotly_chart(fig_spectral, use_container_width=True)
                
                # Export Options
                st.markdown("---")
                st.markdown("### 💾 Export Your Results")
                
                col_e1, col_e2, col_e3 = st.columns(3)
                
                with col_e1:
                    export_data = {k: v for k, v in results.items() if k not in ['pitch_contour']}
                    # Convert numpy types to Python types
                    export_data_clean = json.loads(json.dumps(export_data, default=lambda x: float(x) if isinstance(x, (np.floating, np.integer)) else x))
                    json_str = json.dumps(export_data_clean, indent=2)
                    st.download_button(
                        "📄 Download JSON",
                        json_str,
                        "voice_analysis.json",
                        "application/json",
                        use_container_width=True
                    )
                
                with col_e2:
                    contour = results['pitch_contour']
                    df_export = pd.DataFrame({
                        'time': contour['times'],
                        'pitch_hz': contour['pitch_hz'],
                        'confidence': contour['confidence']
                    })
                    csv = df_export.to_csv(index=False)
                    st.download_button(
                        "📊 Download CSV",
                        csv,
                        "pitch_data.csv",
                        "text/csv",
                        use_container_width=True
                    )
                
                with col_e3:
                    summary = f"""TuneScope Pro - Voice Analysis Report
{'='*50}

📅 Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}

🎵 VOCAL RANGE
Range: {results['min_note']} - {results['max_note']}
Span: {int(results['max_note_midi']-results['min_note_midi'])} semitones
Tessitura: {results['tessitura_low']} - {results['tessitura_high']}

🎭 VOICE TYPE
Type: {voice['type']}
Description: {voice['description']}
Confidence: {voice['confidence']*100:.0f}%

🎯 SINGING ACCURACY
Score: {sing_acc['accuracy_score']:.0f}/100
Mean Error: ±{sing_acc['mean_cents_error']:.1f} cents
Consistency: {sing_acc['consistency']:.0f}/100
Stable Notes: {sing_acc['stable_notes_percent']:.0f}%
Vibrato: {'Yes' if sing_acc['has_vibrato'] else 'No'}

🎨 VOICE TONE
Classification: {results['timbre_classification']}

📊 TECHNICAL DETAILS
Duration: {results['duration']:.2f} seconds
Voiced Ratio: {results['voiced_ratio']*100:.1f}%
Detection Method: {results['pitch_method']}

{'='*50}
Generated by TuneScope Pro
"""
                    st.download_button(
                        "📝 Download Report",
                        summary,
                        "voice_report.txt",
                        "text/plain",
                        use_container_width=True
                    )
    
    # TAB 3: Song Finder
    with tab3:
        st.markdown("## 🎵 Find Perfect Songs for Your Voice (100+ Songs)")
        
        if not st.session_state.analysis_results or 'error' in st.session_state.analysis_results:
            st.info("👆 Analyze your voice first to get personalized song recommendations")
        else:
            st.markdown("### 🔍 Filter Songs by Your Preferences")
            
            col_f1, col_f2, col_f3, col_f4 = st.columns(4)
            
            with col_f1:
                era_filter = st.selectbox(
                    "🕰️ Era",
                    ['All', 'traditional', 'classical', 'vintage', '80s-90s', '2000s', '2010s', '2020s']
                )
            
            with col_f2:
                genre_filter = st.selectbox(
                    "🎸 Genre",
                    ['All', 'pop', 'rock', 'R&B', 'soul', 'jazz', 'folk', 'ballad', 'EDM', 'musical']
                )
            
            with col_f3:
                mood_filter = st.selectbox(
                    "😊 Mood",
                    ['All', 'romantic', 'uplifting', 'emotional', 'energetic', 'peaceful', 'confident', 'sad']
                )
            
            with col_f4:
                difficulty_filter = st.selectbox(
                    "📊 Difficulty",
                    ['All', 'beginner', 'intermediate', 'advanced']
                )
            
            # Build preferences
            user_prefs = {}
            if era_filter != 'All':
                user_prefs['era'] = era_filter
            if genre_filter != 'All':
                user_prefs['genre'] = genre_filter
            if mood_filter != 'All':
                user_prefs['mood'] = mood_filter
            if difficulty_filter != 'All':
                user_prefs['difficulty'] = difficulty_filter
            
            with st.spinner("Finding perfect songs for you..."):
                recommendations = recommend_songs(
                    st.session_state.analysis_results,
                    st.session_state.catalog,
                    user_prefs
                )
            
            # Perfect Fit Songs
            st.markdown("---")
            st.markdown("### ✅ Perfect Match Songs (Recommended)")
            st.caption(f"Found {len(recommendations['fit'])} songs that fit your voice perfectly")
            
            if len(recommendations['fit']) == 0:
                st.info("No perfect matches with current filters. Try adjusting filters or check Stretch Songs below!")
            else:
                for idx, song in enumerate(recommendations['fit'], 1):
                    with st.expander(f"🎵 {idx}. **{song['title']}** by {song['artist']} • {song['era']} • {song['mood']}"):
                        col_s1, col_s2 = st.columns([2, 1])
                        
                        with col_s1:
                            st.markdown(f"**🎼 Key:** {song['key']} | **📏 Range:** {song['range']}")
                            st.markdown(f"**🎸 Genre:** {song['genre']} | **😊 Mood:** {song['mood']}")
                            
                            difficulty_emoji = {'beginner': '🟢', 'intermediate': '🟡', 'advanced': '🔴'}
                            st.markdown(f"**{difficulty_emoji.get(song['difficulty'], '⚪')} Difficulty:** {song['difficulty'].title()}")
                            
                            st.markdown("**Why this song:**")
                            for reason in song['reasons']:
                                st.markdown(f"• {reason}")
                        
                        with col_s2:
                            st.markdown(f"### {difficulty_emoji.get(song['difficulty'], '⚪')}")
                            st.markdown(f"**{song['difficulty'].upper()}**")
                            
                            if st.button(f"🎤 Practice This", key=f"practice_{idx}"):
                                st.success(f"Great choice! Start with **{song['title']}**")
                                st.info(f"💡 **Tip:** Warm up with scales in key of {song['key']} first!")
            
            # Stretch Songs
            st.markdown("---")
            st.markdown("### 📈 Growth Songs (Expand Your Range)")
            st.caption(f"Found {len(recommendations['stretch'])} songs to help you grow")
            
            if len(recommendations['stretch']) > 0:
                for song in recommendations['stretch'][:5]:
                    with st.expander(f"🎵 **{song['title']}** - {song['artist']}"):
                        st.markdown(f"**Range:** {song['range']} | **Difficulty:** {song['difficulty']} | **Era:** {song['era']}")
                        for reason in song['reasons']:
                            st.markdown(f"• {reason}")
                        st.warning("⚠️ Warm up thoroughly before attempting stretch songs!")
            
            # Avoid Songs
            with st.expander("⚠️ Songs to Avoid (For Now)"):
                if len(recommendations['avoid']) > 0:
                    st.caption("These songs may strain your voice with your current range")
                    for song in recommendations['avoid'][:5]:
                        st.markdown(f"**{song['title']}** - {song['artist']}")
                        for reason in song['reasons']:
                            st.markdown(f"  {reason}")
                else:
                    st.success("Great! No songs are too difficult for you!")
    
    # TAB 4: Training Plans
    with tab4:
        st.markdown("## 🏋️ Personalized Voice Training Plans")
        
        if st.session_state.analysis_results and 'error' not in st.session_state.analysis_results:
            results = st.session_state.analysis_results
            voice_type = results['voice_type']['type']
            accuracy = results['singing_accuracy']['accuracy_score']
            
            # Determine skill level
            if accuracy >= 85:
                skill_level = 'Advanced'
            elif accuracy >= 70:
                skill_level = 'Intermediate'
            else:
                skill_level = 'Beginner'
            
            st.markdown(f"""
            <div class='success-card'>
                <b>👤 Your Profile:</b> {voice_type} • {skill_level} Level • {accuracy:.0f}/100 Accuracy
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("---")
            
            # Daily Routine
            st.markdown(f"### 📅 Your {skill_level} Daily Routine")
            
            routine = PRACTICE_ROUTINES[skill_level]
            
            st.markdown(f"""
            <div class='info-card'>
                <b>⏰ Duration:</b> {routine['duration']}<br><br>
                <b>📋 Your Routine:</b>
            </div>
            """, unsafe_allow_html=True)
            
            for step in routine['routine']:
                st.markdown(f"✓ {step}")
            
            st.markdown("---")
            
            # Specific Exercises
            st.markdown("### 🎯 Exercises for Your Voice Type")
            
            col_ex1, col_ex2 = st.columns(2)
            
            with col_ex1:
                st.markdown("#### 🔥 Warm-up (Essential)")
                for exercise in VOCAL_EXERCISES['Warm-up']:
                    st.markdown(f"• {exercise}")
                
                st.markdown("#### 🎵 Pitch Accuracy")
                for exercise in VOCAL_EXERCISES['Pitch Accuracy']:
                    st.markdown(f"• {exercise}")
            
            with col_ex2:
                st.markdown("#### 🫁 Breath Control")
                for exercise in VOCAL_EXERCISES['Breath Control']:
                    st.markdown(f"• {exercise}")
                
                st.markdown("#### 📈 Range Extension")
                for exercise in VOCAL_EXERCISES['Range Extension']:
                    st.markdown(f"• {exercise}")
            
            st.markdown("---")
            
            # Weekly Plan
            st.markdown("### 📆 7-Day Practice Plan")
            
            weekly_plan = {
                'Monday': ['Warm-up', 'Breath exercises', 'Scale practice', 'Easy song'],
                'Tuesday': ['Warm-up', 'Pitch training', 'Interval work', 'Medium song'],
                'Wednesday': ['Light warm-up', 'Tone exercises', 'Rest day (light practice)'],
                'Thursday': ['Full warm-up', 'Range extension', 'Technical exercises', 'Challenging song'],
                'Friday': ['Warm-up', 'Repertoire practice', '2-3 songs', 'Performance prep'],
                'Saturday': ['Warm-up', 'Free exploration', 'Try new songs', 'Fun practice'],
                'Sunday': ['Gentle exercises', 'Review week', 'Light singing', 'Rest & recovery']
            }
            
            for day, activities in weekly_plan.items():
                with st.expander(f"📅 {day}"):
                    for activity in activities:
                        st.markdown(f"• {activity}")
        
        else:
            st.info("Analyze your voice first to get a personalized training plan!")
    
    # TAB 5: Singing Coach
    with tab5:
        st.markdown("## 📈 Real-Time Singing Coach")
        
        st.markdown("""
        <div class='info-card'>
            <b>🎤 How to Use Singing Coach:</b><br>
            1. Choose a reference song or note<br>
            2. Sing along while recording<br>
            3. Get instant feedback on accuracy<br>
            4. See where you're sharp/flat<br>
            5. Practice problem areas
        </div>
        """, unsafe_allow_html=True)
        
        if st.session_state.analysis_results and 'error' not in st.session_state.analysis_results:
            results = st.session_state.analysis_results
            
            st.markdown("### 🎯 Your Singing Performance Analysis")
            
            sing_acc = results['singing_accuracy']
            
            # Performance Gauge
            col_g1, col_g2, col_g3 = st.columns(3)
            
            with col_g1:
                accuracy_pct = sing_acc['accuracy_score'] / 100
                fig_gauge = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=sing_acc['accuracy_score'],
                    title={'text': "Pitch Accuracy"},
                    gauge={'axis': {'range': [0, 100]},
                           'bar': {'color': "#3b82f6"},
                           'steps': [
                               {'range': [0, 60], 'color': "#fecaca"},
                               {'range': [60, 85], 'color': "#fde68a"},
                               {'range': [85, 100], 'color': "#bbf7d0"}
                           ]}
                ))
                fig_gauge.update_layout(height=250)
                st.plotly_chart(fig_gauge, use_container_width=True)
            
            with col_g2:
                consistency_pct = sing_acc['consistency'] / 100
                fig_consist = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=sing_acc['consistency'],
                    title={'text': "Consistency"},
                    gauge={'axis': {'range': [0, 100]},
                           'bar': {'color': "#8b5cf6"}}
                ))
                fig_consist.update_layout(height=250)
                st.plotly_chart(fig_consist, use_container_width=True)
            
            with col_g3:
                fig_stable = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=sing_acc['stable_notes_percent'],
                    title={'text': "Stable Notes %"},
                    gauge={'axis': {'range': [0, 100]},
                           'bar': {'color': "#10b981"}}
                ))
                fig_stable.update_layout(height=250)
                st.plotly_chart(fig_stable, use_container_width=True)
            
            st.markdown("---")
            
            # Detailed Feedback
            st.markdown("### 💡 Personalized Coaching Tips")
            
            col_t1, col_t2 = st.columns(2)
            
            with col_t1:
                st.markdown("#### ✅ Strengths")
                if sing_acc['accuracy_score'] >= 80:
                    st.success("✓ Excellent pitch control")
                if sing_acc['consistency'] >= 80:
                    st.success("✓ Very consistent singing")
                if sing_acc['stable_notes_percent'] >= 70:
                    st.success("✓ Good note stability")
                if sing_acc['has_vibrato']:
                    st.success("✓ Natural vibrato present")
            
            with col_t2:
                st.markdown("#### 📚 Areas to Improve")
                if sing_acc['accuracy_score'] < 80:
                    st.warning("• Practice pitch matching with piano")
                if sing_acc['consistency'] < 80:
                    st.warning("• Work on maintaining steady pitch")
                if sing_acc['stable_notes_percent'] < 70:
                    st.warning("• Practice holding long notes")
                if not sing_acc['has_vibrato']:
                    st.info("• Vibrato will develop naturally with practice")
            
            st.markdown("---")
            
            # Practice Recommendations
            st.markdown("### 🎯 Recommended Next Steps")
            
            if sing_acc['accuracy_score'] < 70:
                st.markdown("""
                <div class='warning-card'>
                    <b>Focus on Fundamentals:</b><br>
                    1. Practice matching single notes on a piano<br>
                    2. Sing simple scales slowly (C major)<br>
                    3. Use a tuner app while practicing<br>
                    4. Record yourself and listen back<br>
                    5. Practice 15-20 minutes daily
                </div>
                """, unsafe_allow_html=True)
            elif sing_acc['accuracy_score'] < 85:
                st.markdown("""
                <div class='info-card'>
                    <b>Intermediate Development:</b><br>
                    1. Practice intervals (3rds, 5ths, octaves)<br>
                    2. Sing along with favorite songs<br>
                    3. Work on difficult passages slowly<br>
                    4. Record and compare to original<br>
                    5. Practice 20-30 minutes daily
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class='success-card'>
                    <b>Advanced Refinement:</b><br>
                    1. Focus on dynamics and expression<br>
                    2. Explore complex repertoire<br>
                    3. Work on performance skills<br>
                    4. Consider vocal coaching<br>
                    5. Maintain daily practice routine
                </div>
                """, unsafe_allow_html=True)
        
        else:
            st.info("Record and analyze your singing to get coaching feedback!")
    
    # TAB 6: Vocal Tips
    with tab6:
        st.markdown("## 🎓 Professional Vocal Tips & Techniques")
        
        tip_category = st.selectbox(
            "Choose a topic:",
            ['Vocal Health', 'Breathing Technique', 'Pitch Training', 'Range Extension', 'Performance Tips', 'Common Mistakes']
        )
        
        if tip_category == 'Vocal Health':
            st.markdown("""
            ### 🏥 Vocal Health & Care
            
            #### Daily Habits:
            • **Hydration**: Drink 8+ glasses of water daily
            • **Sleep**: Get 7-8 hours for vocal recovery
            • **Avoid**: Smoking, excessive alcohol, shouting
            • **Humidifier**: Use in dry environments
            
            #### Before Singing:
            ✓ Warm up for 10-15 minutes
            ✓ Drink room temperature water
            ✓ Avoid dairy products (creates mucus)
            ✓ Don't sing on a full stomach
            
            #### After Singing:
            ✓ Cool down with gentle exercises
            ✓ Rest your voice
            ✓ Stay hydrated
            ✓ Avoid whispering (strains cords)
            
            #### Warning Signs:
            🚨 Stop singing if you experience:
            • Pain or sharp discomfort
            • Persistent hoarseness (>2 weeks)
            • Loss of range
            • Voice breaks or cracks
            
            ⚠️ **See a doctor if symptoms persist!**
            """)
        
        elif tip_category == 'Breathing Technique':
            st.markdown("""
            ### 🫁 Proper Breathing for Singing
            
            #### Diaphragmatic Breathing:
            1. **Place hand on belly**: Feel it expand
            2. **Inhale deeply**: Belly should push out
            3. **Shoulders relaxed**: Don't lift them
            4. **Slow exhale**: Control the airflow
            
            #### Exercises:
            **Exercise 1 - Hissing:**
            • Inhale for 4 counts
            • Hiss "ssss" for 8-16 counts
            • Gradually increase duration
            
            **Exercise            {'title': 'Happy', 'artist': 'Pharrell Williams', 'key': 'F', 'typical_low': 'F3', 'typical_high': 'C5', 'era': '2010s', 'genre': 'pop', 'tags': 'upbeat,treble-bright', 'difficulty': 'intermediate', 'mood': 'joyful'},
            {'title': 'Uptown Funk', 'artist': 'Bruno Mars', 'key': 'Dm', 'typical_low': 'D3', 'typical_high': 'D5', 'era': '2010s', 'genre': 'funk', 'tags': 'energetic,mid-forward', 'difficulty': 'intermediate', 'mood': 'party'},
            {'title': 'All of Me', 'artist': 'John Legend', 'key': 'Ab', 'typical_low': 'Eb3', 'typical_high': 'Eb5', 'era': '2010s', 'genre': 'pop', 'tags': 'romantic,balanced', 'difficulty': 'intermediate', 'mood': 'romantic'},
            {'title': 'Hello', 'artist': 'Adele', 'key': 'Fm', 'typical_low': 'F3', 'typical_high': 'C5', 'era': '2010s', 'genre': 'pop', 'tags': 'dramatic,mid-forward', 'difficulty': 'advanced', 'mood': 'nostalgic'},
            {'title': 'Let It Go', 'artist': 'Idina Menzel', 'key': 'Ab', 'typical_low': 'Eb3', 'typical_high': 'Eb5', 'era': '2010s', 'genre': 'musical', 'tags': 'powerful,treble-bright', 'difficulty': 'advanced', 'mood': 'empowering'},
            
            # 2020s Contemporary
            {'title': 'Blinding Lights', 'artist': 'The Weeknd', 'key': 'Fm', 'typical_low': 'F3', 'typical_high': 'C5', 'era': '2020s', 'genre': 'pop', 'tags': 'synth-pop,treble-bright', 'difficulty': 'intermediate', 'mood': 'energetic'},
            {'title': 'Drivers License', 'artist': 'Olivia Rodrigo', 'key': 'Bb', 'typical_low': 'Bb3', 'typical_high': 'F5', 'era': '2020s', 'genre': 'pop', 'tags': 'emotional,balanced', 'difficulty': 'intermediate', 'mood': 'heartbroken'},
            {'title': 'Levitating', 'artist': 'Dua Lipa', 'key': 'F#', 'typical_low': 'F#3', 'typical_high': 'C#5', 'era': '2020s', 'genre': 'pop', 'tags': 'disco,treble-bright', 'difficulty': 'intermediate', 'mood': 'fun'},
            {'title': 'Good 4 U', 'artist': 'Olivia Rodrigo', 'key': 'F', 'typical_low': 'F3', 'typical_high': 'C5', 'era': '2020s', 'genre': 'pop-rock', 'tags': 'energetic,mid-forward', 'difficulty': 'intermediate', 'mood': 'angry'},
            {'title': 'Heat Waves', 'artist': 'Glass Animals', 'key': 'Db', 'typical_low': 'Db3', 'typical_high': 'Ab4', 'era': '2020s', 'genre': 'indie', 'tags': 'chill,balanced', 'difficulty': 'beginner', 'mood': 'nostalgic'},
            {'title': 'As It Was', 'artist': 'Harry Styles', 'key': 'C#m', 'typical_low': 'C#3', 'typical_high': 'G#4', 'era': '2020s', 'genre': 'pop', 'tags': 'indie-pop,mid-forward', 'difficulty': 'beginner', 'mood': 'reflective'},
            {'title': 'Anti-Hero', 'artist': 'Taylor Swift', 'key': 'C', 'typical_low': 'C3', 'typical_high': 'G4', 'era': '2020s', 'genre': 'pop', 'tags': 'storytelling,balanced', 'difficulty': 'beginner', 'mood': 'self-aware'},
            {'title': 'Flowers', 'artist': 'Miley Cyrus', 'key': 'G', 'typical_low': 'D3', 'typical_high': 'G4', 'era': '2020s', 'genre': 'pop', 'tags': 'empowering,mid-forward', 'difficulty': 'beginner', 'mood': 'confident'},
            {'title': 'Vampire', 'artist': 'Olivia Rodrigo', 'key': 'Cm', 'typical_low': 'C3', 'typical_high': 'Bb4', 'era': '2020s', 'genre': 'pop', 'tags': 'dramatic,treble-bright', 'difficulty': 'intermediate', 'mood': 'vengeful'},
            {'title': 'Cruel Summer', 'artist': 'Taylor Swift', 'key': 'A', 'typical_low': 'E3', 'typical_high': 'C#5', 'era': '2020s', 'genre': 'pop', 'tags': 'synth-pop,treble-bright', 'difficulty': 'intermediate', 'mood': 'passionate'},
            
            # R&B/Soul Modern
            {'title': 'Redbone', 'artist': 'Childish Gambino', 'key': 'Eb', 'typical_low': 'Eb3', 'typical_high': 'Bb4', 'era': '2010s', 'genre': 'R&B', 'tags': 'smooth,mid-forward', 'difficulty': 'intermediate', 'mood': 'seductive'},
            {'title': 'Earned It', 'artist': 'The Weeknd', 'key': 'Cm', 'typical_low': 'C3', 'typical_high': 'Ab4', 'era': '2010s', 'genre': 'R&B', 'tags': 'sultry,bass-heavy', 'difficulty': 'intermediate', 'mood': 'seductive'},
            {'title': 'Shallow', 'artist': 'Lady Gaga', 'key': 'G', 'typical_low': 'G3', 'typical_high': 'D5', 'era': '2010s', 'genre': 'pop', 'tags': 'powerful,balanced', 'difficulty': 'advanced', 'mood': 'emotional'},
            
            # Classic Training Songs
            {'title': 'Amazing Grace', 'artist': 'Traditional', 'key': 'G', 'typical_low': 'G3', 'typical_high': 'D5', 'era': 'traditional', 'genre': 'hymn', 'tags': 'hymn,mid-forward', 'difficulty': 'beginner', 'mood': 'spiritual'},
            {'title': 'Ave Maria', 'artist': 'Schubert', 'key': 'Bb', 'typical_low': 'F3', 'typical_high': 'Ab5', 'era': 'classical', 'genre': 'classical', 'tags': 'sacred,treble-bright', 'difficulty': 'advanced', 'mood': 'reverent'},
            {'title': 'Hallelujah', 'artist': 'Leonard Cohen', 'key': 'C', 'typical_low': 'C3', 'typical_high': 'C5', 'era': 'vintage', 'genre': 'folk', 'tags': 'ballad,mid-forward', 'difficulty': 'intermediate', 'mood': 'contemplative'},
            {'title': 'Danny Boy', 'artist': 'Traditional', 'key': 'C', 'typical_low': 'C3', 'typical_high': 'D5', 'era': 'traditional', 'genre': 'folk', 'tags': 'folk,balanced', 'difficulty': 'intermediate', 'mood': 'melancholic'},
            {'title': 'O Holy Night', 'artist': 'Adolphe Adam', 'key': 'C', 'typical_low': 'C3', 'typical_high': 'C5', 'era': 'classical', 'genre': 'christmas', 'tags': 'powerful,balanced', 'difficulty': 'advanced', 'mood': 'reverent'},
            {'title': 'Happy Birthday', 'artist': 'Traditional', 'key': 'F', 'typical_low': 'F3', 'typical_high': 'F4', 'era': 'traditional', 'genre': 'celebration', 'tags': 'easy,balanced', 'difficulty': 'beginner', 'mood': 'celebratory'},
            {'title': 'Edelweiss', 'artist': 'R&H', 'key': 'Bb', 'typical_low': 'Bb3', 'typical_high': 'Eb4', 'era': 'vintage', 'genre': 'musical', 'tags': 'gentle,mid-forward', 'difficulty': 'beginner', 'mood': 'peaceful'},
            {'title': 'Scarborough Fair', 'artist': 'Traditional', 'key': 'Em', 'typical_low': 'E3', 'typical_high': 'E4', 'era': 'traditional', 'genre': 'folk', 'tags': 'haunting,mid-forward', 'difficulty': 'beginner', 'mood': 'mysterious'},
            
            # Additional Modern Hits
            {'title': 'Perfect', 'artist': 'Ed Sheeran', 'key': 'Ab', 'typical_low': 'Eb3', 'typical_high': 'Bb4', 'era': '2010s', 'genre': 'pop', 'tags': 'romantic,balanced', 'difficulty': 'intermediate', 'mood': 'romantic'},
            {'title': 'Roar', 'artist': 'Katy Perry', 'key': 'Bb', 'typical_low': 'Bb3', 'typical_high': 'F5', 'era': '2010s', 'genre': 'pop', 'tags': 'empowering,treble-bright', 'difficulty': 'intermediate', 'mood': 'empowering'},
            {'title': 'Firework', 'artist': 'Katy Perry', 'key': 'Ab', 'typical_low': 'Ab3', 'typical_high': 'Db5', 'era': '2010s', 'genre': 'pop', 'tags': 'inspirational,treble-bright', 'difficulty': 'intermediate', 'mood': 'uplifting'},
            {'title': 'Skyfall', 'artist': 'Adele', 'key': 'Cm', 'typical_low': 'C3', 'typical_high': 'Bb4', 'era': '2010s', 'genre': 'pop', 'tags': 'dramatic,bass-heavy', 'difficulty': 'advanced', 'mood': 'epic'},
            {'title': 'Radioactive', 'artist': 'Imagine Dragons', 'key': 'Bm', 'typical_low': 'B2', 'typical_high': 'F#4', 'era': '2010s', 'genre': 'rock', 'tags': 'powerful,bass-heavy', 'difficulty': 'intermediate', 'mood': 'intense'},
            {'title': 'Counting Stars', 'artist': 'OneRepublic', 'key': 'Am', 'typical_low': 'A2', 'typical_high': 'E4', 'era': '2010s', 'genre': 'pop', 'tags': 'upbeat,mid-forward', 'difficulty': 'beginner', 'mood': 'hopeful'},
            {'title': 'Wake Me Up', 'artist': 'Avicii', 'key': 'Bm', 'typical_low': 'B2', 'typical_high': 'F#4', 'era': '2010s', 'genre': 'EDM', 'tags': 'energetic,balanced', 'difficulty': 'beginner', 'mood': 'uplifting'},
            {'title': 'Chandelier', 'artist': 'Sia', 'key': 'F', 'typical_low': 'F3', 'typical_high': 'C6', 'era': '2010s', 'genre': 'pop', 'tags': 'powerful,treble-bright', 'difficulty': 'advanced', 'mood': 'desperate'},
            {'title': 'Titanium', 'artist': 'David Guetta ft. Sia', 'key': 'Eb', 'typical_low': 'Eb3', 'typical_high': 'Bb4', 'era': '2010s', 'genre': 'EDM', 'tags': 'empowering,treble-bright', 'difficulty': 'intermediate', 'mood': 'strong'},
        ]
        
        pd.DataFrame(songs).to_csv(CATALOG_FILE, index=False)
    return pd.read_csv(CATALOG_FILE)

def recommend_songs(analysis_results: Dict, catalog: pd.DataFrame, user_preferences: Dict = None) -> Dict:
    """Enhanced song recommendations with filtering."""
    if 'error' in analysis_results:
        return {'fit': [], 'stretch': [], 'avoid': []}
    
    user_tess_low = analysis_results['tessitura_low_midi']
    user_tess_high = analysis_results['tessitura_high_midi']
    user_timbre = analysis_results['timbre_classification'].lower()
    
    # Apply user filters
    filtered_catalog = catalog.copy()
    if user_preferences:
        if user_preferences.get('era'):
            filtered_catalog = filtered_catalog[filtered_catalog['era'] == user_preferences['era']]
        if user_preferences.get('genre'):
            filtered_catalog = filtered_catalog[filtered_catalog['genre'] == user_preferences['genre']]
        if user_preferences.get('mood'):
            filtered_catalog = filtered_catalog[filtered_catalog['mood'] == user_preferences['mood']]
        if user_preferences.get('difficulty'):
            filtered_catalog = filtered_catalog[filtered_catalog['difficulty'] == user_preferences['difficulty']]
    
    fit_songs, stretch_songs, avoid_songs = [], [], []
    
    for _, song in filtered_catalog.iterrows():
        try:
            song_low = hz_to_midi(librosa.note_to_hz(song['typical_low']))
            song_high = hz_to_midi(librosa.note_to_hz(song['typical_high']))
            
            rec = {
                'title': song['title'],
                'artist': song['artist'],
                'key': song['key'],
                'range': f"{song['typical_low']}-{song['typical_high']}",
                'difficulty': song['difficulty'],
                'era': song['era'],
                'genre': song['genre'],
                'mood': song['mood'],
                'reasons': []
            }
            
            if song_low >= user_tess_low - 2 and song_high <= user_tess_high + 2:
                rec['reasons'].append(f"✓ Perfect fit for your tessitura")
                if user_timbre in song['tags'].lower():
                    rec['reasons'].append(f"✓ Matches your {user_timbre} voice")
                fit_songs.append(rec)
            elif song_high <= user_tess_high + 5 and song_low >= user_tess_low - 5:
                if song_high > user_tess_high:
                    rec['reasons'].append(f"⬆️ Extends upper range by {int(song_high - user_tess_high)} semitones")
                if song_low < user_tess_low:
                    rec['reasons'].append(f"⬇️ Extends lower range by {int(user_tess_low - song_low)} semitones")
                stretch_songs.append(rec)
            else:
                if song_high > user_tess_high + 5:
                    rec['reasons'].append(f"❌ Too high ({midi_to_note_name(song_high)} needed)")
                if song_low < user_tess_low - 5:
                    rec['reasons'].append(f"❌ Too low ({midi_to_note_name(song_low)} needed)")
                avoid_songs.append(rec)
        except:
            continue
    
    return {'fit': fit_songs[:15], 'stretch': stretch_songs[:10], 'avoid': avoid_songs[:5]}

# Plotting functions
def plot_pitch_contour(analysis_results: Dict) -> go.Figure:
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
        x=voiced_times, y=voiced_pitch, mode='lines+markers',
        name='Your Pitch', line=dict(color='#3b82f6', width=2),
        marker=dict(size=3),
        hovertemplate='<b>Time:</b> %{x:.2f}s<br><b>Hz:</b> %{y:.1f}<br><b>Note:</b> %{text}<extra></extra>',
        text=[midi_to_note_name(m) for m in voiced_midi]
    ))
    
    tess_low_hz = midi_to_hz(analysis_results['tessitura_low_midi'])
    tess_high_hz = midi_to_hz(analysis_results['tessitura_high_midi'])
    
    fig.add_hrect(y0=tess_low_hz, y1=tess_high_hz, fillcolor='green', opacity=0.15, line_width=0,
                  annotation_text="Your Comfort Zone", annotation_position="top left")
    
    fig.update_layout(
        title='Your Pitch Performance Over Time',
        xaxis_title='Time (seconds)',
        yaxis_title='Frequency (Hz)',
        height=450,
        template='plotly_white',
        hovermode='closest'
    )
    
    return fig

def plot_note_distribution(analysis_results: Dict) -> go.Figure:
    note_dist = analysis_results['note_distribution']
    notes = list(note_dist.keys())
    counts = list(note_dist.values())
    
    fig = go.Figure([go.Bar(
        x=notes, y=counts,
        marker_color='#8b5cf6',
        text=counts,
        textposition='auto',
        hovertemplate='<b>%{x}</b><br>Count: %{y}<extra></extra>'
    )])
    
    fig.update_layout(
        title='Notes You Sang Most',
        xaxis_title='Note',
        yaxis_title='Frequency',
        height=350,
        template='plotly_white'
    )
    return fig

def plot_spectral_profile(analysis_results: Dict) -> go.Figure:
    features = analysis_results['spectral_features']
    bands = ['Bass<br>(Low)', 'Mid<br>(Middle)', 'Treble<br>(High)']
    energies = [
        features['low_energy_ratio'] * 100,
        features['mid_energy_ratio'] * 100,
        features['high_energy_ratio'] * 100
    ]
    colors = ['#ef4444', '#f59e0b', '#3b82f6']
    
    fig = go.Figure([go.Bar(
        x=bands, y=energies,
        marker_color=colors,
        text=[f'{e:.1f}%' for e in energies],
        textposition='auto',
        hovertemplate='<b>%{x}</b><br>Energy: %{y:.1f}%<extra></extra>'
    )])
    
    fig.update_layout(
        title='Your Voice Tone Profile',
        yaxis_title='Energy Percentage',
        height=350,
        template='plotly_white',
        showlegend=False
    )
    return fig

# Custom CSS
def apply_custom_css():
    st.markdown("""
    <style>
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem; border-radius: 12px; color: white;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1); margin: 0.5rem 0;
    }
    .info-card {
        background: #f8fafc; padding: 1.2rem; border-radius: 10px;
        border-left: 4px solid #3b82f6; margin: 0.8rem 0;
    }
    .success-card {
        background: linear-gradient(135deg, #84fab0 0%, #8fd3f4 100%);
        padding: 1.2rem; border-radius: 10px; color: #1a1a1a;
        font-weight: 600; margin: 0.8rem 0;
    }
    .warning-card {
        background: #fff3cd; padding: 1.2rem; border-radius: 10px;
        border-left: 4px solid #ffc107; margin: 0.8rem 0;
    }
    h1 {
        background: linear-gradient(120deg, #667eea, #764ba2);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    }
    .stButton>button {
        border-radius: 8px; font-weight: 600; transition: all 0.3s;
    }
    .stButton>button:hover {
        transform: translateY(-2px); box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }
    </style>
    """, unsafe_allow_html=True)

# Main App
def main():
    st.set_page_config(page_title="TuneScope Pro", page_icon="🎤", layout="wide")
    
    apply_custom_css()
    
    # Session state
    if 'analysis_results' not in st.session_state:
        st.session_state.analysis_results = None
    if 'audio_data' not in st.session_state:
        st.session_state.audio_data = None
    if 'catalog' not in st.session_state:
        st.session_state.catalog = initialize_expanded_catalog()
    if 'practice_log' not in st.session_state:
        st.session_state.practice_log = []
    
    # Header
    st.markdown("""
    <div style='text-align:center; padding:2rem 0;'>
        <h1 style='font-size:3.5rem; margin-bottom:0.5rem;'>🎤 TuneScope Pro</h1>
        <p style='font-size:1.3rem; color:#64748b;'>
            Your Personal Voice Coach • 100+ Songs • Singing Accuracy • Training Plans
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown("### ⚙️ Settings")
        
        st.markdown("**Voice Sensitivity**")
        voicing_threshold = st.slider(
            "Detection Threshold",
            0.0, 1.0, 0.20, 0.05,
            help="Lower = detects quieter singing",
            label_visibility="collapsed"
        )
        
        if voicing_threshold < 0.25:
            st.caption("🔊 Very sensitive - picks up soft singing")
        elif voicing_threshold > 0.5:
            st.caption("🔇 Less sensitive - only loud/clear voice")
        else:
            st.caption("✓ Balanced - recommended")
        
        st.markdown("---")
        
        st.markdown("### 🎯 Quick Stats")
        if st.session_state.analysis_results and 'error' not in st.session_state.analysis_results:
            results = st.session_state.analysis_results
            st.metric("Voice Type", results['voice_type']['type'])
            st.metric("Range", f"{int(results['max_note_midi'] - results['min_note_midi'])} semitones")
            st.metric("Accuracy", f"{results['singing_accuracy']['accuracy_score']:.0f}/100")
        else:
            st.info("Analyze your voice to see stats")
        
        st.markdown("---")
        
        st.markdown("### 📚 Practice Log")
        if len(st.session_state.practice_log) > 0:
            for entry in st.session_state.practice_log[-3:]:
                st.text(f"{entry['date']}: {entry['score']}/100")
        else:
            st.caption("No practice sessions yet")
        
        st.markdown("---")
        
        with st.expander("ℹ️ System Info"):
            st.text(f"Pitch Method: YIN (librosa)\nSample Rate: {SAMPLE_RATE}Hz\nCatalog: {len(st.session_state.catalog)} songs")
    
    # Main Tabs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "🎙️ Record & Analyze",
        "📊 Detailed Analysis",
        "🎵 Song Finder (100+)",
        "🏋️ Training Plans",
        "📈 Singing Coach",
        "🎓 Vocal Tips"
    ])
    
    # TAB 1: Record & Analyze
    with tab1:
        st.markdown("## 🎤 Record or Upload Your Voice")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 📤 Upload Audio File")
            uploaded_file = st.file_uploader(
                "Drop your singing/speaking audio here",
                type=['wav', 'mp3', 'm4a', 'flac', 'ogg'],
                help="Any audio format works!"
            )
            
            if uploaded_file:
                try:
                    with st.spinner("Loading your audio..."):
                        audio_bytes = uploaded_file.read()
                        y, sr = librosa.load(io.BytesIO(audio_bytes), sr=None, mono=True)
                        st.session_state.audio_data = (y, sr)
                        
                        st.success(f"✅ Loaded: **{uploaded_file.name}**")
                        
                        duration = len(y) / sr
                        st.markdown(f"""
                        <div class='info-card'>
                            <b>📊 Audio Info:</b><br>
                            ⏱️ Duration: {duration:.2f} seconds<br>
                            🎚️ Sample Rate: {sr:,} Hz<br>
                            📏 Quality: {'Excellent' if sr >= 44100 else 'Good' if sr >= 22050 else 'Basic'}
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Quick waveform
                        fig = go.Figure()
                        times = np.linspace(0, duration, min(len(y), 10000))
                        y_plot = y[:len(times)] if len(y) > len(times) else y
                        
                        fig.add_trace(go.Scatter(
                            x=times, y=y_plot,
                            mode='lines',
                            line=dict(color='#3b82f6', width=1),
                            fill='tozeroy',
                            name='Waveform'
                        ))
                        
                        fig.update_layout(
                            title='Your Audio Waveform',
                            xaxis_title='Time (s)',
                            yaxis_title='Amplitude',
                            height=200,
                            template='plotly_white',
                            showlegend=False
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                except Exception as e:
                    st.error(f"❌ Error loading audio: {str(e)}")
                    st.info("💡 Try converting to WAV or MP3 format")
        
        with col2:
            st.markdown("### 🎙️ Record Live")
            st.markdown("""
            <div class='info-card'>
                <b>📱 How to Record:</b><br><br>
                <b>Option 1: Use your phone/computer</b><br>
                • Open Voice Recorder app<br>
                • Sing or speak clearly<br>
                • Save and upload here<br><br>
                <b>Option 2: Online recorder</b><br>
                • Use online-voice-recorder.com<br>
                • Record directly in browser<br>
                • Download and upload<br><br>
                <b>💡 Recording Tips:</b><br>
                ✓ Quiet room (no background noise)<br>
                ✓ 15-30 seconds minimum<br>
                ✓ Sing a scale or favorite song<br>
                ✓ Stand 6-12 inches from mic<br>
                ✓ Sing at comfortable volume
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Analyze Button
        col_analyze, col_demo, col_clear = st.columns([2, 1, 1])
        
        with col_analyze:
            if st.button("🔬 ANALYZE MY VOICE", type="primary", use_container_width=True):
                if not st.session_state.audio_data:
                    st.error("❌ Please upload audio first!")
                else:
                    y, sr = st.session_state.audio_data
                    
                    progress_bar = st.progress(0)
                    status = st.empty()
                    
                    def update_progress(msg, val):
                        status.text(f"🔄 {msg}")
                        progress_bar.progress(val)
                    
                    try:
                        results = analyze_audio(y, sr, update_progress, voicing_threshold)
                        
                        if 'error' in results:
                            st.error(f"❌ {results['error']}")
                            st.markdown(f""""""
TuneScope Pro - Complete Voice Training & Analysis Platform
===========================================================
Professional vocal analysis, training, and song recommendations.

Features: Real-time pitch tracking, singing accuracy scoring,
personalized training plans, 100+ songs catalog, voice improvement tips.

Version: 2.0 | Author: Claude
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
import datetime
from typing import Dict, Tuple, List
import warnings
warnings.filterwarnings('ignore')

import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Constants
SAMPLE_RATE = 22050
HOP_LENGTH = 256
MIN_FREQUENCY = 65
MAX_FREQUENCY = 2093

VOICE_TYPES = {
    'Bass': {'min': 40, 'max': 64, 'tessitura': (45, 57), 'description': 'Deep, rich, powerful low voice'},
    'Baritone': {'min': 45, 'max': 69, 'tessitura': (50, 62), 'description': 'Warm mid-range male voice'},
    'Tenor': {'min': 48, 'max': 72, 'tessitura': (55, 67), 'description': 'Higher male voice, bright tone'},
    'Alto': {'min': 53, 'max': 77, 'tessitura': (60, 72), 'description': 'Lower female voice, rich tone'},
    'Mezzo-Soprano': {'min': 57, 'max': 81, 'tessitura': (64, 76), 'description': 'Mid-range female voice'},
    'Soprano': {'min': 60, 'max': 84, 'tessitura': (67, 79), 'description': 'High female voice, bright'}
}

VOCAL_EXERCISES = {
    'Warm-up': ['Lip trills', 'Humming scales', 'Sirens', 'Yawn-sighs', 'Gentle scales'],
    'Breath Control': ['Hissing exercises', 'Sustained notes', 'Staccato practice', 'Crescendo/Decrescendo'],
    'Range Extension': ['Ascending scales', 'Octave jumps', 'Arpeggio practice', 'Descending patterns'],
    'Pitch Accuracy': ['Interval training', 'Matching pitches', 'Scale practice', 'Slow melodies'],
    'Tone Quality': ['Vowel modifications', 'Resonance exercises', 'Forward placement', 'Open throat technique']
}

PRACTICE_ROUTINES = {
    'Beginner': {
        'duration': '15-20 min',
        'routine': [
            '5 min: Breathing exercises',
            '5 min: Gentle warm-ups',
            '5 min: Scale practice (major)',
            '5 min: Simple song practice'
        ]
    },
    'Intermediate': {
        'duration': '30-40 min',
        'routine': [
            '5 min: Deep breathing',
            '10 min: Full warm-up routine',
            '10 min: Scale & interval work',
            '10 min: Song practice (2-3 songs)',
            '5 min: Cool down'
        ]
    },
    'Advanced': {
        'duration': '45-60 min',
        'routine': [
            '5 min: Breathing exercises',
            '10 min: Comprehensive warm-up',
            '15 min: Technical exercises',
            '15 min: Repertoire practice',
            '10 min: Performance simulation',
            '5 min: Cool down & reflection'
        ]
    }
}

CATALOG_FILE = "song_catalog.csv"

# Utility Functions
def hz_to_midi(hz: float) -> float:
    if hz <= 0: return 0
    return 12 * np.log2(hz / 440.0) + 69

def midi_to_hz(midi: float) -> float:
    return 440.0 * (2.0 ** ((midi - 69) / 12.0))

def midi_to_note_name(midi: float) -> str:
    if midi <= 0: return "N/A"
    note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    note_num = int(round(midi))
    octave = (note_num // 12) - 1
    note = note_names[note_num % 12]
    return f"{note}{octave}"

def cents_from_midi(hz: float, ref_midi: float) -> float:
    if hz <= 0: return 0
    return (hz_to_midi(hz) - ref_midi) * 100

def smooth_pitch(pitch: np.ndarray, confidence: np.ndarray, window_size: int = 5, conf_threshold: float = 0.25) -> np.ndarray:
    pitch_masked = pitch.copy()
    pitch_masked[confidence < conf_threshold] = 0
    smoothed = signal.medfilt(pitch_masked, kernel_size=window_size)
    return smoothed

def compute_spectral_features(y: np.ndarray, sr: int) -> Dict:
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
    low = spectral_features['low_energy_ratio']
    mid = spectral_features['mid_energy_ratio']
    high = spectral_features['high_energy_ratio']
    
    if low > 0.4: return "Bass-heavy"
    elif high > 0.35: return "Treble-bright"
    elif mid > 0.5: return "Mid-forward"
    else: return "Balanced"

def detect_pitch_yin(y: np.ndarray, sr: int) -> Tuple[np.ndarray, np.ndarray]:
    pitch = librosa.yin(y, fmin=MIN_FREQUENCY, fmax=MAX_FREQUENCY, sr=sr, hop_length=HOP_LENGTH)
    rms = librosa.feature.rms(y=y, hop_length=HOP_LENGTH)[0]
    confidence = np.clip(rms / (np.max(rms) + 1e-6), 0, 1)
    min_len = min(len(pitch), len(confidence))
    return pitch[:min_len], confidence[:min_len]

def classify_voice_type(midi_notes: np.ndarray, tessitura_range: Tuple[float, float]) -> Dict:
    min_note = float(np.min(midi_notes))
    max_note = float(np.max(midi_notes))
    tess_low, tess_high = float(tessitura_range[0]), float(tessitura_range[1])
    
    scores = {}
    for voice_type, ranges in VOICE_TYPES.items():
        range_overlap = (min(max_note, ranges['max']) - max(min_note, ranges['min'])) / (ranges['max'] - ranges['min'])
        tess_overlap = (min(tess_high, ranges['tessitura'][1]) - max(tess_low, ranges['tessitura'][0])) / (ranges['tessitura'][1] - ranges['tessitura'][0])
        scores[voice_type] = float(max(0, range_overlap * 0.4 + tess_overlap * 0.6))
    
    sorted_types = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    best_type, best_score = sorted_types[0]
    is_borderline = len(sorted_types) > 1 and sorted_types[1][1] > best_score * 0.85
    
    return {
        'type': best_type,
        'confidence': float(best_score),
        'is_borderline': bool(is_borderline),
        'alternative': sorted_types[1][0] if is_borderline else None,
        'all_scores': {k: float(v) for k, v in sorted_types},
        'description': VOICE_TYPES[best_type]['description']
    }

def analyze_singing_accuracy(y: np.ndarray, sr: int, reference_notes: List[float] = None) -> Dict:
    """Analyze how accurately user is singing."""
    pitch, confidence = detect_pitch_yin(y, sr)
    pitch_smoothed = smooth_pitch(pitch, confidence, conf_threshold=0.2)
    voiced_mask = (pitch_smoothed > 0) & (confidence > 0.2)
    voiced_pitch = pitch_smoothed[voiced_mask]
    
    if len(voiced_pitch) < 5:
        return {'accuracy_score': 0, 'message': 'Too short - sing longer'}
    
    midi_notes = np.array([hz_to_midi(f) for f in voiced_pitch if f > 0])
    cents_errors = [abs(cents_from_midi(hz, round(hz_to_midi(hz)))) for hz in voiced_pitch if hz > 0]
    
    # Accuracy metrics
    mean_error = np.mean(cents_errors)
    consistency = 100 - np.std(cents_errors)
    
    # Vibrato detection
    pitch_variation = np.std(np.diff(midi_notes))
    has_vibrato = pitch_variation > 0.3 and pitch_variation < 1.5
    
    # Note stability
    stable_notes = sum(1 for e in cents_errors if e < 25) / len(cents_errors) * 100
    
    accuracy_score = max(0, 100 - mean_error)
    
    feedback = []
    if accuracy_score >= 90:
        feedback.append("🌟 Excellent pitch accuracy!")
    elif accuracy_score >= 75:
        feedback.append("👍 Good intonation!")
    elif accuracy_score >= 60:
        feedback.append("⚠️ Practice pitch matching")
    else:
        feedback.append("📚 Focus on ear training")
    
    if stable_notes < 50:
        feedback.append("💡 Work on holding steady notes")
    
    if has_vibrato:
        feedback.append("🎵 Nice vibrato detected!")
    
    return {
        'accuracy_score': float(accuracy_score),
        'mean_cents_error': float(mean_error),
        'consistency': float(max(0, consistency)),
        'stable_notes_percent': float(stable_notes),
        'has_vibrato': bool(has_vibrato),
        'feedback': feedback
    }

def analyze_audio(y: np.ndarray, sr: int, progress_callback=None, voicing_threshold: float = 0.25) -> Dict:
    results = {}
    
    if progress_callback:
        progress_callback("Preprocessing...", 0.1)
    
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
    
    pitch, confidence = detect_pitch_yin(y, sr)
    pitch_smoothed = smooth_pitch(pitch, confidence, conf_threshold=voicing_threshold)
    voiced_mask = (pitch_smoothed > 0) & (confidence > voicing_threshold)
    voiced_pitch = pitch_smoothed[voiced_mask]
    
    if len(voiced_pitch) < 10:
        return {
            'error': f'Insufficient voice detected. Speak/sing louder and longer. Try threshold: {voicing_threshold-0.05:.2f}',
            'voiced_ratio': float(np.sum(voiced_mask) / len(voiced_mask) if len(voiced_mask) > 0 else 0)
        }
    
    results['voiced_ratio'] = float(np.sum(voiced_mask) / len(voiced_mask))
    
    if progress_callback:
        progress_callback("Analyzing range...", 0.5)
    
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
        progress_callback("Voice classification...", 0.6)
    
    results['voice_type'] = classify_voice_type(midi_notes, (tess_low, tess_high))
    
    if progress_callback:
        progress_callback("Accuracy check...", 0.7)
    
    cents_errors = [cents_from_midi(hz, round(hz_to_midi(hz))) for hz in voiced_pitch if hz > 0]
    
    results['pitch_accuracy'] = {
        'mean_cents_error': float(np.mean(np.abs(cents_errors))),
        'median_cents_error': float(np.median(cents_errors)),
        'std_cents_error': float(np.std(cents_errors)),
        'intonation_score': float(max(0, 100 - np.mean(np.abs(cents_errors))))
    }
    
    # Singing accuracy
    results['singing_accuracy'] = analyze_singing_accuracy(y, sr)
    
    if progress_callback:
        progress_callback("Timbre analysis...", 0.8)
    
    spectral_features = compute_spectral_features(y, sr)
    results['spectral_features'] = spectral_features
    results['timbre_classification'] = classify_timbre(spectral_features)
    
    # Note distribution
    note_histogram = {}
    for midi in midi_notes[::10]:
        note_name = midi_to_note_name(midi)
        note_histogram[note_name] = note_histogram.get(note_name, 0) + 1
    results['note_distribution'] = note_histogram
    
    # Downsampled contour
    downsample = max(1, len(pitch_smoothed) // 500)
    results['pitch_contour'] = {
        'times': [float(x) for x in (np.arange(len(pitch_smoothed))[::downsample] * HOP_LENGTH / sr).tolist()],
        'pitch_hz': [float(x) for x in pitch_smoothed[::downsample].tolist()],
        'confidence': [float(x) for x in confidence[::downsample].tolist()],
        'voiced_mask': [bool(x) for x in voiced_mask[::downsample].tolist()]
    }
    
    if progress_callback:
        progress_callback("Complete!", 1.0)
    
    return results

def initialize_expanded_catalog():
    """Create comprehensive 100+ song catalog."""
    if not os.path.exists(CATALOG_FILE):
        songs = [
            # Classic/Vintage (1920s-1970s)
            {'title': 'Fly Me to the Moon', 'artist': 'Frank Sinatra', 'key': 'C', 'typical_low': 'C3', 'typical_high': 'D4', 'era': 'vintage', 'genre': 'jazz', 'tags': 'swing,romantic,mid-forward', 'difficulty': 'intermediate', 'mood': 'romantic'},
            {'title': 'My Way', 'artist': 'Frank Sinatra', 'key': 'D', 'typical_low': 'A2', 'typical_high': 'D4', 'era': 'vintage', 'genre': 'pop', 'tags': 'powerful,bass-heavy', 'difficulty': 'intermediate', 'mood': 'confident'},
            {'title': 'Unchained Melody', 'artist': 'Righteous Brothers', 'key': 'C', 'typical_low': 'C3', 'typical_high': 'F4', 'era': 'vintage', 'genre': 'ballad', 'tags': 'emotional,mid-forward', 'difficulty': 'intermediate', 'mood': 'romantic'},
            {'title': 'Stand By Me', 'artist': 'Ben E. King', 'key': 'A', 'typical_low': 'E3', 'typical_high': 'C#4', 'era': 'vintage', 'genre': 'soul', 'tags': 'smooth,bass-heavy', 'difficulty': 'beginner', 'mood': 'comforting'},
            {'title': 'What a Wonderful World', 'artist': 'Louis Armstrong', 'key': 'F', 'typical_low': 'F3', 'typical_high': 'F4', 'era': 'vintage', 'genre': 'jazz', 'tags': 'warm,bass-heavy', 'difficulty': 'beginner', 'mood': 'uplifting'},
            {'title': 'Can\'t Help Falling in Love', 'artist': 'Elvis Presley', 'key': 'C', 'typical_low': 'C3', 'typical_high': 'C4', 'era': 'vintage', 'genre': 'ballad', 'tags': 'gentle,balanced', 'difficulty': 'beginner', 'mood': 'romantic'},
            {'title': 'At Last', 'artist': 'Etta James', 'key': 'F', 'typical_low': 'F3', 'typical_high': 'Ab4', 'era': 'vintage', 'genre': 'blues', 'tags': 'soulful,mid-forward', 'difficulty': 'intermediate', 'mood': 'romantic'},
            {'title': 'Somewhere Over Rainbow', 'artist': 'Judy Garland', 'key': 'Eb', 'typical_low': 'Eb3', 'typical_high': 'Bb4', 'era': 'vintage', 'genre': 'musical', 'tags': 'dreamy,balanced', 'difficulty': 'intermediate', 'mood': 'hopeful'},
            {'title': 'Feeling Good', 'artist': 'Nina Simone', 'key': 'Gm', 'typical_low': 'G3', 'typical_high': 'Bb4', 'era': 'vintage', 'genre': 'jazz', 'tags': 'powerful,mid-forward', 'difficulty': 'intermediate', 'mood': 'empowering'},
            {'title': 'Dream a Little Dream', 'artist': 'Ella Fitzgerald', 'key': 'C', 'typical_low': 'C4', 'typical_high': 'E5', 'era': 'vintage', 'genre': 'jazz', 'tags': 'sweet,treble-bright', 'difficulty': 'intermediate', 'mood': 'dreamy'},
            
            # 1980s-1990s Classics
            {'title': 'I Will Always Love You', 'artist': 'Whitney Houston', 'key': 'A', 'typical_low': 'C#3', 'typical_high': 'B4', 'era': '80s-90s', 'genre': 'pop', 'tags': 'powerful,treble-bright', 'difficulty': 'advanced', 'mood': 'emotional'},
            {'title': 'Careless Whisper', 'artist': 'George Michael', 'key': 'Dm', 'typical_low': 'D3', 'typical_high': 'F4', 'era': '80s-90s', 'genre': 'pop', 'tags': 'smooth,balanced', 'difficulty': 'intermediate', 'mood': 'romantic'},
            {'title': 'Every Breath You Take', 'artist': 'The Police', 'key': 'Ab', 'typical_low': 'Eb3', 'typical_high': 'Bb4', 'era': '80s-90s', 'genre': 'rock', 'tags': 'melodic,mid-forward', 'difficulty': 'beginner', 'mood': 'melancholic'},
            {'title': 'Sweet Child O\' Mine', 'artist': 'Guns N\' Roses', 'key': 'Db', 'typical_low': 'Ab2', 'typical_high': 'Db5', 'era': '80s-90s', 'genre': 'rock', 'tags': 'energetic,treble-bright', 'difficulty': 'advanced', 'mood': 'energetic'},
            {'title': 'Don\'t Stop Believin\'', 'artist': 'Journey', 'key': 'E', 'typical_low': 'E3', 'typical_high': 'E4', 'era': '80s-90s', 'genre': 'rock', 'tags': 'anthemic,balanced', 'difficulty': 'intermediate', 'mood': 'uplifting'},
            {'title': 'Total Eclipse of Heart', 'artist': 'Bonnie Tyler', 'key': 'Ab', 'typical_low': 'Eb3', 'typical_high': 'Ab4', 'era': '80s-90s', 'genre': 'rock', 'tags': 'dramatic,mid-forward', 'difficulty': 'intermediate', 'mood': 'passionate'},
            {'title': 'Take On Me', 'artist': 'a-ha', 'key': 'A', 'typical_low': 'A3', 'typical_high': 'E5', 'era': '80s-90s', 'genre': 'pop', 'tags': 'upbeat,treble-bright', 'difficulty': 'advanced', 'mood': 'energetic'},
            {'title': 'Livin\' on a Prayer', 'artist': 'Bon Jovi', 'key': 'Em', 'typical_low': 'E3', 'typical_high': 'B4', 'era': '80s-90s', 'genre': 'rock', 'tags': 'powerful,mid-forward', 'difficulty': 'intermediate', 'mood': 'motivational'},
            {'title': 'Nothing Compares 2 U', 'artist': 'Sinead O\'Connor', 'key': 'F', 'typical_low': 'F3', 'typical_high': 'Bb4', 'era': '80s-90s', 'genre': 'pop', 'tags': 'emotional,balanced', 'difficulty': 'intermediate', 'mood': 'sad'},
            {'title': 'With or Without You', 'artist': 'U2', 'key': 'D', 'typical_low': 'D3', 'typical_high': 'D4', 'era': '80s-90s', 'genre': 'rock', 'tags': 'atmospheric,mid-forward', 'difficulty': 'beginner', 'mood': 'reflective'},
            
            # 2000s Pop/R&B
            {'title': 'A Thousand Miles', 'artist': 'Vanessa Carlton', 'key': 'B', 'typical_low': 'F#3', 'typical_high': 'C#5', 'era': '2000s', 'genre': 'pop', 'tags': 'melodic,treble-bright', 'difficulty': 'intermediate', 'mood': 'nostalgic'},
            {'title': 'Crazy', 'artist': 'Gnarls Barkley', 'key': 'Cm', 'typical_low': 'C3', 'typical_high': 'Eb4', 'era': '2000s', 'genre': 'soul', 'tags': 'groovy,mid-forward', 'difficulty': 'intermediate', 'mood': 'cool'},
            {'title': 'Rehab', 'artist': 'Amy Winehouse', 'key': 'F', 'typical_low': 'F3', 'typical_high': 'C5', 'era': '2000s', 'genre': 'soul', 'tags': 'jazzy,balanced', 'difficulty': 'intermediate', 'mood': 'defiant'},
            {'title': 'Chasing Cars', 'artist': 'Snow Patrol', 'key': 'A', 'typical_low': 'E3', 'typical_high': 'C#4', 'era': '2000s', 'genre': 'rock', 'tags': 'gentle,mid-forward', 'difficulty': 'beginner', 'mood': 'peaceful'},
            {'title': 'Bleeding Love', 'artist': 'Leona Lewis', 'key': 'Fm', 'typical_low': 'F3', 'typical_high': 'C5', 'era': '2000s', 'genre': 'pop', 'tags': 'powerful,treble-bright', 'difficulty': 'advanced', 'mood': 'emotional'},
            {'title': 'Apologize', 'artist': 'OneRepublic', 'key': 'Cm', 'typical_low': 'C3', 'typical_high': 'G4', 'era': '2000s', 'genre': 'pop', 'tags': 'melodic,mid-forward', 'difficulty': 'intermediate', 'mood': 'regretful'},
            {'title': 'Beautiful', 'artist': 'Christina Aguilera', 'key': 'Eb', 'typical_low': 'Eb3', 'typical_high': 'Bb4', 'era': '2000s', 'genre': 'pop', 'tags': 'empowering,balanced', 'difficulty': 'intermediate', 'mood': 'uplifting'},
            {'title': 'Just Dance', 'artist': 'Lady Gaga', 'key': 'C#m', 'typical_low': 'C#3', 'typical_high': 'G#4', 'era': '2000s', 'genre': 'pop', 'tags': 'energetic,treble-bright', 'difficulty': 'intermediate', 'mood': 'party'},
            {'title': 'Viva La Vida', 'artist': 'Coldplay', 'key': 'Ab', 'typical_low': 'Eb3', 'typical_high': 'Bb4', 'era': '2000s', 'genre': 'rock', 'tags': 'anthemic,balanced', 'difficulty': 'intermediate', 'mood': 'epic'},
            {'title': 'Poker Face', 'artist': 'Lady Gaga', 'key': 'G#m', 'typical_low': 'G#3', 'typical_high': 'D#4', 'era': '2000s', 'genre': 'pop', 'tags': 'catchy,mid-forward', 'difficulty': 'beginner', 'mood': 'mysterious'},
            
            # 2010s Modern Pop
            {'title': 'Rolling in the Deep', 'artist': 'Adele', 'key': 'Cm', 'typical_low': 'C3', 'typical_high': 'F4', 'era': '2010s', 'genre': 'pop', 'tags': 'powerful,bass-heavy', 'difficulty': 'intermediate', 'mood': 'angry'},
            {'title': 'Someone Like You', 'artist': 'Adele', 'key': 'A', 'typical_low': 'E3', 'typical_high': 'C#5', 'era': '2010s', 'genre': 'pop', 'tags': 'emotional,balanced', 'difficulty': 'intermediate', 'mood': 'heartbroken'},
            {'title': 'Shape of You', 'artist': 'Ed Sheeran', 'key': 'C#m', 'typical_low': 'C#3', 'typical_high': 'F#4', 'era': '2010s', 'genre': 'pop', 'tags': 'rhythmic,mid-forward', 'difficulty': 'beginner', 'mood': 'flirty'},
            {'title': 'Thinking Out Loud', 'artist': 'Ed Sheeran', 'key': 'D', 'typical_low': 'A2', 'typical_high': 'A4', 'era': '2010s', 'genre': 'pop', 'tags': 'romantic,balanced', 'difficulty': 'intermediate', 'mood': 'romantic'},
            {'title': 'Stay With Me', 'artist': 'Sam Smith', 'key': 'C', 'typical_low': 'C3', 'typical_high': 'G4', 'era': '2010s', 'genre': 'pop', 'tags': 'soulful,mid-forward', 'difficulty': 'intermediate', 'mood': 'pleading'},
            {'title': 'Happy', 'artist': 'Pharrell Williams', 'key': 'F', 'typical

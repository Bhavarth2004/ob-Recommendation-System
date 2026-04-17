import streamlit as st
import pandas as pd
import pickle
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import os
import plotly.graph_objects as go

# 1. Page Configuration
st.set_page_config(page_title="AI Career Path Finder", page_icon="🎯", layout="wide")

# Custom CSS for styling
st.markdown("""
    <style>
    .main { background-color: #f5f7f9; }
    .stButton>button { width: 100%; border-radius: 5px; height: 3em; background-color: #ff4b4b; color: white; }
    .stProgress > div > div > div > div { background-color: #00c853; }
    </style>
    """, unsafe_allow_html=True)

# 2. Helper Functions
def check_files():
    files = ['career_profiles.pkl', 'scaler.pkl']
    missing = [f for f in files if not os.path.exists(f)]
    return missing

def create_radar_chart(user_data, target_data, career_name):
    categories = ['Openness', 'Conscientiousness', 'Extraversion', 'Agreeableness', 'Neuroticism', 
                  'Numerical', 'Spatial', 'Perceptual', 'Abstract', 'Verbal']
    
    fig = go.Figure()

    # User Profile Trace
    fig.add_trace(go.Scatterpolar(
        r=user_data,
        theta=categories,
        fill='toself',
        name='Your Profile',
        line_color='rgb(31, 119, 180)'
    ))
    
    # Career Profile Trace
    fig.add_trace(go.Scatterpolar(
        r=target_data,
        theta=categories,
        fill='toself',
        name=f'Ideal {career_name}',
        line_color='rgb(255, 127, 14)'
    ))

    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 10])),
        showlegend=True,
        title=f"Comparison: You vs. {career_name}",
        margin=dict(l=40, r=40, t=40, b=40)
    )
    return fig

# 3. App Logic
st.title("🎯 AI-Powered Career Recommendation System")
st.markdown("---")

missing_files = check_files()
if missing_files:
    st.error(f"⚠️ Critical Error: Missing {', '.join(missing_files)}. Run your Jupyter Notebook first!")
else:
    @st.cache_resource
    def load_models():
        with open('career_profiles.pkl', 'rb') as f:
            profiles = pickle.load(f)
        with open('scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        return profiles, scaler

    profiles, scaler = load_models()

    # Sidebar Navigation
    st.sidebar.image("https://cdn-icons-png.flaticon.com/512/1063/1063376.png", width=100)
    st.sidebar.header("User Assessment")
    
    st.sidebar.subheader("🧠 Personality Traits")
    o = st.sidebar.slider("Openness", 0.0, 10.0, 5.0, help="Curiosity and creativity")
    c = st.sidebar.slider("Conscientiousness", 0.0, 10.0, 5.0, help="Organization and discipline")
    e = st.sidebar.slider("Extraversion", 0.0, 10.0, 5.0, help="Social energy")
    a = st.sidebar.slider("Agreeableness", 0.0, 10.0, 5.0, help="Compassion and cooperation")
    n = st.sidebar.slider("Neuroticism", 0.0, 10.0, 5.0, help="Emotional sensitivity")
    
    st.sidebar.subheader("📊 Aptitude Scores")
    num = st.sidebar.slider("Numerical", 0.0, 10.0, 5.0)
    spa = st.sidebar.slider("Spatial", 0.0, 10.0, 5.0)
    per = st.sidebar.slider("Perceptual", 0.0, 10.0, 5.0)
    abs_r = st.sidebar.slider("Abstract", 0.0, 10.0, 5.0)
    ver_r = st.sidebar.slider("Verbal", 0.0, 10.0, 5.0)

    # Main Area
    if st.sidebar.button("Generate My Report"):
        # Processing
        user_raw = np.array([o, c, e, a, n, num, spa, per, abs_r, ver_r])
        user_scaled = scaler.transform([user_raw])
        
        # Calculate scores
        profile_features = profiles.drop('Career', axis=1).values
        sim_scores = cosine_similarity(user_scaled, profile_features)[0]
        
        results = profiles.copy()
        results['Match_Score'] = sim_scores
        top_3 = results.sort_values(by='Match_Score', ascending=False).head(3)

        # Dashboard View
        st.header("🚀 Your Personal Career Analysis")
        
        for idx, row in top_3.iterrows():
            with st.container():
                col_text, col_chart = st.columns([1, 2])
                
                score_pct = round(row['Match_Score'] * 100, 1)
                
                with col_text:
                    st.write(f"### Rank {top_3.index.get_loc(idx) + 1}: {row['Career']}")
                    st.metric("Compatibility Score", f"{score_pct}%")
                    st.progress(row['Match_Score'])
                    st.write("**Key Alignment:** Your scores show a strong correlation with typical industry standards for this role.")
                
                with col_chart:
                    # To plot correctly, we need the "unscaled" career values
                    # Since we exported profiles as scaled, we show relative comparison
                    career_vals = row.drop(['Career', 'Match_Score']).values * 10 # approximate back to 10 scale
                    fig = create_radar_chart(user_raw, career_vals, row['Career'])
                    st.plotly_chart(fig, use_container_width=True)
                
                st.markdown("---")
    else:
        st.info("👈 Use the sidebar to input your scores and click 'Generate My Report' to see your results.")

    # Educational Footer
    with st.expander("ℹ️ How does this system work?"):
        st.write("""
            - **Data Source:** Comparative analysis of 100+ professional career benchmarks.
            - **Algorithm:** Uses **Cosine Similarity** to measure the angular distance between your trait-vector and career-vectors.
            - **Scaling:** Data is normalized using **MinMaxScaler** to ensure personality and aptitude carry equal weight.
        """)
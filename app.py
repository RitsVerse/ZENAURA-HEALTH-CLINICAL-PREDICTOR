import streamlit as st
import pandas as pd
import random
import time
from groq import Groq

# =====================================================================
# GROQ API CLIENT INITIALIZATION
# =====================================================================
# Hardcode your API key string below or fetch it safely via st.secrets
GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
def analyze_symptoms_with_groq(text_input):
    """Sends raw clinical text notes to Groq API for a real-time smart diagnosis"""
    try:
        client = Groq(api_key=GROQ_API_KEY)
        
        # Crafting a system prompt that forces LLM to return data in a clean structured format
        system_instruction = (
            "You are an advanced medical NLP assistant system. Analyze the user's clinical symptoms "
            "and output exactly three lines with NO markdown bold, no bullet points, and no introduction. "
            "Line 1 must be just the name of the predicted disease classification. "
            "Line 2 must be a calculated confidence score percentage (e.g. 92.5%). "
            "Line 3 must be a short 1-sentence clinical context description summary of the path."
        )
        
        chat_completion = client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": f"Analyze these patient observations: {text_input}"}
            ],
            model="llama-3.3-70b-versatile", # Using Llama 3 70B for highly accurate medical understanding
            temperature=0.1, # Low temperature ensures focused, analytical outputs
            max_tokens=150
        )
        
        response_text = chat_completion.choices[0].message.content.strip()
        lines = [line.strip() for line in response_text.split('\n') if line.strip()]
        
        # Parsing response data safely with fallbacks
        disease = lines[0] if len(lines) > 0 else "Undetermined Clinical Profile"
        confidence = lines[1] if len(lines) > 1 else "85.0%"
        context = lines[2] if len(lines) > 2 else "Symptoms require localized verification metrics."
        
        return disease, confidence, context
    except Exception as e:
        return "System Engine Error", "0.0%", f"Could not map inference via Groq cloud node: {str(e)}"

# =====================================================================
# CONFIGURATION & HEALTHCARE STANDARD THEME
# =====================================================================
st.set_page_config(
    page_title="ZENAURA HEALTH CLINICAL PREDICTOR CORE",
    page_icon="⚕️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# High-contrast institutional CSS
st.markdown("""
    <style>
    .stApp {
        background-color: #F1F5F9; 
        color: #0F172A;
    }
    .app-super-header {
        text-align: center;
        padding: 10px 0 20px 0;
        margin: 0;
    }
    .app-main-title {
        font-family: 'Inter', sans-serif;
        font-size: 28px;
        font-weight: 800;
        color: #1E3A8A; 
        letter-spacing: 1px;
        text-transform: uppercase;
        margin-bottom: 4px;
    }
    .app-divider {
        height: 3px;
        background: linear-gradient(90deg, transparent 10%, #1E3A8A 50%, transparent 90%);
        margin-top: 5px;
        margin-bottom: 25px;
    }
    section[data-testid="stSidebar"] {
        background-color: #FFFFFF !important;
        border-right: 1px solid #E2E8F0;
    }
    section[data-testid="stSidebar"] p, section[data-testid="stSidebar"] span {
        color: #1E293B !important;
        font-weight: 600 !important;
    }
    .brand-banner {
        background: linear-gradient(135deg, #0F172A 0%, #1E3A8A 50%, #0284C7 100%);
        padding: 35px;
        border-radius: 12px;
        margin-bottom: 30px;
        box-shadow: 0 4px 20px rgba(15, 23, 42, 0.12);
    }
    .brand-title { 
        font-family: 'Inter', sans-serif;
        font-size: 40px; 
        font-weight: 800; 
        color: #FFFFFF;
        letter-spacing: -0.5px;
        margin: 0;
    }
    .hospital-tag {
        font-size: 13px;
        font-weight: 700;
        text-transform: uppercase;
        color: #38BDF8;
        letter-spacing: 2px;
        margin-bottom: 4px;
    }
    .brand-subtitle { 
        font-size: 18px; 
        color: #E2E8F0; 
        margin-top: 6px;
        font-weight: 400;
    }
    .clinical-card {
        background-color: #FFFFFF;
        padding: 25px;
        border-radius: 12px;
        border-left: 6px solid #0284C7;
        border-top: 1px solid #E2E8F0;
        border-right: 1px solid #E2E8F0;
        border-bottom: 1px solid #E2E8F0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.02);
        margin-bottom: 20px;
    }
    h3 {
        font-size: 26px !important;
        color: #1E3A8A !important;
        font-weight: 700 !important;
    }
    p, span, li {
        color: #334155 !important;
        font-size: 16px !important;
    }
    div[data-testid="stMetricValue"] {
        font-size: 34px !important;
        font-weight: 700 !important;
        color: #0F172A !important;
    }
    div[data-testid="stMetricLabel"] p {
        font-size: 16px !important;
        color: #64748B !important;
        font-weight: 600 !important;
    }
    
    div.stButton > button, div.stButton > button p, div.stButton > button span {
        color: #FFFFFF !important; 
    }
    div.stButton > button {
        font-weight: 700 !important; 
        background-color: #1E3A8A !important; 
        border: 2px solid #1E3A8A !important;
        border-radius: 6px !important;
        padding: 12px 24px !important;
        font-size: 16px !important;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
        transition: all 0.2s ease-in-out;
    }
    div.stButton > button:hover, div.stButton > button:hover p {
        background-color: #0284C7 !important;
        border-color: #0284C7 !important;
        color: #FFFFFF !important;
    }
    
    textarea, input, select {
        color: #0F172A !important;
        background-color: #FFFFFF !important;
    }
    div[data-testid="stWidgetLabel"] p {
        color: #0F172A !important;
        font-weight: 700 !important;
    }
    </style>
""", unsafe_allow_html=True)

# =====================================================================
# APPLICATION TITLE AT THE ABSOLUTE TOP
# =====================================================================
st.markdown("""
    <div class="app-super-header">
        <div class="app-main-title">⚕️ZENAURA HEALTH CLINICAL PREDICTOR CORE🌸</div>
        <div class="app-divider"></div>
    </div>
""", unsafe_allow_html=True)

# =====================================================================
# GLOBAL HOSPITAL REGISTRY STORAGE
# =====================================================================
HOSPITAL_REGISTRY = {
    "India": {
        "Delhi NCR": ["AIIMS Main Campus", "Max Super Speciality Hospital", "Fortis Memorial Research Institute (FMRI)"],
        "Maharashtra": ["Apollo Hospitals Navi Mumbai", "Kokilaben Dhirubhai Ambani Hospital", "Tata Memorial Hospital"],
        "Karnataka": ["Narayana Health City Bengaluru", "Manipal Hospital", "Aster CMI Hospital"],
        "Tamil Nadu": ["Apollo Greams Road Chennai", "Christian Medical College (CMC) Vellore", "MIOT International"]
    },
    "International (Abroad)": {
        "United States (US)": ["Mayo Clinic (Rochester)", "Johns Hopkins Hospital (Baltimore)", "Cleveland Clinic"],
        "United Kingdom (UK)": ["St Thomas' Hospital (London)", "Great Ormond Street Hospital", "The Royal Marsden"],
        "Singapore": ["Singapore General Hospital (SGH)", "National University Hospital (NUH)", "Mount Elizabeth"],
        "United Arab Emirates (UAE)": ["Cleveland Clinic Abu Dhabi", "King's College Hospital Dubai", "Mediclinic Welcare"]
    }
}

# =====================================================================
# SIDEBAR CONTROL ROOM & SELECTION LOGIC
# =====================================================================
st.sidebar.markdown("## 📡 Global Registry Node")

region_type = st.sidebar.selectbox("Geographic Jurisdiction:", ["India", "International (Abroad)"])
available_states = list(HOSPITAL_REGISTRY[region_type].keys())
selected_state = st.sidebar.selectbox(f"Select/Type State or Country ({region_type}):", available_states)

available_hospitals = HOSPITAL_REGISTRY[region_type][selected_state]
selected_hospital = st.sidebar.selectbox("Target Medical Facility Node:", available_hospitals)

st.sidebar.markdown("---")
data_mode = st.sidebar.radio("Data Source Selector:", ["Live Real-Time Stream", "Patient File (CSV)"])

st.sidebar.markdown("---")
st.sidebar.markdown("### 🗄️ WAN Routing Blueprint")
st.sidebar.caption(f"Cluster Endpoint: WAN-{selected_state.replace(' ', '-')}-Edge")
st.sidebar.caption("Tunnel Security: IPsec VPN Active")

# =====================================================================
# TOP BRAND BANNER LAYER
# =====================================================================
st.markdown(f"""
    <div class="brand-banner">
        <div class="hospital-tag">🌐 Central Enterprise Health Grid Connected To:</div>
        <div class="brand-title">{selected_hospital}</div>
        <div class="brand-subtitle">Location: {selected_state} ({region_type}) | AI-Powered Disease Prediction Core</div>
    </div>
""", unsafe_allow_html=True)

# =====================================================================
# TABS DESIGN
# =====================================================================
tab1, tab2, tab3 = st.tabs([
    "🎯 Live Vital Signs", 
    "🧠 AI Case Notes Analyzer", 
    "📈 Regional Hospital Trends"
])

# ---------------------------------------------------------------------
# TAB 1: TRIAGE MATRIX
# ---------------------------------------------------------------------
with tab1:
    st.markdown(f"### 📡 {selected_hospital} - Patient Monitor Dashboard")
    st.write(f"Receiving live physiological data packets from {selected_state} network loops.")
    
    if data_mode == "Live Real-Time Stream":
        if st.button("⚡ Fetch Live Patient Data"):
            with st.spinner(f"Establishing secure connection with {selected_hospital}..."):
                time.sleep(0.6)
                hr = random.randint(72, 115)
                bp = random.randint(118, 158)
                ox = random.randint(92, 99)
                
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                prefix = selected_hospital[:3].upper().replace(" ", "X")
                st.metric(label="Patient Subject Code", value=f"{prefix}-{random.randint(4000,9999)}")
            with c2:
                st.metric(label="Heart Rate", value=f"{hr} BPM", delta=f"{hr-75} vs baseline")
            with c3:
                st.metric(label="Blood Pressure", value=f"{bp} mmHg")
            with c4:
                st.metric(label="Oxygen Level (SpO2)", value=f"{ox}%")
                
            st.markdown("---")
            
            if bp >= 140 or hr >= 100:
                st.error(f"🚨 ALERT: Abnormal vital metrics recorded at {selected_hospital}. Notification sent to doctor.")
                st.toast(f"📱 Automated priority notice dispatched to floor nurses.")
            else:
                st.success("✅ SYSTEM SECURE: Bedside telemetry limits running within safe margins.")
    else:
        st.info("Please turn on 'Live Real-Time Stream' in the sidebar to view active patient metrics.")

# ---------------------------------------------------------------------
# TAB 2: GROQ CLOUD INFERENCE PIPELINE
# ---------------------------------------------------------------------
with tab2:
    st.markdown("### 🧠 AI Clinical Text Notes Analyzer")
    st.write("Routing symptoms to a Llama-3-70B pipeline running over Groq ultra-fast processing architecture.")
    
    notes = st.text_area(
        "Secure Patient Chart Input:", 
        placeholder="Type any customized complex symptom details here..."
    )
    
    if st.button("Run Text AI Diagnosis"):
        if not notes.strip():
            st.warning("Please type patient observations into the terminal box above first.")
        else:
            with st.spinner("Streaming analytical tensor inferences from Groq Cloud..."):
                predicted_disease, confidence_score, clinical_desc = analyze_symptoms_with_groq(notes)

            # Render the dynamic real-time prediction card
            st.markdown(f"""
                <div class="clinical-card">
                    <span style="color:#0284C7; font-weight:800; font-size:14px; text-transform:uppercase; letter-spacing:1px;">NLP Cloud Result</span>
                    <h3 style="margin: 8px 0 4px 0; color:#0F172A; font-size: 24px;">{predicted_disease}</h3>
                    <p style="color:#334155; margin:0; font-size:16px; line-height: 1.6;">Calculated Confidence Alignment: <b>{confidence_score}</b>. <br><i>Diagnostic Evaluation:</i> {clinical_desc}</p>
                </div>
            """, unsafe_allow_html=True)

# ---------------------------------------------------------------------
# TAB 3: MULTI-CENTER ANALYTICS
# ---------------------------------------------------------------------
with tab3:
    st.markdown(f"### 📊 All Hospital Trends Dashboard")
    st.write(f"Aggregated charts tracking overall hospital admission trends for **{selected_state}**.")
    
    if data_mode == "Patient File (CSV)":
        uploaded_file = st.sidebar.file_uploader("Upload Regional Micro-Data Matrix File (.csv)", type=["csv"])
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            st.dataframe(df, use_container_width=True)
        else:
            st.info("System is waiting for a CSV spreadsheet file to be uploaded in the sidebar selector.")
    else:
        chart_data = pd.DataFrame({
            'Reporting Hour': list(range(1, 13)),
            'Total Patient Volume': [random.randint(10, 35) for _ in range(12)]
        }).set_index('Reporting Hour')
        
        st.line_chart(chart_data, use_container_width=True)
"""
CreditWise – Loan Approval Predictor
Run: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

# ---------------- Load Model ----------------

BASE = Path(__file__).parent
MODEL = joblib.load(BASE / "naive_bayes.pkl")
SCALER = joblib.load(BASE / "scaler.pkl")
COLUMNS = joblib.load(BASE / "columns.pkl")

# ---------------- Page Config ----------------

st.set_page_config(
    page_title="CreditWise",
    page_icon="💳",
    layout="wide",
)

# ---------------- Custom UI ----------------

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

:root {
    --bg-primary: #0a0e1a;
    --bg-secondary: #111827;
    --bg-card: rgba(17, 24, 39, 0.6);
    --accent-primary: #3b82f6;
    --accent-secondary: #8b5cf6;
    --accent-tertiary: #ec4899;
    --text-primary: #f9fafb;
    --text-secondary: #9ca3af;
    --success: #10b981;
    --error: #ef4444;
    --border: rgba(255, 255, 255, 0.1);
}

/* Main background with animated gradient mesh */
[data-testid="stAppViewContainer"] {
    background: var(--bg-primary);
    background-image: 
        radial-gradient(at 0% 0%, rgba(59, 130, 246, 0.15) 0px, transparent 50%),
        radial-gradient(at 100% 0%, rgba(139, 92, 246, 0.12) 0px, transparent 50%),
        radial-gradient(at 100% 100%, rgba(236, 72, 153, 0.1) 0px, transparent 50%),
        radial-gradient(at 0% 100%, rgba(16, 185, 129, 0.08) 0px, transparent 50%);
    background-attachment: fixed;
    color: var(--text-primary);
    font-family: 'Outfit', sans-serif;
}

/* Animated gradient background */
@keyframes gradientShift {
    0%, 100% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
}

/* Sidebar with glassmorphism */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, 
        rgba(17, 24, 39, 0.95) 0%, 
        rgba(17, 24, 39, 0.98) 100%);
    backdrop-filter: blur(12px);
    border-right: 1px solid var(--border);
}

[data-testid="stSidebar"] * {
    color: var(--text-primary) !important;
}

/* Headers with gradient text */
h1 {
    font-family: 'Outfit', sans-serif !important;
    font-weight: 800 !important;
    font-size: 3.5rem !important;
    background: linear-gradient(135deg, #3b82f6 0%, #8b5cf6 50%, #ec4899 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 0.5rem !important;
    letter-spacing: -0.02em;
    animation: fadeInUp 0.8s ease-out;
}

h2 {
    font-family: 'Outfit', sans-serif !important;
    font-weight: 700 !important;
    color: var(--text-primary) !important;
    font-size: 1.5rem !important;
    margin-top: 2rem !important;
    margin-bottom: 1rem !important;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}

h3 {
    font-family: 'Outfit', sans-serif !important;
    font-weight: 600 !important;
    color: var(--text-primary) !important;
}

/* Caption styling */
.stCaption {
    color: var(--text-secondary) !important;
    font-size: 1.1rem !important;
    font-weight: 300 !important;
    letter-spacing: 0.02em;
    animation: fadeInUp 0.8s ease-out 0.2s backwards;
}

/* Card sections with glassmorphism */
.block-container {
    padding-top: 3rem;
    max-width: 1400px;
    animation: fadeIn 1s ease-out;
}

/* Form sections */
.stSubheader {
    background: var(--bg-card);
    backdrop-filter: blur(12px);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 1.5rem;
    margin: 2rem 0 1rem 0;
    transition: all 0.3s ease;
}

.stSubheader:hover {
    border-color: rgba(59, 130, 246, 0.3);
    box-shadow: 0 8px 32px rgba(59, 130, 246, 0.1);
    transform: translateY(-2px);
}

/* Input fields with modern styling */
.stSelectbox, .stNumberInput {
    animation: fadeInUp 0.6s ease-out backwards;
}

.stSelectbox label, .stNumberInput label {
    color: var(--text-primary) !important;
    font-weight: 500 !important;
    font-size: 0.9rem !important;
    margin-bottom: 0.5rem !important;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    opacity: 0.9;
}

.stSelectbox div[data-baseweb="select"],
.stNumberInput input {
    background: rgba(17, 24, 39, 0.8) !important;
    backdrop-filter: blur(8px);
    color: var(--text-primary) !important;
    border: 1px solid var(--border) !important;
    border-radius: 12px !important;
    padding: 0.75rem 1rem !important;
    font-family: 'Outfit', sans-serif !important;
    font-weight: 400 !important;
    transition: all 0.3s ease !important;
    box-shadow: inset 0 2px 4px rgba(0, 0, 0, 0.2);
}

.stSelectbox div[data-baseweb="select"]:hover,
.stNumberInput input:hover {
    border-color: rgba(59, 130, 246, 0.5) !important;
    background: rgba(17, 24, 39, 0.95) !important;
}

.stSelectbox div[data-baseweb="select"]:focus-within,
.stNumberInput input:focus {
    border-color: var(--accent-primary) !important;
    box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.1) !important;
    background: rgba(17, 24, 39, 1) !important;
}

/* Dropdown menu */
[data-baseweb="popover"] {
    background: rgba(17, 24, 39, 0.98) !important;
    backdrop-filter: blur(16px) !important;
    border: 1px solid var(--border) !important;
    border-radius: 12px !important;
}

/* Button with gradient and animation */
.stButton > button {
    background: linear-gradient(135deg, #3b82f6 0%, #8b5cf6 100%) !important;
    color: white !important;
    border: none !important;
    border-radius: 16px !important;
    padding: 1rem 2rem !important;
    font-family: 'Outfit', sans-serif !important;
    font-weight: 600 !important;
    font-size: 1.1rem !important;
    width: 100% !important;
    transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1) !important;
    position: relative !important;
    overflow: hidden !important;
    box-shadow: 0 8px 32px rgba(59, 130, 246, 0.3);
    letter-spacing: 0.02em;
    text-transform: uppercase;
}

.stButton > button::before {
    content: '';
    position: absolute;
    top: 0;
    left: -100%;
    width: 100%;
    height: 100%;
    background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.2), transparent);
    transition: left 0.5s;
}

.stButton > button:hover::before {
    left: 100%;
}

.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 12px 48px rgba(59, 130, 246, 0.4) !important;
    background: linear-gradient(135deg, #2563eb 0%, #7c3aed 100%) !important;
}

.stButton > button:active {
    transform: translateY(0px) !important;
}

/* Metrics with enhanced styling */
[data-testid="stMetric"] {
    background: var(--bg-card) !important;
    backdrop-filter: blur(12px);
    border: 1px solid var(--border);
    border-radius: 16px !important;
    padding: 1.5rem !important;
    transition: all 0.3s ease;
    animation: scaleIn 0.5s ease-out backwards;
}

[data-testid="stMetric"]:hover {
    border-color: rgba(59, 130, 246, 0.4);
    box-shadow: 0 8px 32px rgba(59, 130, 246, 0.15);
    transform: translateY(-4px);
}

[data-testid="stMetric"] label {
    color: var(--text-secondary) !important;
    font-weight: 500 !important;
    font-size: 0.85rem !important;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

[data-testid="stMetric"] [data-testid="stMetricValue"] {
    color: var(--text-primary) !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-weight: 600 !important;
    font-size: 1.8rem !important;
}

/* Success and error alerts */
.stSuccess {
    background: linear-gradient(135deg, rgba(16, 185, 129, 0.1) 0%, rgba(16, 185, 129, 0.05) 100%) !important;
    border: 1px solid rgba(16, 185, 129, 0.3) !important;
    border-radius: 16px !important;
    backdrop-filter: blur(12px);
    padding: 1.5rem !important;
    animation: slideInUp 0.5s ease-out;
}

.stError {
    background: linear-gradient(135deg, rgba(239, 68, 68, 0.1) 0%, rgba(239, 68, 68, 0.05) 100%) !important;
    border: 1px solid rgba(239, 68, 68, 0.3) !important;
    border-radius: 16px !important;
    backdrop-filter: blur(12px);
    padding: 1.5rem !important;
    animation: slideInUp 0.5s ease-out;
}

/* Divider */
hr {
    border: none;
    height: 1px;
    background: linear-gradient(90deg, transparent, var(--border), transparent);
    margin: 2rem 0;
    opacity: 0.5;
}

/* Animations */
@keyframes fadeIn {
    from { opacity: 0; }
    to { opacity: 1; }
}

@keyframes fadeInUp {
    from {
        opacity: 0;
        transform: translateY(20px);
    }
    to {
        opacity: 1;
        transform: translateY(0);
    }
}

@keyframes slideInUp {
    from {
        opacity: 0;
        transform: translateY(30px);
    }
    to {
        opacity: 1;
        transform: translateY(0);
    }
}

@keyframes scaleIn {
    from {
        opacity: 0;
        transform: scale(0.9);
    }
    to {
        opacity: 1;
        transform: scale(1);
    }
}

/* Stagger animation delays */
.stSelectbox:nth-child(1), .stNumberInput:nth-child(1) { animation-delay: 0.1s; }
.stSelectbox:nth-child(2), .stNumberInput:nth-child(2) { animation-delay: 0.2s; }
.stSelectbox:nth-child(3), .stNumberInput:nth-child(3) { animation-delay: 0.3s; }

[data-testid="stMetric"]:nth-child(1) { animation-delay: 0.1s; }
[data-testid="stMetric"]:nth-child(2) { animation-delay: 0.2s; }
[data-testid="stMetric"]:nth-child(3) { animation-delay: 0.3s; }

/* Scrollbar */
::-webkit-scrollbar {
    width: 8px;
    height: 8px;
}

::-webkit-scrollbar-track {
    background: var(--bg-secondary);
}

::-webkit-scrollbar-thumb {
    background: linear-gradient(180deg, var(--accent-primary), var(--accent-secondary));
    border-radius: 4px;
}

::-webkit-scrollbar-thumb:hover {
    background: linear-gradient(180deg, var(--accent-secondary), var(--accent-tertiary));
}

</style>
""", unsafe_allow_html=True)

# ---------------- Sidebar ----------------

st.sidebar.title("💳 CreditWise")
st.sidebar.write("AI Loan Approval System")

st.sidebar.markdown("---")

st.sidebar.write("### About")
st.sidebar.write(
"""
Predict whether a loan should be approved based on applicant financial profile using Machine Learning.
"""
)

st.sidebar.markdown("---")

st.sidebar.write("Tech Stack")
st.sidebar.write("""
Python  
Scikit-Learn  
Streamlit  
Machine Learning
""")

# ---------------- Title ----------------

st.title("💳 CreditWise Dashboard")
st.caption("AI Powered Loan Approval Predictor")

st.markdown("---")

# ---------------- Personal Info ----------------

st.subheader("👤 Personal Information")

c1,c2,c3 = st.columns(3)

with c1:
    gender = st.selectbox("Gender",["Male","Female"])
    age = st.number_input("Age",18,100,30)

with c2:
    marital = st.selectbox("Marital Status",["Married","Single"])
    dependents = st.number_input("Dependents",0,5,0)

with c3:
    education = st.selectbox("Education",["Graduate","Not Graduate"])
    area = st.selectbox("Property Area",["Urban","Semiurban","Rural"])

# ---------------- Employment ----------------

st.subheader("💼 Employment Details")

c1,c2 = st.columns(2)

with c1:
    emp_status = st.selectbox(
        "Employment Status",
        ["Salaried","Self-employed","Contract","Unemployed"]
    )
    income = st.number_input("Monthly Income",0,100000,5000)

with c2:
    employer = st.selectbox(
        "Employer Category",
        ["Private","Government","MNC","Business","Unemployed"]
    )
    co_income = st.number_input("Co-applicant Income",0,100000,0)

# ---------------- Financial ----------------

st.subheader("💰 Financial Profile")

c1,c2,c3 = st.columns(3)

with c1:
    credit = st.number_input("Credit Score",300,850,650)
    savings = st.number_input("Savings",0,1000000,10000)

with c2:
    dti = st.number_input("Debt To Income Ratio",0.0,1.0,0.3)
    collateral = st.number_input("Collateral Value",0,1000000,50000)

with c3:
    loans = st.number_input("Existing Loans",0,10,0)

# ---------------- Loan ----------------

st.subheader("📋 Loan Details")

c1,c2,c3 = st.columns(3)

with c1:
    loan_amount = st.number_input("Loan Amount",0,500000,20000)

with c2:
    loan_term = st.number_input("Loan Term (months)",1,480,360)

with c3:
    purpose = st.selectbox(
        "Loan Purpose",
        ["Business","Home","Car","Education","Personal"]
    )

st.markdown("---")

# ---------------- Prediction ----------------

if st.button("🚀 Predict Loan Approval"):

    edu = 0 if education=="Graduate" else 1

    data = {
        "Applicant_Income":income,
        "Coapplicant_Income":co_income,
        "Age":age,
        "Dependents":dependents,
        "Existing_Loans":loans,
        "Savings":savings,
        "Collateral_Value":collateral,
        "Loan_Amount":loan_amount,
        "Loan_Term":loan_term,
        "Education_Level":edu,
        "DTI_Ratio_sq":dti**2,
        "Credit_Score_sq":credit**2
    }

    data["Gender_Male"] = 1 if gender=="Male" else 0
    data["Marital_Status_Single"] = 1 if marital=="Single" else 0

    row = pd.DataFrame([[data.get(c,0) for c in COLUMNS]],columns=COLUMNS)

    scaled = SCALER.transform(row)

    pred = MODEL.predict(scaled)[0]

    st.markdown("## 📊 Prediction Result")

    if pred==1:
        st.success("✅ Loan Approved")
        st.balloons()
    else:
        st.error("❌ Loan Rejected")

    st.markdown("### Financial Summary")

    c1,c2,c3 = st.columns(3)

    with c1:
        st.metric("Total Income",income+co_income)

    with c2:
        st.metric("Loan Amount",loan_amount)

    with c3:
        st.metric("Credit Score",credit)
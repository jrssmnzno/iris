import streamlit as st
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Iris Flower Predictor",
    page_icon="🌸",
    layout="centered",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Mono:wght@400;500&family=DM+Sans:wght@300;400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: #0e0e12;
    color: #f0ece4;
}
.stApp { background-color: #0e0e12; }
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 2rem 2.5rem 4rem; max-width: 800px; }

.hero {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
    border: 1px solid #2a2a4a;
    border-radius: 20px;
    padding: 2.5rem 3rem;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
}
.hero::before {
    content: '🌸';
    position: absolute;
    right: 2rem; top: 50%;
    transform: translateY(-50%);
    font-size: 6rem;
    opacity: 0.15;
}
.hero h1 {
    font-family: 'DM Serif Display', serif;
    font-size: 2.8rem;
    color: #e8d5c4;
    margin: 0 0 0.4rem;
    letter-spacing: -1px;
}
.hero p {
    font-size: 1rem;
    color: #8b8fa8;
    margin: 0;
    font-weight: 300;
}

.card {
    background: #13131d;
    border: 1px solid #252535;
    border-radius: 16px;
    padding: 2rem;
    margin-bottom: 1.5rem;
}

.section-label {
    font-size: 0.72rem;
    color: #5a5a78;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    margin-bottom: 1rem;
    display: block;
}

.result-box {
    background: #0d2e1a;
    border: 1px solid #2d6a4f;
    border-radius: 12px;
    padding: 1.5rem;
    text-align: center;
    margin-top: 1rem;
}
.result-box .species-name {
    font-family: 'DM Serif Display', serif;
    font-size: 2.2rem;
    color: #6ec98a;
    display: block;
    margin-bottom: 0.3rem;
}
.result-box .result-label {
    font-size: 0.8rem;
    color: #5a5a78;
    text-transform: uppercase;
    letter-spacing: 1px;
}

.prob-row {
    display: flex;
    align-items: center;
    gap: 1rem;
    margin-bottom: 0.6rem;
}
.prob-label {
    font-family: 'DM Mono', monospace;
    font-size: 0.8rem;
    color: #8b8fa8;
    width: 90px;
    flex-shrink: 0;
    text-transform: capitalize;
}
.prob-bar-bg {
    flex: 1;
    background: #1e1e2e;
    border-radius: 99px;
    height: 8px;
    overflow: hidden;
}
.prob-bar-fill {
    height: 100%;
    border-radius: 99px;
    transition: width 0.4s ease;
}
.prob-val {
    font-family: 'DM Mono', monospace;
    font-size: 0.75rem;
    color: #5a5a78;
    width: 36px;
    text-align: right;
    flex-shrink: 0;
}

.stButton > button {
    background: linear-gradient(135deg, #c9a96e, #a07840) !important;
    color: #0e0e12 !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 500 !important;
    font-size: 0.95rem !important;
    padding: 0.65rem 1.2rem !important;
    width: 100% !important;
    transition: opacity .2s !important;
    margin-top: 0.5rem !important;
}
.stButton > button:hover { opacity: 0.85 !important; }

.stSlider > div > div > div { background: #c9a96e !important; }

div[data-testid="stSlider"] label {
    font-size: 0.82rem !important;
    color: #8b8fa8 !important;
}
</style>
""", unsafe_allow_html=True)

# ── Cached data & model ───────────────────────────────────────────────────────
@st.cache_data
def load_data():
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    return df, iris

@st.cache_resource
def train_model():
    _, iris = load_data()
    X = pd.DataFrame(iris.data, columns=iris.feature_names)
    y = iris.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model, iris

df, iris = load_data()
model, iris = train_model()

COLORS = ['#c9a96e', '#6ea8c9', '#6ec98a']
SPECIES_EMOJI = {'setosa': '🌼', 'versicolor': '🌿', 'virginica': '🌺'}

# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class='hero'>
    <h1>Iris Predictor</h1>
    <p>Enter flower measurements below and the model will identify the species.</p>
</div>
""", unsafe_allow_html=True)

# ── Input Card ────────────────────────────────────────────────────────────────
st.markdown("<div class='card'>", unsafe_allow_html=True)
st.markdown("<span class='section-label'>🌿 Sepal Measurements</span>", unsafe_allow_html=True)
col1, col2 = st.columns(2)
with col1:
    sl = st.slider("Sepal Length (cm)", 4.0, 8.0, 5.8, 0.1, key="sl")
with col2:
    sw = st.slider("Sepal Width (cm)",  2.0, 4.5, 3.0, 0.1, key="sw")

st.markdown("<br><span class='section-label'>🌸 Petal Measurements</span>", unsafe_allow_html=True)
col3, col4 = st.columns(2)
with col3:
    pl = st.slider("Petal Length (cm)", 1.0, 7.0, 4.0, 0.1, key="pl")
with col4:
    pw = st.slider("Petal Width (cm)",  0.1, 2.5, 1.2, 0.1, key="pw")

st.markdown("</div>", unsafe_allow_html=True)

# ── Predict Button ────────────────────────────────────────────────────────────
predict_clicked = st.button("🔍 Predict Species", key="btn_pred")

# ── Result ────────────────────────────────────────────────────────────────────
if predict_clicked:
    input_df = pd.DataFrame([[sl, sw, pl, pw]], columns=iris.feature_names)
    pred = model.predict(input_df)[0]
    proba = model.predict_proba(input_df)[0]
    species_name = iris.target_names[pred]
    emoji = SPECIES_EMOJI.get(species_name, '🌸')

    st.markdown(f"""
    <div class='result-box'>
        <span class='species-name'>{emoji} {species_name.capitalize()}</span>
        <span class='result-label'>Predicted Species</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br><span class='section-label'>Prediction Confidence</span>", unsafe_allow_html=True)
    for i, (sp, prob) in enumerate(zip(iris.target_names, proba)):
        pct = int(prob * 100)
        st.markdown(f"""
        <div class='prob-row'>
            <span class='prob-label'>{sp}</span>
            <div class='prob-bar-bg'>
                <div class='prob-bar-fill' style='width:{pct}%; background:{COLORS[i]};'></div>
            </div>
            <span class='prob-val'>{pct}%</span>
        </div>
        """, unsafe_allow_html=True)

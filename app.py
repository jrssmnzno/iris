import streamlit as st
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

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
.block-container { padding: 2rem 2.5rem 4rem; max-width: 860px; }

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
}
.prob-val {
    font-family: 'DM Mono', monospace;
    font-size: 0.75rem;
    color: #5a5a78;
    width: 36px;
    text-align: right;
    flex-shrink: 0;
}

.dataset-card {
    background: #13131d;
    border: 1px solid #252535;
    border-radius: 16px;
    padding: 2rem;
    margin-top: 2rem;
}
.dataset-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 1.2rem;
    flex-wrap: wrap;
    gap: 0.5rem;
}
.dataset-title {
    font-family: 'DM Serif Display', serif;
    font-size: 1.3rem;
    color: #e8d5c4;
}
.dataset-badge {
    background: #1e1e2e;
    border: 1px solid #2a2a3a;
    border-radius: 99px;
    padding: 0.25rem 0.8rem;
    font-family: 'DM Mono', monospace;
    font-size: 0.72rem;
    color: #c9a96e;
}

/* highlight row matching user input */
.your-input-row {
    background: #1a2e1a !important;
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

/* Tab styling */
.stTabs [data-baseweb="tab-list"] {
    background: #1a1a2a !important;
    border-radius: 10px !important;
    padding: 4px !important;
    gap: 4px !important;
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    color: #5a5a78 !important;
    border-radius: 8px !important;
    font-size: 0.82rem !important;
    font-family: 'DM Sans', sans-serif !important;
}
.stTabs [aria-selected="true"] {
    background: #252535 !important;
    color: #e8d5c4 !important;
}

/* Dataframe dark theme */
.stDataFrame {
    border-radius: 10px;
    overflow: hidden;
}
[data-testid="stDataFrameResizable"] {
    background: #0e0e12 !important;
}

/* Divider */
.divider {
    border: none;
    border-top: 1px solid #1e1e2e;
    margin: 2rem 0;
}
</style>
""", unsafe_allow_html=True)

# ── Cached data & model ───────────────────────────────────────────────────────
@st.cache_data
def load_data():
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)
    return df, iris

@st.cache_resource
def load_model():
    with open("iris_model.pkl", "rb") as f:
        return pickle.load(f)

df, iris = load_data()
model = load_model()

COLORS  = ['#c9a96e', '#6ea8c9', '#6ec98a']
SPECIES_EMOJI = {'setosa': '🌼', 'versicolor': '🌿', 'virginica': '🌺'}
SPECIES_COLOR = {'setosa': '#c9a96e', 'versicolor': '#6ea8c9', 'virginica': '#6ec98a'}

# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class='hero'>
    <h1>Iris Predictor</h1>
    <p>Enter flower measurements — the model will identify the species instantly.</p>
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
predicted_species = None
if predict_clicked:
    input_df = pd.DataFrame([[sl, sw, pl, pw]], columns=iris.feature_names)
    pred      = model.predict(input_df)[0]
    proba     = model.predict_proba(input_df)[0]
    predicted_species = iris.target_names[pred]
    emoji     = SPECIES_EMOJI.get(predicted_species, '🌸')
    sp_color  = SPECIES_COLOR.get(predicted_species, '#6ec98a')

    st.markdown(f"""
    <div class='result-box' style='border-color:{sp_color}40; background:{sp_color}12;'>
        <span class='species-name' style='color:{sp_color};'>{emoji} {predicted_species.capitalize()}</span>
        <span class='result-label'>Predicted Species</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br><span class='section-label'>Prediction Confidence</span>", unsafe_allow_html=True)
    for i, (sp, prob) in enumerate(zip(iris.target_names, proba)):
        pct = int(prob * 100)
        st.markdown(f"""
        <div class='prob-row'>
            <span class='prob-label'>{SPECIES_EMOJI.get(sp,'')} {sp}</span>
            <div class='prob-bar-bg'>
                <div class='prob-bar-fill' style='width:{pct}%; background:{COLORS[i]};'></div>
            </div>
            <span class='prob-val'>{pct}%</span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

# ── Divider ───────────────────────────────────────────────────────────────────
st.markdown("<hr class='divider'>", unsafe_allow_html=True)

# ── Dataset Table ─────────────────────────────────────────────────────────────
st.markdown(f"""
<div class='dataset-header'>
    <span class='dataset-title'>📋 Iris Dataset Reference</span>
    <span class='dataset-badge'>150 samples · 3 species · 4 features</span>
</div>
""", unsafe_allow_html=True)
st.caption("Use this table to compare your inputs against real measurements in the dataset.")

tab_all, tab_setosa, tab_versicolor, tab_virginica = st.tabs([
    "🌸 All Species",
    "🌼 Setosa",
    "🌿 Versicolor",
    "🌺 Virginica"
])

# Build a display df with rounded values and row index starting at 1
display_df = df.copy()
display_df.index = range(1, len(display_df) + 1)
display_df.index.name = "#"
for col in iris.feature_names:
    display_df[col] = display_df[col].round(1)

# If user has predicted, highlight rows closest to their input
def highlight_closest(row, user_vals, feature_cols, top_n=3):
    user_arr = pd.Series(user_vals, index=feature_cols)
    dist = ((display_df[feature_cols] - user_arr) ** 2).sum(axis=1)
    closest_idx = dist.nsmallest(top_n).index
    if row.name in closest_idx:
        return ['background-color: #1a2e20; color: #6ec98a'] * len(row)
    return [''] * len(row)

user_vals  = [sl, sw, pl, pw]
feat_cols  = list(iris.feature_names)

with tab_all:
    if predict_clicked:
        st.caption(f"🟢 Highlighted rows = closest matches to your input  ({sl}, {sw}, {pl}, {pw})")
        styled = display_df.style.apply(
            highlight_closest, user_vals=user_vals,
            feature_cols=feat_cols, top_n=5, axis=1
        )
        st.dataframe(styled, use_container_width=True, height=360)
    else:
        st.dataframe(display_df, use_container_width=True, height=360)

with tab_setosa:
    sub = display_df[display_df['species'] == 'setosa']
    st.caption(f"50 samples — small petals, wide sepals")
    st.dataframe(sub, use_container_width=True, height=320)

with tab_versicolor:
    sub = display_df[display_df['species'] == 'versicolor']
    st.caption(f"50 samples — medium measurements across all features")
    st.dataframe(sub, use_container_width=True, height=320)

with tab_virginica:
    sub = display_df[display_df['species'] == 'virginica']
    st.caption(f"50 samples — largest petals of the three species")
    st.dataframe(sub, use_container_width=True, height=320)

# ── Summary Stats ─────────────────────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
with st.expander("📈 View Summary Statistics"):
    st.dataframe(
        df.groupby('species')[iris.feature_names]
        .mean().round(2)
        .rename_axis("Species")
        .style.background_gradient(cmap='YlOrBr', axis=None),
        use_container_width=True
    )
    st.caption("Average measurements per species — useful for manual comparison.")

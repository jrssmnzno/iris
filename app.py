import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Iris Explorer",
    page_icon="🌸",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Mono:wght@400;500&family=DM+Sans:wght@300;400;500&display=swap');

/* ── Base ── */
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: #0e0e12;
    color: #f0ece4;
}
.stApp { background-color: #0e0e12; }

/* ── Hide default streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 2rem 2.5rem 4rem; max-width: 1400px; }

/* ── Hero banner ── */
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
    font-size: 3rem;
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

/* ── Stat pills ── */
.stat-row { display: flex; gap: 1rem; margin-bottom: 2rem; flex-wrap: wrap; }
.stat-pill {
    background: #161620;
    border: 1px solid #2a2a3a;
    border-radius: 12px;
    padding: 1rem 1.5rem;
    flex: 1; min-width: 130px;
    text-align: center;
}
.stat-pill .val {
    font-family: 'DM Mono', monospace;
    font-size: 1.8rem;
    color: #c9a96e;
    display: block;
    font-weight: 500;
}
.stat-pill .lbl {
    font-size: 0.72rem;
    color: #5a5a78;
    text-transform: uppercase;
    letter-spacing: 1px;
}

/* ── Section header ── */
.section-title {
    font-family: 'DM Serif Display', serif;
    font-size: 1.5rem;
    color: #e8d5c4;
    margin: 0 0 1.2rem;
    padding-bottom: 0.5rem;
    border-bottom: 1px solid #2a2a3a;
}

/* ── Feature grid cards ── */
.feature-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 1.2rem;
    margin-bottom: 2.5rem;
}
.feat-card {
    background: #13131d;
    border: 1px solid #252535;
    border-radius: 16px;
    padding: 1.6rem;
    transition: border-color .2s;
}
.feat-card:hover { border-color: #c9a96e; }
.feat-card .card-icon { font-size: 1.8rem; margin-bottom: 0.6rem; display: block; }
.feat-card .card-title {
    font-family: 'DM Serif Display', serif;
    font-size: 1.15rem;
    color: #e8d5c4;
    margin-bottom: 0.3rem;
}
.feat-card .card-desc {
    font-size: 0.8rem;
    color: #5a5a78;
    margin-bottom: 1.2rem;
    line-height: 1.5;
}

/* ── Dataset section ── */
.dataset-section {
    background: #13131d;
    border: 1px solid #252535;
    border-radius: 16px;
    padding: 1.6rem;
    margin-top: 1rem;
}

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(135deg, #c9a96e, #a07840) !important;
    color: #0e0e12 !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 500 !important;
    font-size: 0.85rem !important;
    padding: 0.5rem 1.2rem !important;
    width: 100% !important;
    transition: opacity .2s !important;
}
.stButton > button:hover { opacity: 0.85 !important; }

/* ── Sliders & selects ── */
.stSlider > div > div > div { background: #c9a96e !important; }
.stSelectbox > div > div { background: #161620 !important; border-color: #2a2a3a !important; }
.stMultiSelect > div > div { background: #161620 !important; border-color: #2a2a3a !important; }

/* ── Dataframe ── */
.stDataFrame { border-radius: 10px; overflow: hidden; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: #0b0b10 !important;
    border-right: 1px solid #1e1e2e !important;
}
[data-testid="stSidebar"] .stMarkdown h2 {
    font-family: 'DM Serif Display', serif;
    color: #c9a96e;
    font-size: 1.1rem;
}

/* ── Success / info boxes ── */
.stSuccess { background: #0d2e1a !important; border-color: #2d6a4f !important; }
.stInfo { background: #0d1e2e !important; border-color: #1a4a6e !important; }

/* ── Plot backgrounds ── */
.stPyplotChart { border-radius: 10px; overflow: hidden; }

/* ── Spinner ── */
.stSpinner > div { border-top-color: #c9a96e !important; }
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
def train_model(n_estimators):
    _, iris = load_data()
    X = pd.DataFrame(iris.data, columns=iris.feature_names)
    y = iris.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    cm = confusion_matrix(y_test, preds)
    return model, acc, cm

def dark_fig():
    fig, ax = plt.subplots(figsize=(5, 3.5))
    fig.patch.set_facecolor('#13131d')
    ax.set_facecolor('#13131d')
    ax.tick_params(colors='#8b8fa8', labelsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor('#2a2a3a')
    ax.xaxis.label.set_color('#8b8fa8')
    ax.yaxis.label.set_color('#8b8fa8')
    ax.title.set_color('#e8d5c4')
    return fig, ax

PALETTE = ['#c9a96e', '#6ea8c9', '#6ec98a']

# ── Load data ─────────────────────────────────────────────────────────────────
df, iris = load_data()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Controls")
    st.markdown("---")
    species_filter = st.multiselect(
        "Filter Species",
        options=list(df['species'].unique()),
        default=list(df['species'].unique())
    )
    n_estimators = st.slider("🌲 Number of Trees", 10, 200, 100, step=10)
    feature_x = st.selectbox("📊 Distribution Feature", iris.feature_names, index=2)
    st.markdown("---")
    st.markdown("<small style='color:#5a5a78'>Iris Dataset · UCI ML Repository<br>150 samples · 3 species · 4 features</small>", unsafe_allow_html=True)

filtered_df = df[df['species'].isin(species_filter)] if species_filter else df

# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class='hero'>
    <h1>Iris Explorer</h1>
    <p>Interactive machine learning dashboard · Classic UCI Iris Dataset</p>
</div>
""", unsafe_allow_html=True)

# ── Stat pills ────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class='stat-row'>
    <div class='stat-pill'><span class='val'>{len(filtered_df)}</span><span class='lbl'>Samples</span></div>
    <div class='stat-pill'><span class='val'>4</span><span class='lbl'>Features</span></div>
    <div class='stat-pill'><span class='val'>{filtered_df['species'].nunique()}</span><span class='lbl'>Species</span></div>
    <div class='stat-pill'><span class='val'>UCI</span><span class='lbl'>Source</span></div>
    <div class='stat-pill'><span class='val'>{n_estimators}</span><span class='lbl'>Trees</span></div>
</div>
""", unsafe_allow_html=True)

# ── 2×2 Feature Grid ──────────────────────────────────────────────────────────
st.markdown("<div class='section-title'>Feature Panels</div>", unsafe_allow_html=True)

col_a, col_b = st.columns(2, gap="medium")

# ── Card 1 · Feature Distribution ────────────────────────────────────────────
with col_a:
    st.markdown("""
    <div class='feat-card'>
        <span class='card-icon'>📊</span>
        <div class='card-title'>Feature Distribution</div>
        <div class='card-desc'>Histogram with KDE overlay per species for the selected feature.</div>
    </div>
    """, unsafe_allow_html=True)
    if st.button("Show Distribution", key="btn_dist"):
        with st.spinner("Rendering..."):
            fig, ax = dark_fig()
            for i, (sp, grp) in enumerate(filtered_df.groupby('species')):
                grp[feature_x].plot.hist(ax=ax, alpha=0.5, color=PALETTE[i], label=sp, bins=15)
                grp[feature_x].plot.kde(ax=ax, color=PALETTE[i], linewidth=2)
            ax.set_title(f"{feature_x}", fontsize=10)
            ax.legend(fontsize=7, facecolor='#1a1a2e', labelcolor='#e8d5c4')
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

# ── Card 2 · Pairplot ─────────────────────────────────────────────────────────
with col_b:
    st.markdown("""
    <div class='feat-card'>
        <span class='card-icon'>🔍</span>
        <div class='card-title'>Pairplot</div>
        <div class='card-desc'>Scatter matrix across all 4 features colored by species.</div>
    </div>
    """, unsafe_allow_html=True)
    if st.button("Show Pairplot", key="btn_pair"):
        with st.spinner("Rendering pairplot — this may take a moment..."):
            sns.set_style("darkgrid", {"axes.facecolor": "#13131d", "figure.facecolor": "#13131d"})
            fig2 = sns.pairplot(
                filtered_df, hue='species',
                palette=dict(zip(iris.target_names, PALETTE)),
                plot_kws={'alpha': 0.6, 's': 20},
                diag_kind='kde'
            )
            fig2.figure.patch.set_facecolor('#13131d')
            st.pyplot(fig2)
            plt.close()

col_c, col_d = st.columns(2, gap="medium")

# ── Card 3 · Random Forest ────────────────────────────────────────────────────
with col_c:
    st.markdown("""
    <div class='feat-card'>
        <span class='card-icon'>🤖</span>
        <div class='card-title'>Random Forest Classifier</div>
        <div class='card-desc'>Train a model and evaluate accuracy, confusion matrix, and feature importance.</div>
    </div>
    """, unsafe_allow_html=True)
    if st.button("Train & Evaluate", key="btn_model"):
        with st.spinner("Training..."):
            model, acc, cm = train_model(n_estimators)
        st.success(f"✅ Accuracy: **{acc * 100:.2f}%**")

        # Confusion matrix
        fig3, ax3 = dark_fig()
        sns.heatmap(
            cm, annot=True, fmt='d', cmap='YlOrBr',
            xticklabels=iris.target_names,
            yticklabels=iris.target_names,
            ax=ax3, linewidths=0.5, linecolor='#0e0e12',
            annot_kws={"size": 10}
        )
        ax3.set_xlabel("Predicted", fontsize=8)
        ax3.set_ylabel("Actual", fontsize=8)
        ax3.set_title("Confusion Matrix", fontsize=10)
        plt.tight_layout()
        st.pyplot(fig3)
        plt.close(fig3)

        # Feature importance
        imp_df = pd.DataFrame({
            'Feature': iris.feature_names,
            'Importance': model.feature_importances_
        }).sort_values('Importance')
        fig4, ax4 = dark_fig()
        ax4.barh(imp_df['Feature'], imp_df['Importance'], color=PALETTE[0], alpha=0.85)
        ax4.set_title("Feature Importance", fontsize=10)
        plt.tight_layout()
        st.pyplot(fig4)
        plt.close(fig4)

# ── Card 4 · Predict Your Own Flower ─────────────────────────────────────────
with col_d:
    st.markdown("""
    <div class='feat-card'>
        <span class='card-icon'>🌺</span>
        <div class='card-title'>Predict Your Own Flower</div>
        <div class='card-desc'>Adjust the measurements and predict the species in real time.</div>
    </div>
    """, unsafe_allow_html=True)
    p1, p2 = st.columns(2)
    with p1:
        sl = st.slider("Sepal Length", 4.0, 8.0, 5.8, 0.1, key="sl")
        sw = st.slider("Sepal Width",  2.0, 4.5, 3.0, 0.1, key="sw")
    with p2:
        pl = st.slider("Petal Length", 1.0, 7.0, 4.0, 0.1, key="pl")
        pw = st.slider("Petal Width",  0.1, 2.5, 1.2, 0.1, key="pw")

    if st.button("Predict Species", key="btn_pred"):
        model, _, _ = train_model(n_estimators)
        input_df = pd.DataFrame([[sl, sw, pl, pw]], columns=iris.feature_names)
        pred = model.predict(input_df)[0]
        proba = model.predict_proba(input_df)[0]
        species_name = iris.target_names[pred].capitalize()
        st.success(f"🌸 **{species_name}**")

        # Probability bar chart
        fig5, ax5 = dark_fig()
        bars = ax5.barh(iris.target_names, proba, color=PALETTE, alpha=0.85)
        ax5.set_xlim(0, 1)
        ax5.set_title("Prediction Probability", fontsize=10)
        for bar, p in zip(bars, proba):
            ax5.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height()/2,
                     f"{p:.2f}", va='center', color='#8b8fa8', fontsize=8)
        plt.tight_layout()
        st.pyplot(fig5)
        plt.close(fig5)

# ── Dataset Preview (bottom) ──────────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
st.markdown("<div class='section-title'>📋 Dataset Preview</div>", unsafe_allow_html=True)
with st.container():
    st.markdown("<div class='dataset-section'>", unsafe_allow_html=True)
    tab1, tab2 = st.tabs(["📄 Raw Data", "📈 Summary Stats"])
    with tab1:
        st.dataframe(filtered_df, use_container_width=True, height=280)
        st.caption(f"{filtered_df.shape[0]} rows × {filtered_df.shape[1]} columns")
    with tab2:
        st.dataframe(filtered_df.describe().round(2), use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

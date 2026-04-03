import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

# --- CACHED DATA LOADER ---
@st.cache_data
def load_data():
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)
    return df, iris

# --- CACHED MODEL TRAINER ---
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
    return model, acc, cm, X_test, y_test

# --- LOAD DATA ---
df, iris = load_data()

# --- SIDEBAR ---
st.sidebar.title("⚙️ Settings")
species_filter = st.sidebar.multiselect(
    "Filter by Species",
    options=df['species'].unique(),
    default=df['species'].unique()
)
n_estimators = st.sidebar.slider("Number of Trees", 10, 200, 100)

filtered_df = df[df['species'].isin(species_filter)]

# --- HEADER ---
st.title("🌸 Iris Dataset Explorer")
st.write("An interactive ML app using the classic Iris dataset.")

# --- DATA PREVIEW ---
st.subheader("📋 Dataset Preview")
st.dataframe(filtered_df)
st.write(f"Showing **{filtered_df.shape[0]}** rows × **{filtered_df.shape[1]}** columns")

# --- STATS BEHIND BUTTON ---
st.subheader("📈 Summary Statistics")
if st.button("Show Summary Stats"):
    st.dataframe(filtered_df.describe())

# --- HISTOGRAM BEHIND BUTTON ---
st.subheader("📊 Feature Distribution")
feature = st.selectbox("Choose a feature", iris.feature_names)
if st.button("Show Distribution Plot"):
    fig, ax = plt.subplots()
    sns.histplot(data=filtered_df, x=feature, hue='species', kde=True, ax=ax)
    ax.set_title(f"Distribution of {feature}")
    st.pyplot(fig)
    plt.close(fig)

# --- PAIRPLOT BEHIND BUTTON ---
st.subheader("🔍 Pairplot (All Features)")
st.caption("⚠️ This chart is expensive to render — click only when needed.")
if st.button("Show Pairplot"):
    with st.spinner("Rendering pairplot..."):
        fig2 = sns.pairplot(filtered_df, hue='species')
        st.pyplot(fig2)
        plt.close()

# --- MODEL TRAINING BEHIND BUTTON ---
st.subheader("🤖 Random Forest Classifier")
st.caption(f"Training with **{n_estimators} trees** on 80% of the data.")
if st.button("Train & Evaluate Model"):
    with st.spinner("Training model..."):
        model, acc, cm, X_test, y_test = train_model(n_estimators)

    st.success(f"✅ Model Accuracy: **{acc * 100:.2f}%**")

    # Confusion Matrix
    st.write("**Confusion Matrix:**")
    fig3, ax3 = plt.subplots()
    sns.heatmap(
        cm, annot=True, fmt='d',
        xticklabels=iris.target_names,
        yticklabels=iris.target_names,
        ax=ax3, cmap='Blues'
    )
    ax3.set_xlabel("Predicted")
    ax3.set_ylabel("Actual")
    st.pyplot(fig3)
    plt.close(fig3)

    # Feature Importance
    st.write("**Feature Importance:**")
    importance_df = pd.DataFrame({
        'Feature': iris.feature_names,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False)
    fig4, ax4 = plt.subplots()
    sns.barplot(data=importance_df, x='Importance', y='Feature', ax=ax4, palette='viridis')
    ax4.set_title("Feature Importance")
    st.pyplot(fig4)
    plt.close(fig4)

# --- PREDICT YOUR OWN FLOWER ---
st.subheader("🌺 Predict Your Own Flower")
st.caption("Adjust the sliders and click Predict.")
col1, col2 = st.columns(2)
with col1:
    sepal_length = st.slider("Sepal Length (cm)", 4.0, 8.0, 5.8)
    sepal_width  = st.slider("Sepal Width (cm)",  2.0, 4.5, 3.0)
with col2:
    petal_length = st.slider("Petal Length (cm)", 1.0, 7.0, 4.0)
    petal_width  = st.slider("Petal Width (cm)",  0.1, 2.5, 1.2)

if st.button("Predict Species"):
    model, _, _, _, _ = train_model(n_estimators)
    input_data = pd.DataFrame([[sepal_length, sepal_width, petal_length, petal_width]],
                               columns=iris.feature_names)
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0]
    st.success(f"🌸 Predicted Species: **{iris.target_names[prediction].capitalize()}**")
    prob_df = pd.DataFrame({'Species': iris.target_names, 'Probability': probability})
    st.dataframe(prob_df)

import streamlit as st
import joblib
import numpy as np

# --- Load saved model and vectorizer ---
model = joblib.load("sentiment_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# --- Streamlit setup ---
st.set_page_config(page_title="Twitter Sentiment Analyzer", page_icon="💬", layout="centered")
st.title("💬 Twitter Sentiment Analyzer")
st.write("Classify any tweet as **Positive**, **Neutral**, **Negative**, or **Irrelevant** using a trained ML model.")

# --- Label mapping based on your dataset ---
label_map = {
    3: "Positive",
    2: "Neutral",
    1: "Negative",
    0: "Irrelevant"
}

# --- Text input ---
user_input = st.text_area("✍️ Enter a tweet:", height=120)

# --- Analyze sentiment button ---
if st.button("Analyze Sentiment"):
    if user_input.strip():
        # Preprocess input
        clean_text = [user_input.lower()]
        X_input = vectorizer.transform(clean_text)

        # Get probabilities
        proba = model.predict_proba(X_input)[0]
        maxp = float(np.max(proba))
        pred_encoded = int(np.argmax(proba))

        # --- Confidence-based correction ---
        if maxp < 0.45:
            pred_label = "Irrelevant"
        elif maxp < 0.60 and pred_encoded == 3:
            pred_label = "Neutral"
        else:
            pred_label = label_map.get(pred_encoded, f"Unknown ({pred_encoded})")

        # --- Display result ---
        if pred_label == "Positive":
            st.success(f"### 😊 Sentiment: **{pred_label}** (confidence: {maxp:.2f})")
        elif pred_label == "Negative":
            st.error(f"### 😠 Sentiment: **{pred_label}** (confidence: {maxp:.2f})")
        elif pred_label == "Neutral":
            st.info(f"### 😐 Sentiment: **{pred_label}** (confidence: {maxp:.2f})")
        elif pred_label == "Irrelevant":
            st.warning(f"### 🤔 Sentiment: **{pred_label}** (confidence: {maxp:.2f})")
        else:
            st.write(f"Sentiment: {pred_label} (confidence: {maxp:.2f})")

        # --- Optional: show all probabilities ---
        st.write("#### Class probabilities:")
        for k, v in sorted(label_map.items(), key=lambda x: x[0]):
            st.write(f"**{v}**: {proba[k]:.3f}")

    else:
        st.warning("⚠️ Please enter some text before analyzing.")

# --- Footer ---
st.markdown("---")
st.markdown("Made with ❤️ using **Streamlit**, **Scikit-learn**, and **Python**.")

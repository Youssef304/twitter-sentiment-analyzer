import streamlit as st
import joblib
import numpy as np
import re, string

# ---------------------------
# Load trained model & vectorizer
# ---------------------------
model = joblib.load("sentiment_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")
le = joblib.load("label_encoder.pkl")

# ---------------------------
# Text Cleaning Function
# ---------------------------
def clean_text(text):
    text = re.sub(r"http\S+|www\S+|https\S+", '', text)
    text = re.sub(r'\@\w+|\#','', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = text.lower()
    return text

# ---------------------------
# Streamlit UI
# ---------------------------
st.set_page_config(page_title="Twitter Sentiment Analyzer", page_icon="💬", layout="centered")

st.title("💬 Twitter Sentiment Analyzer")
st.write("Enter a tweet below and instantly get its **sentiment prediction** using a trained MLP model!")

tweet = st.text_area("✏️ Enter your tweet:", "")

if st.button("🔍 Analyze Sentiment"):
    if tweet.strip() == "":
        st.warning("⚠️ Please enter a tweet first.")
    else:
        # Clean and vectorize
        clean_tweet = clean_text(tweet)
        vectorized_tweet = vectorizer.transform([clean_tweet])

        # Predict sentiment
        prediction = model.predict(vectorized_tweet)
        proba = model.predict_proba(vectorized_tweet)

        # Decode label if numeric
        if isinstance(prediction[0], (int, float, np.integer)):
            sentiment = le.inverse_transform(prediction)[0]
        else:
            sentiment = prediction[0]

        sentiment = str(sentiment).lower()

        # ---------------------------
        # Display Results
        # ---------------------------
        st.subheader("🧠 Predicted Sentiment:")

        if "pos" in sentiment:
            st.success(f"😀 Positive")
            st.progress(float(np.max(proba)))
        elif "neg" in sentiment:
            st.error(f"😠 Negative")
            st.progress(float(np.max(proba)))
        else:
            st.info(f"😐 Neutral")
            st.progress(float(np.max(proba)))

        # Show confidence
        st.caption(f"Model confidence: **{np.max(proba)*100:.2f}%**")

# Footer
st.markdown("---")
st.markdown("Built with ❤️ using Streamlit and scikit-learn")

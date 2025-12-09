import streamlit as st
import joblib
import os

# Loading trained model
model_path = "spam_classifier_model.pkl"

if not os.path.exists(model_path):
    st.error("Model file 'spam_classifier_model.pkl' not found! Make sure it's uploaded to the repo.")
    st.stop()

model = joblib.load(model_path)

# page configuration
st.set_page_config(page_title="Spam Detector", page_icon="🔍", layout="centered")

# UI
st.title("Email & SMS Spam Classifier")
st.markdown("**99%+ Accuracy** • Powered by Logistic Regression + TF-IDF")

text = st.text_area("Paste any message below 👇", height=150, placeholder="e.g., Congratulations! You've won $1000...")

if st.button("Check for Spam", type="primary", use_container_width=True):
    if text.strip() == "":
        st.warning("Please enter some text!")
    else:
        with st.spinner("Analyzing..."):
            prediction = model.predict([text])[0]
            probability = model.predict_proba([text])[0].max()

        if prediction == 1:
            st.error(f"SPAM DETECTED ({probability:.1%} confidence)")
            st.snow()
        else:
            st.success(f"NOT SPAM – Looks safe ({probability:.1%} confidence)")
            st.balloons()

st.caption("Trained on 20,000+ real spam + ham emails/SMS")

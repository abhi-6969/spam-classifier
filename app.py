st.set_page_config(page_title="Spam Detector", page_icon="🔍")
st.title("🔍 Email & SMS Spam Classifier")
st.markdown(" Built with 99% Accuracy | Powered by Logistic Regression + TF-IDF")

import streamlit as st
import joblib

model = joblib.load("spam_classifier_model.pkl")

st.title("SMS & Email Spam Detector")
st.markdown("Built in <3 hours · 99%+ Accuracy")

text = st.text_area("Enter message (SMS or Email):", height=150)

if st.button("Check Spam"):
    if text.strip():
        pred = model.predict([text])[0]
        prob = model.predict_proba([text])[0].max()
        
        if pred == 1:
            st.error(f"SPAM DETECTED ({prob:.1%} confidence)")
            st.warning("Do NOT click links or reply!")
        else:
            st.success(f"NOT SPAM ({prob:.1%} confidence)")
            st.balloons()
    else:
        st.info("Please type a message")

st.set_page_config(page_title="Spam Detector", page_icon="🔍")
st.title("🔍 Email & SMS Spam Classifier")
st.markdown(" Built with 99% Accuracy | Powered by Logistic Regression + TF-IDF")



import streamlit as st
import joblib

model = joblib.load('spam_classifier_model.pkl')

st.title("📧 SMS & Email Spam Classifier")
st.write("Enter any message below to check if it's spam")

text = st.text_area("Message", height=200)

if st.button("Analyze"):
    if text.strip():
        pred = model.predict([text])[0]
        prob = model.predict_proba([text])[0].max()
        
        if pred == 1:
            st.error(f"🚨 SPAM DETECTED ({prob:.1%} confidence)")
        else:
            st.success(f"✅ NOT SPAM ({prob:.1%} confidence)")
    else:
        st.warning("Please enter some text")
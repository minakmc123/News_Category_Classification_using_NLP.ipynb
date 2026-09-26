import joblib
import streamlit as st

# ---------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------
st.set_page_config(page_title="News Category Classifier", page_icon="📰")

st.title("📰 News Category Classifier")
st.write(
    "Enter a news headline and this app will predict its category "
    "(Business, Entertainment, Politics, Sports, or Technology) using a "
    "TF-IDF + Multinomial Naive Bayes model."
)


# ---------------------------------------------------------------
# Load the trained vectorizer and model (cached so it only loads once)
# ---------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    vectorizer = joblib.load("pvectorizer.pkl")
    model = joblib.load("pmodel.pkl")
    return vectorizer, model


vectorizer, model = load_artifacts()

# ---------------------------------------------------------------
# User input
# ---------------------------------------------------------------
headline = st.text_input("News headline", placeholder="e.g. Stock market reached a record high")

if st.button("Predict Category", type="primary"):
    if not headline.strip():
        st.warning("Please enter a headline first.")
    else:
        # Transform input text using the saved TF-IDF vectorizer
        vector = vectorizer.transform([headline])
        prediction = model.predict(vector)[0]

        st.success(f"**Predicted Category:** {prediction}")

        # Show prediction probabilities if the model supports it
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(vector)[0]
            st.subheader("Prediction Confidence")
            prob_data = {
                cls: float(p) for cls, p in zip(model.classes_, probs)
            }
            st.bar_chart(prob_data)

st.divider()
st.caption(
    "Model: TfidfVectorizer + MultinomialNB, trained on a small sample "
    "news-headline dataset."
)

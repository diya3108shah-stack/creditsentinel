import streamlit as st
import pickle

st.set_page_config(
    page_title="CreditSentinel",
    page_icon="🛡️",
    layout="centered"
)

st.title("🛡️ CreditSentinel")
st.caption("NLP powered consumer credit risk classifier")
st.divider()

@st.cache_resource
def load_model():
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)
    return model, vectorizer

model, vectorizer = load_model()

st.markdown("""
This tool reads a consumer complaint and predicts which credit product category
it belongs to, based on patterns learned from hundreds of thousands of real CFPB complaints.
""")

st.divider()

st.markdown("**Try one of these example complaints or write your own:**")

examples = [
    "They charged me twice on my credit card and refused to issue a refund despite multiple calls to customer service.",
    "I never opened this account but it is showing up on my credit report and hurting my score.",
    "The debt collector kept calling me at work even after I told them to stop. They are harassing me.",
    "My mortgage payment was applied incorrectly and now they are saying I am behind on payments.",
    "I applied for a personal loan and was denied but they never told me why or sent me an adverse action notice."
]

selected = st.selectbox(
    "Pick an example or scroll down to type your own:",
    ["Choose an example..."] + examples
)

if selected != "Choose an example...":
    user_input = st.text_area(
        "Complaint text:",
        value=selected,
        height=150
    )
else:
    user_input = st.text_area(
        "Complaint text:",
        placeholder="Type or paste a consumer complaint here...",
        height=150
    )

if st.button("Classify Complaint", type="primary"):
    if user_input.strip() == "":
        st.warning("Please enter a complaint first.")
    else:
        vec = vectorizer.transform([user_input])
        prediction = model.predict(vec)[0]
        probabilities = model.predict_proba(vec)[0]
        classes = model.classes_

        st.divider()
        st.markdown("### Result")

        st.success(f"**Predicted Category:** {prediction}")

        st.markdown("**Confidence across all categories:**")
        prob_df = sorted(
            zip(classes, probabilities),
            key=lambda x: x[1],
            reverse=True
        )
        for label, prob in prob_df:
            bar_color = "🟩" if label == prediction else "⬜"
            st.write(f"{bar_color} **{label}**: {prob:.1%}")

        st.divider()
        st.caption("Model trained on 50,000 CFPB consumer complaints using TF-IDF and Logistic Regression.")

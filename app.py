"""
Fake News Detector - Streamlit App
Paste a headline/article, get:
  1. An initial ML prediction (TF-IDF + Logistic Regression / Random Forest)
  2. A live web search for the same claim (Tavily)
  3. An evidence/verification layer that reconciles the two
  4. A final 3-way verdict: VERIFIED REAL / LIKELY FAKE / UNVERIFIED
"""

import os
import streamlit as st
import joblib
import numpy as np
from lime.lime_text import LimeTextExplainer
import streamlit.components.v1 as components
from dotenv import load_dotenv

from utils import clean_text
from verify import verify_claim

# set_page_config must be the very first Streamlit command in the script.
st.set_page_config(page_title="Fake News Detector", page_icon="📰", layout="centered")

# Loads TAVILY_API_KEY from a local .env file if present (for local dev).
# On Streamlit Cloud, set it instead under Settings -> Secrets.
load_dotenv()
try:
    if "TAVILY_API_KEY" not in os.environ and "TAVILY_API_KEY" in st.secrets:
        os.environ["TAVILY_API_KEY"] = st.secrets["TAVILY_API_KEY"]
except Exception:
    pass  # no secrets.toml present (e.g. running locally with .env only)


@st.cache_resource
def load_artifacts():
    model = joblib.load("models/best_model.pkl")
    vectorizer = joblib.load("models/vectorizer.pkl")
    return model, vectorizer


def predict_proba_wrapper(texts, model, vectorizer):
    """Wrapper needed by LIME: takes raw texts, returns class probabilities."""
    cleaned = [clean_text(t) for t in texts]
    vecs = vectorizer.transform(cleaned)
    return model.predict_proba(vecs)


def render_verdict_badge(verdict_label: str):
    colors = {
        "🟢 VERIFIED REAL": "#1e7e34",
        "🔴 LIKELY FAKE": "#c0392b",
        "🟡 UNVERIFIED": "#b8860b",
    }
    color = colors.get(verdict_label, "#555")
    st.markdown(
        f"""
        <div style="padding:14px 18px; border-radius:10px; background-color:{color}22;
                    border:1px solid {color}; margin:10px 0;">
            <span style="font-size:22px; font-weight:700; color:{color};">
                {verdict_label}
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def main():
    st.title("📰 Fake News Detector")
    st.caption(
        "ML baseline (TF-IDF + Logistic Regression / Random Forest) "
        "cross-checked against live web evidence via Tavily"
    )

    try:
        model, vectorizer = load_artifacts()
    except FileNotFoundError:
        st.error(
            "No trained model found. Run `python train_model.py` first "
            "(after downloading the dataset into the `data/` folder)."
        )
        return

    text_input = st.text_area(
        "Paste a news headline or claim:",
        height=180,
        placeholder="e.g. India wins the 2026 Cricket World Cup",
    )

    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        check_clicked = st.button("🔍 Check News", type="primary", use_container_width=True)
    with col2:
        verify_clicked = st.button("🌐 Verify with Web", use_container_width=True)
    with col3:
        explain_clicked = st.button("🧠 Explain", use_container_width=True)

    if check_clicked or verify_clicked or explain_clicked:
        if not text_input.strip():
            st.warning("Please paste some text first.")
            return

        # ---------- Step 1: ML model initial prediction ----------
        cleaned = clean_text(text_input)
        vec = vectorizer.transform([cleaned])
        pred = model.predict(vec)[0]
        proba = model.predict_proba(vec)[0]

        ml_label = "REAL" if pred == 1 else "FAKE"
        ml_label_display = "🟢 Real (ML)" if pred == 1 else "🔴 Fake (ML)"
        ml_confidence = proba[pred] * 100

        st.subheader("Step 1 · ML Prediction")
        st.markdown(f"**{ml_label_display}**")
        st.metric("ML Confidence", f"{ml_confidence:.1f}%")
        st.progress(float(proba[1]))
        st.caption(f"P(Real) = {proba[1]:.3f} | P(Fake) = {proba[0]:.3f}")

        # ---------- Step 2 & 3: Web search + evidence verification ----------
        if verify_clicked or check_clicked:
            st.divider()
            st.subheader("Step 2 · Live Web Verification")
            with st.spinner("Searching the web for matching coverage..."):
                try:
                    result = verify_claim(
                        claim=text_input.strip(),
                        ml_label=ml_label,
                        ml_confidence=ml_confidence,
                    )
                except Exception as e:
                    result = None
                    st.error(f"Web verification failed: {e}")

            if result:
                if result.evidence:
                    st.markdown("**Sources found:**")
                    for e in result.evidence[:5]:
                        trust_tag = "✅ trusted" if e.is_trusted else ""
                        st.markdown(
                            f"- [{e.title}]({e.url}) — `{e.domain}` {trust_tag}\n"
                            f"  \n  <span style='color:gray;font-size:13px;'>{e.snippet}</span>",
                            unsafe_allow_html=True,
                        )
                else:
                    st.caption("No web results returned for this claim.")

                st.divider()
                st.subheader("Step 3 · Final Verdict")
                render_verdict_badge(result.verdict_label)
                st.write(result.reason)

                with st.expander("Why this verdict? (ML vs. web evidence)"):
                    st.markdown(
                        f"""
                        - **ML Prediction:** {ml_label} ({ml_confidence:.1f}% confidence)
                        - **Matched sources:** {', '.join(result.matched_sources) or 'none'}
                        - **Final verdict:** {result.verdict_label}

                        The ML model looks only at writing style and word patterns learned
                        from the training dataset, so it can be wrong — especially on
                        recent events it never saw during training. Live web evidence is
                        used here to confirm or override the ML prediction. If no article
                        is found, that alone does **not** prove the claim is fake; it's
                        marked **UNVERIFIED** instead of **FAKE**.
                        """
                    )

        # ---------- Explainability (LIME) ----------
        if explain_clicked:
            st.divider()
            st.subheader("Why did the ML model decide this?")
            with st.spinner("Generating explanation..."):
                explainer = LimeTextExplainer(class_names=["Fake", "Real"])
                exp = explainer.explain_instance(
                    text_input,
                    lambda texts: predict_proba_wrapper(texts, model, vectorizer),
                    num_features=10,
                )
                html_content = exp.as_html()
                styled_html = f"""
                <div style="background-color:white; padding:16px; border-radius:8px;
                            font-size:16px; line-height:1.6; color:black;">
                    {html_content}
                </div>
                """
                components.html(styled_html, height=800, scrolling=True)
                st.caption(
                    "🟠 Orange words pushed the prediction toward **Real**, "
                    "🔵 blue words pushed it toward **Fake**. Longer bars = stronger influence."
                )

    st.divider()
    with st.expander("About this project"):
        st.markdown(
            """
            **Pipeline:**
            1. Raw text → cleaning/lemmatization → TF-IDF (unigrams + bigrams) →
               classical ML classifier (Logistic Regression / Random Forest) → initial prediction.
            2. The same claim is searched live on the web via **Tavily**.
            3. Search results are compared against the claim to build an evidence layer.
            4. ML prediction + web evidence are reconciled into one of three final verdicts:
               🟢 **VERIFIED REAL**, 🔴 **LIKELY FAKE**, or 🟡 **UNVERIFIED**
               (no article found ≠ proof of fake).

            **Explainability:** LIME highlights which words pushed the ML model toward
            Fake or Real, so the classifier isn't a black box.

            **Dataset:** Kaggle Fake and Real News Dataset (~44k articles).
            """
        )


if __name__ == "__main__":
    main()
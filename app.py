"""
Fake News Detector - Streamlit App

Three actions, three clear jobs:
  1. Check News      -> simple ML prediction, then the final combined verdict
  2. Verify with Web  -> the actual web sources/references behind that verdict
  3. Explain           -> one combined explanation: why the ML model decided
                          what it decided, AND why the web evidence agrees
                          or disagrees with it
"""

import os
import streamlit as st
import joblib
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


# ---------------------------------------------------------------------------
# Styling
# ---------------------------------------------------------------------------

CUSTOM_CSS = """
<style>
.gt-card {
    background-color: #161b22;
    border: 1px solid #30363d;
    border-radius: 12px;
    padding: 20px 22px;
    margin: 10px 0 18px 0;
}
.gt-label-row {
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 6px;
}
.gt-dot {
    width: 12px;
    height: 12px;
    border-radius: 50%;
    display: inline-block;
}
.gt-label-text {
    font-size: 20px;
    font-weight: 700;
}
.gt-conf {
    font-size: 34px;
    font-weight: 700;
    margin-top: 6px;
}
.gt-conf-caption {
    color: #8b949e;
    font-size: 13px;
    margin-top: -4px;
}
.gt-verdict-badge {
    padding: 14px 18px;
    border-radius: 10px;
    margin: 6px 0 14px 0;
}
.gt-verdict-text {
    font-size: 22px;
    font-weight: 700;
}
.gt-verdict-reason {
    color: #c9d1d9;
    font-size: 15px;
    margin-top: 8px;
}
.gt-source-card {
    background-color: #161b22;
    border: 1px solid #30363d;
    border-radius: 10px;
    padding: 14px 16px;
    margin-bottom: 10px;
}
.gt-source-title a {
    font-size: 16px;
    font-weight: 600;
    text-decoration: none;
    color: #58a6ff;
}
.gt-source-meta {
    font-size: 12.5px;
    color: #8b949e;
    margin: 4px 0 8px 0;
}
.gt-trusted-tag {
    background-color: #1e7e3422;
    color: #3fb950;
    border: 1px solid #3fb95055;
    border-radius: 6px;
    padding: 1px 8px;
    font-size: 11.5px;
    font-weight: 600;
    margin-left: 6px;
}
.gt-source-snippet {
    color: #adbac7;
    font-size: 14px;
    line-height: 1.5;
}
.gt-explain-block {
    background-color: #161b22;
    border: 1px solid #30363d;
    border-radius: 12px;
    padding: 18px 20px;
    margin: 10px 0 18px 0;
}
.gt-explain-block h4 {
    margin-top: 0;
    margin-bottom: 10px;
}
</style>
"""

VERDICT_COLORS = {
    "🟢 VERIFIED REAL": "#1e7e34",
    "🔴 LIKELY FAKE": "#c0392b",
    "🟡 UNVERIFIED": "#b8860b",
}

ML_COLORS = {"REAL": "#3fb950", "FAKE": "#f85149"}


# ---------------------------------------------------------------------------
# Model loading + core prediction
# ---------------------------------------------------------------------------

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


def run_ml_prediction(text, model, vectorizer):
    cleaned = clean_text(text)
    vec = vectorizer.transform([cleaned])
    pred = model.predict(vec)[0]
    proba = model.predict_proba(vec)[0]
    label = "REAL" if pred == 1 else "FAKE"
    return {
        "label": label,
        "confidence": float(proba[pred] * 100),
        "p_real": float(proba[1]),
        "p_fake": float(proba[0]),
    }


def get_verification(text, ml_label, ml_confidence):
    """Runs (and caches, per input text) the web verification step so the
    three buttons can share one result instead of re-searching every time."""
    cache_key = f"verify::{text}"
    if cache_key in st.session_state:
        return st.session_state[cache_key]

    with st.spinner("Searching the web for matching coverage..."):
        try:
            result = verify_claim(claim=text.strip(), ml_label=ml_label, ml_confidence=ml_confidence)
        except Exception as e:
            st.error(f"Web verification failed: {e}")
            result = None

    st.session_state[cache_key] = result
    return result


# ---------------------------------------------------------------------------
# Render helpers
# ---------------------------------------------------------------------------

def render_ml_card(ml):
    color = ML_COLORS[ml["label"]]
    st.markdown(
        f"""
        <div class="gt-card">
            <div class="gt-label-row">
                <span class="gt-dot" style="background-color:{color};"></span>
                <span class="gt-label-text" style="color:{color};">{ml["label"]}</span>
            </div>
            <div class="gt-conf">{ml["confidence"]:.1f}%</div>
            <div class="gt-conf-caption">ML model confidence</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_verdict_badge(verdict_label, reason=None):
    color = VERDICT_COLORS.get(verdict_label, "#555")
    reason_html = f'<div class="gt-verdict-reason">{reason}</div>' if reason else ""
    st.markdown(
        f"""
        <div class="gt-verdict-badge" style="background-color:{color}22; border:1px solid {color};">
            <span class="gt-verdict-text" style="color:{color};">{verdict_label}</span>
            {reason_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_source_cards(evidence):
    if not evidence:
        st.caption("No web results were returned for this claim.")
        return

    for e in evidence[:5]:
        trust_tag = '<span class="gt-trusted-tag">✅ Trusted source</span>' if e.is_trusted else ""
        st.markdown(
            f"""
            <div class="gt-source-card">
                <div class="gt-source-title"><a href="{e.url}" target="_blank">{e.title or e.url}</a></div>
                <div class="gt-source-meta">{e.domain}{trust_tag}</div>
                <div class="gt-source-snippet">{e.snippet}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_ml_explanation(text_input, model, vectorizer):
    st.markdown("#### 🧠 Why the ML model decided this")
    with st.spinner("Generating ML explanation..."):
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
        components.html(styled_html, height=700, scrolling=True)
        st.caption(
            "🟠 Orange words pushed the prediction toward **Real**, "
            "🔵 blue words pushed it toward **Fake**. Longer bars = stronger influence."
        )


def render_web_explanation(ml, result):
    st.markdown("#### 🌐 Why the web evidence agrees or disagrees")
    matched = ", ".join(result.matched_sources) if result.matched_sources else "none found"
    st.markdown(
        f"""
        <div class="gt-explain-block">
            <p><b>ML said:</b> {ml["label"]} ({ml["confidence"]:.1f}% confidence)</p>
            <p><b>Matched sources:</b> {matched}</p>
            <p><b>Final verdict:</b> {result.verdict_label}</p>
            <p style="color:#adbac7;">{result.reason}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.caption(
        "The ML model only looks at writing style and word patterns learned from "
        "training data, so it can be confidently wrong — especially on recent events "
        "it never saw. Live web evidence is used to confirm or override it. No article "
        "found does **not** prove a claim is fake, so that case is marked **UNVERIFIED** "
        "rather than **FAKE**."
    )


# ---------------------------------------------------------------------------
# Main app
# ---------------------------------------------------------------------------

def main():
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
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

        ml = run_ml_prediction(text_input, model, vectorizer)

        # ---------------- Action 1: Check News ----------------
        # Simple ML result, then the final combined verdict. No raw source
        # list here — that's what "Verify with Web" is for.
        if check_clicked:
            st.subheader("Step 1 · ML Prediction")
            render_ml_card(ml)

            result = get_verification(text_input, ml["label"], ml["confidence"])
            if result:
                st.subheader("Step 2 · Final Verdict")
                render_verdict_badge(result.verdict_label, result.reason)
                st.caption("Tap **Verify with Web** to see the sources behind this verdict.")

        # ---------------- Action 2: Verify with Web ----------------
        # The actual references: real links, titles, and snippets.
        elif verify_clicked:
            st.subheader("Step 1 · ML Prediction")
            render_ml_card(ml)

            result = get_verification(text_input, ml["label"], ml["confidence"])
            if result:
                st.subheader("Step 2 · Web Sources")
                render_source_cards(result.evidence)

                st.subheader("Step 3 · Final Verdict")
                render_verdict_badge(result.verdict_label, result.reason)

        # ---------------- Action 3: Explain ----------------
        # One combined explanation: ML reasoning + web-evidence reasoning.
        elif explain_clicked:
            st.subheader("Step 1 · ML Prediction")
            render_ml_card(ml)

            result = get_verification(text_input, ml["label"], ml["confidence"])

            st.subheader("Explanation")
            render_ml_explanation(text_input, model, vectorizer)
            st.divider()
            if result:
                render_web_explanation(ml, result)

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

            **Explainability:** combines LIME word-importance for the ML model with
            the web-evidence reasoning behind the final verdict, so neither layer is
            a black box.

            **Dataset:** Kaggle Fake and Real News Dataset (~44k articles).
            """
        )


if __name__ == "__main__":
    main()

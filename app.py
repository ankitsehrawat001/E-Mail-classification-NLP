import re
from pathlib import Path

import joblib
import streamlit as st


MODEL_PATH = Path(__file__).resolve().with_name("sms_spam_logistic_regression.joblib")


def clean_text(text: str) -> str:
    """Apply the same preprocessing used when the model was trained."""
    text = text.lower()
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"\b[\w.+-]+@[\w.-]+\.[a-zA-Z]{2,}\b", " ", text)
    text = re.sub(r"<[^>]+>", " ", text)
    return re.sub(r"\s+", " ", text).strip()


@st.cache_resource
def load_artifact():
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"Model file was not found: {MODEL_PATH}. "
            "Place it beside app.py and restart the app."
        )
    return joblib.load(MODEL_PATH)


st.set_page_config(
    page_title="InboxShield | SMS Spam Check",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@400;500;600;700&display=swap');

    :root {
        --ink: #f2f6ff;
        --muted: #9aa9c4;
        --mint: #75f0c2;
        --violet: #9a8bff;
        --panel: rgba(20, 30, 55, .76);
        --line: rgba(174, 193, 232, .13);
    }
    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
    }
    .stApp {
        color: var(--ink);
        background:
            radial-gradient(ellipse at 10% 0%, rgba(78, 87, 184, .22), transparent 34%),
            radial-gradient(ellipse at 92% 24%, rgba(52, 180, 156, .13), transparent 30%),
            linear-gradient(135deg, #080d1a 0%, #0d1427 52%, #10172a 100%);
        background-size: 180% 180%;
        animation: atmosphere 18s ease-in-out infinite alternate;
    }
    @keyframes atmosphere {
        from { background-position: 0% 0%; }
        to { background-position: 100% 70%; }
    }
    .block-container {
        max-width: 1180px;
        padding: 2.2rem 2rem 4rem;
    }
    [data-testid="stHeader"] { background: transparent; }
    [data-testid="stToolbar"], #MainMenu, footer { visibility: hidden; }
    .brand-row {
        display: flex; align-items: center; gap: 11px;
        color: #c8d3e8; font-size: .9rem; font-weight: 700;
        letter-spacing: .09em; text-transform: uppercase;
        margin: .2rem 0 2.8rem;
    }
    .brand-icon {
        display: grid; place-items: center; width: 38px; height: 38px;
        border-radius: 13px; background: linear-gradient(145deg, #75f0c2, #79a8ff);
        color: #10172a; font-size: 1.15rem;
        box-shadow: 0 8px 30px rgba(117, 240, 194, .2);
    }
    .hero-copy { padding: 1.4rem 0 1.5rem; }
    .eyebrow {
        color: var(--mint); font-size: .78rem; font-weight: 700;
        letter-spacing: .18em; text-transform: uppercase;
    }
    .hero-title {
        color: var(--ink); font-family: 'Space Grotesk', sans-serif;
        font-size: clamp(2.7rem, 5vw, 4.5rem); font-weight: 700;
        letter-spacing: -.065em; line-height: 1.04; margin: .8rem 0 1rem;
    }
    .hero-title span {
        color: transparent;
        background: linear-gradient(90deg, #75f0c2, #9aabff 75%);
        -webkit-background-clip: text; background-clip: text;
    }
    .hero-subtitle {
        color: var(--muted); font-size: 1.08rem; line-height: 1.75;
        max-width: 550px;
    }
    .hero-art {
        min-height: 258px; height: 100%; position: relative; overflow: hidden;
        display: grid; place-items: center; border-radius: 30px;
        border: 1px solid rgba(177, 197, 239, .14);
        background:
            radial-gradient(circle at 50% 49%, rgba(117, 240, 194, .16), transparent 38%),
            linear-gradient(145deg, rgba(37, 49, 82, .84), rgba(17, 26, 47, .75));
        box-shadow: inset 0 1px rgba(255,255,255,.05), 0 24px 70px rgba(0,0,0,.18);
    }
    .hero-art:before, .hero-art:after {
        content: ''; position: absolute; border: 1px solid rgba(117,240,194,.18);
        border-radius: 50%; width: 190px; height: 190px;
        animation: orbit 16s linear infinite;
    }
    .hero-art:after {
        width: 245px; height: 245px; border-color: rgba(154,139,255,.18);
        animation-direction: reverse; animation-duration: 22s;
    }
    @keyframes orbit { to { transform: rotate(360deg); } }
    .shield-orb {
        z-index: 1; display: grid; place-items: center; width: 132px; height: 132px;
        border-radius: 38px; transform: rotate(-8deg);
        background: linear-gradient(145deg, rgba(117,240,194,.2), rgba(113,142,255,.2));
        border: 1px solid rgba(196, 255, 236, .35);
        box-shadow: 0 0 55px rgba(117,240,194,.22), inset 0 1px rgba(255,255,255,.18);
        animation: float 4s ease-in-out infinite;
    }
    .shield-orb span { font-size: 4.25rem; transform: rotate(8deg); }
    .float-chip {
        position: absolute; z-index: 2; padding: 9px 13px; border-radius: 13px;
        color: #dce7f7; font-size: .78rem; font-weight: 600;
        background: rgba(11, 19, 37, .78); border: 1px solid var(--line);
        box-shadow: 0 12px 30px rgba(0,0,0,.22);
        animation: float 5s ease-in-out infinite;
    }
    .chip-left { left: 9%; top: 24%; }
    .chip-right { right: 8%; bottom: 23%; animation-delay: -2s; }
    @keyframes float { 0%,100% { transform: translateY(0); } 50% { transform: translateY(-9px); } }
    .section-heading {
        font-family: 'Space Grotesk', sans-serif; font-size: 1.45rem;
        font-weight: 600; letter-spacing: -.025em; margin: 2.7rem 0 .35rem;
    }
    .section-note { color: var(--muted); margin-bottom: 1.1rem; }
    .input-panel {
        padding: 1.35rem 1.45rem .6rem; border-radius: 22px;
        background: var(--panel); border: 1px solid var(--line);
        box-shadow: 0 20px 55px rgba(0,0,0,.14);
    }
    [data-testid="stTextArea"] label {
        color: #e8efff !important; font-size: .95rem; font-weight: 600;
    }
    [data-testid="stTextArea"] textarea {
        color: #f2f6ff !important; background: rgba(8, 14, 29, .78) !important;
        border: 1px solid rgba(174,193,232,.16) !important;
        border-radius: 15px !important; line-height: 1.6 !important;
        transition: border-color .2s ease, box-shadow .2s ease;
    }
    [data-testid="stTextArea"] textarea:focus {
        border-color: rgba(117,240,194,.7) !important;
        box-shadow: 0 0 0 3px rgba(117,240,194,.1) !important;
    }
    [data-testid="stTextArea"] textarea::placeholder { color: #74839f !important; }
    .stButton > button {
        min-height: 3.15rem; border: 0; border-radius: 13px;
        color: #07151b !important; font-size: 1rem; font-weight: 700;
        background: linear-gradient(100deg, #75f0c2, #8acbff) !important;
        box-shadow: 0 10px 28px rgba(117,240,194,.16);
        transition: transform .18s ease, box-shadow .18s ease, filter .18s ease;
    }
    .stButton > button:hover {
        transform: translateY(-2px); filter: brightness(1.06);
        box-shadow: 0 14px 36px rgba(117,240,194,.25);
    }
    .stButton > button:active { transform: translateY(0) scale(.99); }
    .result-card {
        overflow: hidden; position: relative; margin-top: 1.6rem; padding: 1.8rem;
        border-radius: 22px; background: var(--panel); border: 1px solid var(--line);
        animation: appear .55s cubic-bezier(.2,.8,.2,1) both;
    }
    .result-card:before {
        content: ''; position: absolute; width: 220px; height: 220px;
        right: -65px; top: -115px; border-radius: 50%;
        background: var(--glow); filter: blur(45px); opacity: .24;
    }
    @keyframes appear {
        from { opacity: 0; transform: translateY(14px) scale(.99); }
        to { opacity: 1; transform: translateY(0) scale(1); }
    }
    .result-label {
        display: inline-flex; align-items: center; gap: 8px; padding: 7px 12px;
        border-radius: 999px; color: var(--accent); background: var(--pill);
        font-size: .78rem; font-weight: 700; letter-spacing: .08em; text-transform: uppercase;
    }
    .result-title {
        position: relative; font-family: 'Space Grotesk', sans-serif;
        color: #f4f7ff; font-size: 2rem; font-weight: 700;
        letter-spacing: -.045em; margin: 1rem 0 .35rem;
    }
    .result-copy { position: relative; color: var(--muted); }
    .probability-box {
        position: relative; padding: 1.2rem 1.3rem; border-radius: 17px;
        background: rgba(7, 13, 27, .48); border: 1px solid rgba(174,193,232,.1);
    }
    .probability-top {
        display: flex; align-items: baseline; justify-content: space-between;
        color: #b4c1d9; font-size: .88rem; margin-bottom: 13px;
    }
    .probability-value {
        color: #f2f6ff; font-family: 'Space Grotesk', sans-serif;
        font-size: 1.65rem; font-weight: 700;
    }
    .probability-track {
        height: 9px; overflow: hidden; border-radius: 99px;
        background: rgba(174,193,232,.12);
    }
    .probability-fill {
        height: 100%; border-radius: inherit; width: var(--value);
        background: linear-gradient(90deg, var(--accent), #a2b5ff);
        box-shadow: 0 0 18px var(--accent); animation: fill .9s ease-out both;
    }
    @keyframes fill { from { width: 0; } to { width: var(--value); } }
    .trust-note {
        color: #8898b4; font-size: .82rem; line-height: 1.6;
        padding: 1rem 0 .2rem;
    }
    .footer-note {
        border-top: 1px solid var(--line); color: #75839d; font-size: .8rem;
        margin-top: 3.5rem; padding-top: 1.2rem; text-align: center;
    }
    @media (max-width: 760px) {
        .block-container { padding: 1.4rem 1rem 3rem; }
        .brand-row { margin-bottom: 1.4rem; }
        .hero-copy { padding: .4rem 0 1rem; }
        .hero-art { min-height: 210px; }
    }
    @media (prefers-reduced-motion: reduce) {
        *, *::before, *::after { animation-duration: .01ms !important; animation-iteration-count: 1 !important; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="brand-row">
        <span class="brand-icon">✉</span>
        <span>InboxShield</span>
    </div>
    """,
    unsafe_allow_html=True,
)

hero_copy, hero_art = st.columns([1.12, 0.88], gap="large", vertical_alignment="center")
with hero_copy:
    st.markdown(
        """
        <div class="hero-copy">
            <div class="eyebrow">Smart message protection</div>
            <div class="hero-title">Spot the spam.<br><span>Keep your inbox safe.</span></div>
            <div class="hero-subtitle">
                Paste any SMS and let our machine-learning model check whether
                it looks like spam—in just a moment.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with hero_art:
    st.markdown(
        """
        <div class="hero-art">
            <div class="float-chip chip-left">✦ Smart detection</div>
            <div class="shield-orb"><span>🛡️</span></div>
            <div class="float-chip chip-right">✉ Inbox protected</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown('<div class="section-heading">Check a message</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="section-note">Your message is analyzed locally by the saved classification model.</div>',
    unsafe_allow_html=True,
)

with st.container(border=True):
    message = st.text_area(
        "Paste your SMS below",
        placeholder="Example: Congratulations! You have won a free prize. Click here to claim...",
        height=170,
        max_chars=5000,
        label_visibility="visible",
    )
    submit = st.button("✦  Analyze message", type="primary", use_container_width=True)

if submit:
    cleaned_message = clean_text(message)
    if not cleaned_message:
        st.warning("Please enter a message with some text before analyzing.")
    else:
        artifact = load_artifact()
        model = artifact["pipeline"]
        label_mapping = artifact["label_mapping"]

        prediction = int(model.predict([cleaned_message])[0])
        probabilities = model.predict_proba([cleaned_message])[0]
        classifier_classes = list(model.named_steps["classifier"].classes_)
        spam_index = classifier_classes.index(1)
        spam_probability = float(probabilities[spam_index])
        predicted_label = label_mapping[prediction]
        percentage = spam_probability * 100

        if predicted_label == "Spam":
            accent, pill, icon = "#ff8b9d", "rgba(255, 107, 130, .12)", "⚠"
            headline = "This message looks like spam"
            explanation = "Be careful with links, requests for personal details, or unexpected offers."
        else:
            accent, pill, icon = "#75f0c2", "rgba(117, 240, 194, .11)", "✓"
            headline = "This message looks safe"
            explanation = "The model did not find strong signs of spam in this message."

        result_left, result_right = st.columns([1, 1], gap="large", vertical_alignment="center")
        with result_left:
            st.markdown(
                f"""
                <div class="result-card" style="--glow:{accent}">
                    <div class="result-label" style="--accent:{accent};--pill:{pill}">
                        <span>{icon}</span> Analysis complete
                    </div>
                    <div class="result-title">{headline}</div>
                    <div class="result-copy">{explanation}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with result_right:
            st.markdown(
                f"""
                <div class="result-card" style="--glow:{accent}">
                    <div class="probability-box" style="--accent:{accent};--value:{percentage:.2f}%">
                        <div class="probability-top">
                            <span>Spam probability</span>
                            <span class="probability-value">{percentage:.2f}%</span>
                        </div>
                        <div class="probability-track">
                            <div class="probability-fill"></div>
                        </div>
                    </div>
                    <div class="trust-note">
                        This score is the model's estimate, not a guarantee.
                        Avoid opening suspicious links even if a message appears safe.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

st.markdown(
    """
    <div class="footer-note">
        InboxShield · Message classification powered by your trained Logistic Regression model
    </div>
    """,
    unsafe_allow_html=True,
)
import streamlit as st
import joblib
import pandas as pd
import os
import time

st.set_page_config(page_title="CardioPredict", page_icon="🫀",
                   layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
<style>
[data-testid="stApp"]{background:#f0f4f8!important}
.block-container{padding:2rem 3rem!important;max-width:1200px!important}
header[data-testid="stHeader"]{background:#f0f4f8!important}
h1,h2,h3{color:#0a2540!important;font-weight:700!important}
div[data-testid="stButton"] > button{
    background:#fff!important;color:#0077b6!important;
    border:1.5px solid #0077b6!important;border-radius:8px!important;
    font-size:15px!important;font-weight:500!important;}
div[data-testid="stButton"] > button:hover{background:#e8f4fd!important}
div[data-testid="stFormSubmitButton"] > button{
    background:linear-gradient(135deg,#0077b6,#023e8a)!important;
    color:#fff!important;border:none!important;border-radius:8px!important;
    font-size:16px!important;font-weight:600!important;padding:14px!important;}
div[data-testid="stFormSubmitButton"] > button:hover{
    background:linear-gradient(135deg,#0096c7,#0077b6)!important;
    box-shadow:0 4px 16px rgba(0,119,182,0.35)!important;}
div[data-testid="stNumberInput"] input,
div[data-testid="stSelectbox"] > div > div{
    background:#fff!important;border:1.5px solid #b0c4d8!important;
    border-radius:6px!important;color:#0a2540!important;font-size:15px!important;}
label[data-testid="stWidgetLabel"] p{color:#2d4a62!important;font-size:14px!important;font-weight:500!important}
div[data-testid="stSelectbox"] svg{color:#0077b6!important}
hr{border-color:#cdd9e5!important}
[data-testid="stForm"]{border:none!important;background:transparent!important;padding:0!important}
code{background:#e8f4fd!important;color:#0077b6!important;border-radius:4px!important}
div[data-testid="stAlert"]{border-radius:8px!important;font-size:14px!important}
.tw-line{color:#0077b6;font-weight:500;font-size:15px;letter-spacing:0.02em;min-height:24px;display:block;margin-bottom:0.5rem}
.ferr div[data-testid="stNumberInput"] input{border-color:#e74c3c!important;box-shadow:0 0 0 2px rgba(231,76,60,0.18)!important}
.ferr div[data-testid="stSelectbox"] > div > div{border-color:#e74c3c!important;box-shadow:0 0 0 2px rgba(231,76,60,0.18)!important}
.fok div[data-testid="stNumberInput"] input{border-color:#2ecc71!important;box-shadow:0 0 0 2px rgba(46,204,113,0.18)!important}
.fok div[data-testid="stSelectbox"] > div > div{border-color:#2ecc71!important;box-shadow:0 0 0 2px rgba(46,204,113,0.18)!important}
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    base = os.path.dirname(__file__)
    model  = joblib.load(os.path.join(base, "models", "heart_disease_model.pkl"))
    scaler = joblib.load(os.path.join(base, "models", "scaler.pkl"))
    return model, scaler

model, scaler = load_model()

# ── Session state ─────────────────────────────────────────────────────────────
defaults = {
    "page": "home", "role": None, "prediction": None,
    "show_errors": False, "field_errors": {},
    "tw_idx": 0, "tw_char": 0, "tw_deleting": False, "tw_pause": 0
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

PHRASES = ["Intelligent Heart Disease Risk Assessment", "Built By: Salman Sarwar"]

def tw_step():
    """One step of typewriter. Returns (text, delay_seconds)."""
    pi  = st.session_state.tw_idx
    ci  = st.session_state.tw_char
    dl  = st.session_state.tw_deleting
    pau = st.session_state.tw_pause
    p   = PHRASES[pi]

    if pau > 0:
        st.session_state.tw_pause -= 1
        return p if not dl else p[:ci], 0.08

    if not dl:
        new_ci = ci + 1
        st.session_state.tw_char = new_ci
        if new_ci == len(p):
            st.session_state.tw_deleting = True
            st.session_state.tw_pause = 50   # pause ~4s at 0.08s per tick
        return p[:new_ci], 0.06
    else:
        new_ci = ci - 1
        st.session_state.tw_char = new_ci
        if new_ci == 0:
            st.session_state.tw_deleting = False
            st.session_state.tw_idx = (pi + 1) % len(PHRASES)
            st.session_state.tw_pause = 5
        return p[:new_ci], 0.04


def field_wrap(key, is_err_map):
    """Return CSS class based on validation state."""
    if not st.session_state.show_errors:
        return ""
    is_err = is_err_map.get(key, False)
    return "ferr" if is_err else "fok"


# ══════════════════════════════════════════════════════════════════════════════
# HOME PAGE
# ══════════════════════════════════════════════════════════════════════════════
def home_page():
    st.markdown("# CardioPredict")

    tw_slot = st.empty()
    txt, delay = tw_step()
    tw_slot.markdown(f'<span class="tw-line">{txt}▌</span>', unsafe_allow_html=True)

    st.divider()
    st.subheader("Select your role to continue")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🧑 Patient", use_container_width=True):
            st.session_state.role = "patient"
            st.session_state.page = "form"
            st.session_state.prediction = None
            st.session_state.show_errors = False
            st.rerun()
        st.caption("View form with doctor guidance")
    with col2:
        if st.button("🩺 Doctor", use_container_width=True):
            st.session_state.role = "doctor"
            st.session_state.page = "form"
            st.session_state.prediction = None
            st.session_state.show_errors = False
            st.rerun()
        st.caption("Full clinical access & prediction")

    time.sleep(delay)
    st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# FORM PAGE
# ══════════════════════════════════════════════════════════════════════════════
def form_page():
    role = st.session_state.role
    is_patient = role == "patient"
    fe = st.session_state.field_errors

    # ── Header ────────────────────────────────────────────────────────────────
    col_title, col_role, col_back = st.columns([7, 1.5, 0.8])
    with col_title:
        st.markdown("## CardioPredict")
        tw_slot = st.empty()
        txt, delay = tw_step()
        tw_slot.markdown(f'<span class="tw-line" style="font-size:13px">{txt}▌</span>',
                         unsafe_allow_html=True)
    with col_role:
        st.markdown(
            f"<div style='padding-top:0.8rem'><b>Role:</b> <code>{'Patient' if is_patient else 'Doctor'}</code></div>",
            unsafe_allow_html=True)
    with col_back:
        st.markdown("<div style='padding-top:0.5rem'>", unsafe_allow_html=True)
        if st.button("← Back", use_container_width=True):
            st.session_state.page = "home"
            st.session_state.role = None
            st.session_state.prediction = None
            st.session_state.show_errors = False
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

    st.divider()

    if is_patient:
        st.warning("This form requires clinical test results. Please visit your doctor who will conduct the necessary tests and fill this form on your behalf. Misuse of this tool may be harmful.")
        st.info("Form is view-only. Please have your doctor fill this during your consultation.")
    else:
        st.info("Enter all patient clinical test results below for accurate prediction.")

    # ── Patient Information ───────────────────────────────────────────────────
    st.subheader("Patient Information")
    c1, c2, c3 = st.columns(3)

    with c1:
        cls = field_wrap("age", fe)
        st.markdown(f'<div class="{cls}">', unsafe_allow_html=True)
        age = st.number_input("Age (years)", min_value=1, max_value=120,
                              value=None, placeholder="1 – 120", disabled=is_patient)
        st.markdown('</div>', unsafe_allow_html=True)

    with c2:
        cls = field_wrap("sex", fe)
        st.markdown(f'<div class="{cls}">', unsafe_allow_html=True)
        sex = st.selectbox("Sex", ["Select","Male","Female"], disabled=is_patient)
        st.markdown('</div>', unsafe_allow_html=True)

    with c3:
        cls = field_wrap("cp", fe)
        st.markdown(f'<div class="{cls}">', unsafe_allow_html=True)
        cp = st.selectbox("Chest pain type",
                          ["Select","Typical angina","Atypical angina","Non-anginal pain","Asymptomatic"],
                          disabled=is_patient)
        st.markdown('</div>', unsafe_allow_html=True)

    c4, c5, c6 = st.columns(3)
    with c4:
        cls = field_wrap("trestbps", fe)
        st.markdown(f'<div class="{cls}">', unsafe_allow_html=True)
        trestbps = st.number_input("Resting blood pressure (mmHg)", min_value=40, max_value=300,
                                   value=None, placeholder="40 – 300", disabled=is_patient)
        st.markdown('</div>', unsafe_allow_html=True)

    with c5:
        cls = field_wrap("chol", fe)
        st.markdown(f'<div class="{cls}">', unsafe_allow_html=True)
        chol = st.number_input("Cholesterol (mg/dL)", min_value=20, max_value=1000,
                               value=None, placeholder="20 – 1000", disabled=is_patient)
        st.markdown('</div>', unsafe_allow_html=True)

    with c6:
        cls = field_wrap("fbs", fe)
        st.markdown(f'<div class="{cls}">', unsafe_allow_html=True)
        fbs = st.selectbox("Fasting blood sugar",
                           ["Select","Below 120 mg/dL","Above 120 mg/dL"], disabled=is_patient)
        st.markdown('</div>', unsafe_allow_html=True)

    # ── Clinical Test Results ─────────────────────────────────────────────────
    st.subheader("Clinical Test Results")
    c7, c8, c9 = st.columns(3)

    with c7:
        cls = field_wrap("restecg", fe)
        st.markdown(f'<div class="{cls}">', unsafe_allow_html=True)
        restecg = st.selectbox("Resting ECG result",
                               ["Select","Normal","ST-T wave abnormality","Left ventricular hypertrophy"],
                               disabled=is_patient)
        st.markdown('</div>', unsafe_allow_html=True)

    with c8:
        cls = field_wrap("thalach", fe)
        st.markdown(f'<div class="{cls}">', unsafe_allow_html=True)
        thalach = st.number_input("Maximum heart rate (bpm)", min_value=50, max_value=250,
                                  value=None, placeholder="50 – 250", disabled=is_patient)
        st.markdown('</div>', unsafe_allow_html=True)

    with c9:
        cls = field_wrap("exang", fe)
        st.markdown(f'<div class="{cls}">', unsafe_allow_html=True)
        exang = st.selectbox("Exercise induced chest pain",
                             ["Select","No","Yes"], disabled=is_patient)
        st.markdown('</div>', unsafe_allow_html=True)

    c10, c11, c12 = st.columns(3)
    with c10:
        cls = field_wrap("oldpeak", fe)
        st.markdown(f'<div class="{cls}">', unsafe_allow_html=True)
        oldpeak = st.number_input("ST depression (mm)", min_value=0.0, max_value=10.0,
                                  value=None, placeholder="0.0 – 10.0",
                                  step=0.1, format="%.1f", disabled=is_patient)
        st.markdown('</div>', unsafe_allow_html=True)

    with c11:
        cls = field_wrap("slope", fe)
        st.markdown(f'<div class="{cls}">', unsafe_allow_html=True)
        slope = st.selectbox("Slope of ST segment",
                             ["Select","Upsloping","Flat","Downsloping"], disabled=is_patient)
        st.markdown('</div>', unsafe_allow_html=True)

    with c12:
        cls = field_wrap("ca", fe)
        st.markdown(f'<div class="{cls}">', unsafe_allow_html=True)
        ca = st.selectbox("Major blood vessels",
                          ["Select","No vessels affected","One vessel affected",
                           "Two vessels affected","Three vessels affected"], disabled=is_patient)
        st.markdown('</div>', unsafe_allow_html=True)

    c13, _, __ = st.columns(3)
    with c13:
        cls = field_wrap("thal", fe)
        st.markdown(f'<div class="{cls}">', unsafe_allow_html=True)
        thal = st.selectbox("Thalassemia type",
                            ["Select","Normal","Fixed defect","Reversible defect","Unknown"],
                            disabled=is_patient)
        st.markdown('</div>', unsafe_allow_html=True)

    st.divider()

    # ── Predict ───────────────────────────────────────────────────────────────
    if not is_patient:
        if st.button("⚡ Analyse Heart Disease Risk", use_container_width=True, type="primary"):

            field_errors = {
                "age":      age is None,
                "sex":      sex == "Select",
                "cp":       cp == "Select",
                "trestbps": trestbps is None,
                "chol":     chol is None,
                "fbs":      fbs == "Select",
                "restecg":  restecg == "Select",
                "thalach":  thalach is None,
                "exang":    exang == "Select",
                "oldpeak":  oldpeak is None,
                "slope":    slope == "Select",
                "ca":       ca == "Select",
                "thal":     thal == "Select",
            }
            st.session_state.field_errors = field_errors
            st.session_state.show_errors  = True

            if any(field_errors.values()):
                st.session_state.prediction = None
                st.rerun()
            else:
                range_err = None
                if not (1 <= age <= 120):          range_err = "Age must be between 1 and 120."
                elif not (40 <= trestbps <= 300):  range_err = "Resting BP must be between 40 – 300 mmHg."
                elif not (20 <= chol <= 1000):     range_err = "Cholesterol must be between 20 – 1000 mg/dL."
                elif not (50 <= thalach <= 250):   range_err = "Max heart rate must be between 50 – 250 bpm."
                elif not (0.0 <= oldpeak <= 10.0): range_err = "ST depression must be between 0.0 – 10.0 mm."

                if range_err:
                    st.session_state.prediction = None
                    st.error(range_err)
                else:
                    cp_map      = {"Typical angina":0,"Atypical angina":1,"Non-anginal pain":2,"Asymptomatic":3}
                    fbs_map     = {"Below 120 mg/dL":0,"Above 120 mg/dL":1}
                    restecg_map = {"Normal":0,"ST-T wave abnormality":1,"Left ventricular hypertrophy":2}
                    exang_map   = {"No":0,"Yes":1}
                    slope_map   = {"Upsloping":0,"Flat":1,"Downsloping":2}
                    ca_map      = {"No vessels affected":0,"One vessel affected":1,
                                   "Two vessels affected":2,"Three vessels affected":3}
                    thal_map    = {"Normal":0,"Fixed defect":1,"Reversible defect":2,"Unknown":3}

                    input_df = pd.DataFrame([{
                        'age':float(age),'sex':float({"Male":1,"Female":0}[sex]),
                        'cp':float(cp_map[cp]),'trestbps':float(trestbps),'chol':float(chol),
                        'fbs':float(fbs_map[fbs]),'restecg':float(restecg_map[restecg]),
                        'thalach':float(thalach),'exang':float(exang_map[exang]),
                        'oldpeak':float(oldpeak),'slope':float(slope_map[slope]),
                        'ca':float(ca_map[ca]),'thal':float(thal_map[thal])
                    }])
                    cols_to_scale = ['age','trestbps','chol','thalach','oldpeak']
                    input_df[cols_to_scale] = scaler.transform(input_df[cols_to_scale])
                    st.session_state.prediction  = int(model.predict(input_df)[0])
                    st.session_state.show_errors = False
                    st.rerun()

        # Error message below button
        if st.session_state.show_errors and any(st.session_state.field_errors.values()):
            st.error("Please fill all required fields before analysing.")

    # ── Result ────────────────────────────────────────────────────────────────
    if st.session_state.prediction is not None and not is_patient:
        note = ("This is an AI predicted result. Doctor must apply their own "
                "clinical experience and judgment before making any medical decision.")
        if st.session_state.prediction == 0:
            st.success("✅  No Heart Disease Detected")
        else:
            st.error("⚠️  Heart Disease Risk Detected")
        st.caption(note)

    # Typewriter tick
    time.sleep(delay)
    st.rerun()


# ── Router ────────────────────────────────────────────────────────────────────
if st.session_state.page == "home":
    home_page()
else:
    form_page()

# streamlit run app.py
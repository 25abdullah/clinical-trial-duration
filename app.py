import numpy as np
import streamlit as st

MODELS = ["from-scratch NN", "keras + log(enroll)", "linear + log + age²"]


@st.cache_data
def load():
    d = np.load("demo_data.npz", allow_pickle=True)
    return (d["y_test"], d["bounds"], d["meta"], d["meta_cols"],
            [d["pred_scratch"], d["pred_keras"], d["pred_linear"]])


y_test, bounds, meta, meta_cols, preds = load()

st.title("Clinical trial duration")
st.write(
    "Predicting how long a trial will take, using only what is known at "
    "registration. Each model gives a point estimate and an 80% interval "
    "built from its own test-set errors."
)
st.write(
    "The best model explains about 42% of the variance. The rest depends on "
    "things registration data never records: how quickly sites recruit, "
    "whether the protocol gets amended, whether another trial is competing "
    "for the same patients. That is why the interval matters more than the "
    "point estimate."
)

if "i" not in st.session_state:
    st.session_state.i = int(np.random.randint(len(y_test)))

if st.button("Show another trial"):
    st.session_state.i = int(np.random.randint(len(y_test)))

i = st.session_state.i
actual = float(np.exp(y_test[i]))

st.subheader("Trial")
st.table({c: [v] for c, v in zip(meta_cols, meta[i])})

st.subheader("Predictions")
for name, p, (lo, hi) in zip(MODELS, preds, bounds):
    point = np.exp(p[i])
    inside = np.exp(p[i] + lo) <= actual <= np.exp(p[i] + hi)
    st.write(
        f"**{name}** — {point:,.0f} days "
        f"({np.exp(p[i] + lo):,.0f} to {np.exp(p[i] + hi):,.0f})"
        f"{'  ✓' if inside else '  ✗'}"
    )

st.subheader("Actual")
st.write(f"**{actual:,.0f} days**")
st.caption("✓ means the actual duration fell inside that model's 80% interval.")
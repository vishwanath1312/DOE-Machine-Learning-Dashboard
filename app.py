import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import (
    mean_squared_error, roc_curve, auc,
    precision_score, recall_score, f1_score,
    confusion_matrix, ConfusionMatrixDisplay
)

# -------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------
st.set_page_config(page_title="DOE ML Dashboard", layout="wide")
st.title("🔬 DOE + Machine Learning Dashboard")

# -------------------------------------------------
# LOAD DATA
# -------------------------------------------------
@st.cache_data
def load_data(file="doe.xlsx"):
    df = pd.read_excel(file)
    df.columns = ["GMO", "Poloxamer", "ProbeTime", "ParticleSize", "Entrapment", "CDR"]
    return df

df = load_data()
X = df[["GMO", "Poloxamer", "ProbeTime"]]
Y = df[["ParticleSize", "Entrapment", "CDR"]]

# Feature engineering for CDR
X_fe = X.copy()
X_fe["GMO_x_ProbeTime"] = X["GMO"] * X["ProbeTime"]
X_fe["Poloxamer_x_ProbeTime"] = X["Poloxamer"] * X["ProbeTime"]

# -------------------------------------------------
# TRAIN MODELS
# -------------------------------------------------
@st.cache_resource
def train_models():
    X_tr, X_te, Y_tr, Y_te = train_test_split(X, Y, test_size=0.2, random_state=42)

    # Forward model for Particle Size + Entrapment
    fwd_rf = MultiOutputRegressor(
        RandomForestRegressor(n_estimators=300, random_state=42)
    )
    fwd_rf.fit(X_tr, Y_tr[["ParticleSize", "Entrapment"]])

    # Dedicated CDR model
    cdr_model = GradientBoostingRegressor(
        n_estimators=400, learning_rate=0.03, max_depth=3, random_state=42
    )
    cdr_model.fit(X_fe.loc[X_tr.index], Y_tr["CDR"])

    # Backward model (responses → formulation)
    bwd_model = MultiOutputRegressor(
        RandomForestRegressor(n_estimators=300, random_state=42)
    )
    bwd_model.fit(Y_tr, X_tr)

    return fwd_rf, cdr_model, bwd_model, X_tr, X_te, Y_tr, Y_te

fwd_rf, cdr_model, bwd_model, X_train, X_test, Y_train, Y_test = train_models()

# -------------------------------------------------
# HELPER FUNCTIONS
# -------------------------------------------------
def classification_metrics(y_true, y_pred):
    thr = np.median(y_true)
    yt = (y_true >= thr).astype(int)
    yp = (y_pred >= thr).astype(int)
    return precision_score(yt, yp, zero_division=0), recall_score(yt, yp, zero_division=0), f1_score(yt, yp, zero_division=0)

def compute_eer(y_true, y_score):
    yt = (y_true >= np.median(y_true)).astype(int)
    fpr, tpr, _ = roc_curve(yt, y_score)
    fnr = 1 - tpr
    i = np.argmin(np.abs(fpr - fnr))
    eer = (fpr[i] + fnr[i]) / 2
    return fpr, tpr, eer, i, auc(fpr, tpr)

def compute_confusion(y_true, y_pred):
    thr = np.median(y_true)
    yt = (y_true >= thr).astype(int)
    yp = (y_pred >= thr).astype(int)
    return confusion_matrix(yt, yp)

def get_cm_labels(name):
    labels = {
        "ParticleSize": ["Small Size", "Large Size"],
        "Entrapment": ["Low EE", "High EE"],
        "CDR": ["Slow Release", "Fast Release"],
        "GMO": ["Low GMO", "High GMO"],
        "Poloxamer": ["Low Poloxamer", "High Poloxamer"],
        "ProbeTime": ["Short Time", "Long Time"]
    }
    return labels.get(name, ["Low", "High"])

# -------------------------------------------------
# TABS
# -------------------------------------------------
tab1, tab2, tab3 = st.tabs(["🔁 Forward Prediction", "🔄 Backward Prediction", "⚙ Optimization"])

# =========================
# TAB 1 – FORWARD
# =========================
with tab1:
    st.subheader("Formulation → Responses")
    c1, c2, c3 = st.columns(3)

    with c1:
        gmo = st.number_input("GMO (%)", float(X.GMO.min()), float(X.GMO.max()), float(X.GMO.mean()))
    with c2:
        pol = st.number_input("Poloxamer 407 (%)", float(X.Poloxamer.min()), float(X.Poloxamer.max()), float(X.Poloxamer.mean()))
    with c3:
        pt = st.number_input("Probe Time (min)", float(X.ProbeTime.min()), float(X.ProbeTime.max()), float(X.ProbeTime.mean()))

    user_X = pd.DataFrame([[gmo, pol, pt]], columns=X.columns)
    user_X_fe = user_X.copy()
    user_X_fe["GMO_x_ProbeTime"] = gmo * pt
    user_X_fe["Poloxamer_x_ProbeTime"] = pol * pt

    ps_ee = fwd_rf.predict(user_X)
    cdr = cdr_model.predict(user_X_fe)

    st.dataframe(pd.DataFrame([[ps_ee[0,0], ps_ee[0,1], cdr[0]]], columns=Y.columns), use_container_width=True)

    st.markdown("---")
    st.subheader("📊 Model Performance")
    ps_ee_test = fwd_rf.predict(X_test)
    cdr_test = cdr_model.predict(X_fe.loc[X_test.index])

    # Performance Table
    rows = []
    for i, col in enumerate(["ParticleSize", "Entrapment"]):
        mse = mean_squared_error(Y_test[col], ps_ee_test[:, i])
        p, r, f1 = classification_metrics(Y_test[col], ps_ee_test[:, i])
        rows.append([col, mse, p, r, f1])
    mse = mean_squared_error(Y_test["CDR"], cdr_test)
    p, r, f1 = classification_metrics(Y_test["CDR"], cdr_test)
    rows.append(["CDR", mse, p, r, f1])
    st.dataframe(pd.DataFrame(rows, columns=["Output", "MSE", "Precision", "Recall", "F1"]), use_container_width=True)

    # ROC + Confusion Matrix
    target = st.selectbox("Select Output for ROC & CM", Y.columns)
    y_pred = cdr_test if target=="CDR" else ps_ee_test[:, Y.columns.get_loc(target)]
    fpr, tpr, eer, idx, roc_auc = compute_eer(Y_test[target], y_pred)
    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, label=f"AUC={roc_auc:.2f}")
    ax.scatter(fpr[idx], tpr[idx], color="red", label=f"EER={eer:.2f}")
    ax.plot([0,1],[0,1],"k--")
    ax.legend()
    st.pyplot(fig)

    cm = compute_confusion(Y_test[target], y_pred)
    fig, ax = plt.subplots()
    ConfusionMatrixDisplay(cm, display_labels=get_cm_labels(target)).plot(ax=ax, cmap="Blues")
    st.pyplot(fig)

# =========================
# TAB 2 – BACKWARD
# =========================
with tab2:
    st.subheader("Responses → Formulation")
    c1, c2, c3 = st.columns(3)
    with c1:
        ps = st.number_input("Particle Size (nm)", float(Y.ParticleSize.min()), float(Y.ParticleSize.max()), float(Y.ParticleSize.mean()))
    with c2:
        ent = st.number_input("Entrapment Efficiency (%)", float(Y.Entrapment.min()), float(Y.Entrapment.max()), float(Y.Entrapment.mean()))
    with c3:
        cdr_val = st.number_input("CDR (%)", float(Y.CDR.min()), float(Y.CDR.max()), float(Y.CDR.mean()))

    user_Y = pd.DataFrame([[ps, ent, cdr_val]], columns=Y.columns)
    pred_X = bwd_model.predict(user_Y)
    st.dataframe(pd.DataFrame(pred_X, columns=X.columns), use_container_width=True)

    # Backward Performance
    preds_test = bwd_model.predict(Y_test)
    rows = []
    for i, col in enumerate(X.columns):
        mse = mean_squared_error(X_test[col], preds_test[:, i])
        p, r, f1 = classification_metrics(X_test[col], preds_test[:, i])
        rows.append([col, mse, p, r, f1])
    st.dataframe(pd.DataFrame(rows, columns=["Parameter", "MSE", "Precision", "Recall", "F1"]), use_container_width=True)

# =========================
# TAB 3 – OPTIMIZATION
# =========================
with tab3:
    st.subheader("🎯 Optimal Formulation")
    if st.button("Compute Optimal Formulation"):
        grid = pd.DataFrame(
            [[g, p, t] for g in np.linspace(X.GMO.min(), X.GMO.max(), 10)
                        for p in np.linspace(X.Poloxamer.min(), X.Poloxamer.max(), 10)
                        for t in np.linspace(X.ProbeTime.min(), X.ProbeTime.max(), 10)],
            columns=X.columns
        )
        grid_fe = grid.copy()
        grid_fe["GMO_x_ProbeTime"] = grid["GMO"] * grid["ProbeTime"]
        grid_fe["Poloxamer_x_ProbeTime"] = grid["Poloxamer"] * grid["ProbeTime"]

        ps_ee = fwd_rf.predict(grid)
        grid["ParticleSize"] = ps_ee[:,0]
        grid["Entrapment"] = ps_ee[:,1]
        grid["CDR"] = cdr_model.predict(grid_fe)

        grid["Score"] = -grid["ParticleSize"] + grid["Entrapment"] + grid["CDR"]
        best = grid.loc[grid.Score.idxmax()]
        st.dataframe(best.to_frame("Optimal Value"), use_container_width=True)

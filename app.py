import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import griddata
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc
from sklearn.inspection import PartialDependenceDisplay
from sklearn.tree import plot_tree

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(page_title="DOE ML Dashboard", layout="wide")
st.title("🔬 DOE + Machine Learning Dashboard")

# -----------------------------
# LOAD DATA
# -----------------------------
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

# -----------------------------
# TRAIN MODELS
# -----------------------------
@st.cache_resource
def train_models():
    X_tr, X_te, Y_tr, Y_te = train_test_split(X, Y, test_size=0.2, random_state=42)

    fwd_rf = MultiOutputRegressor(RandomForestRegressor(n_estimators=100, random_state=42))
    fwd_rf.fit(X_tr, Y_tr[["ParticleSize", "Entrapment"]])

    cdr_model = GradientBoostingRegressor(n_estimators=200, learning_rate=0.05, max_depth=3, random_state=42)
    cdr_model.fit(X_fe.loc[X_tr.index], Y_tr["CDR"])

    bwd_model = MultiOutputRegressor(RandomForestRegressor(n_estimators=100, random_state=42))
    bwd_model.fit(Y_tr, X_tr)

    return fwd_rf, cdr_model, bwd_model, X_tr, X_te, Y_tr, Y_te

fwd_rf, cdr_model, bwd_model, X_train, X_test, Y_train, Y_test = train_models()

# -----------------------------
# HELPER FUNCTIONS
# -----------------------------
def classification_metrics(y_true, y_pred):
    thr = np.median(y_true)
    yt = (y_true >= thr).astype(int)
    yp = (y_pred >= thr).astype(int)
    return precision_score(yt, yp, zero_division=0), recall_score(yt, yp, zero_division=0), f1_score(yt, yp, zero_division=0)

def plot_roc(y_true, y_pred, ax=None, label=None):
    thr = np.median(y_true)
    yt = (y_true >= thr).astype(int)
    fpr, tpr, _ = roc_curve(yt, y_pred)
    roc_auc = auc(fpr, tpr)
    if ax is None:
        fig, ax = plt.subplots()
    ax.plot(fpr, tpr, label=f'{label} (AUC={roc_auc:.2f})')
    ax.plot([0,1],[0,1],'k--')
    ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
    ax.set_title("ROC Curve")
    ax.legend()
    return ax

# -----------------------------
# TABS
# -----------------------------
tab1, tab2, tab3 = st.tabs(["🔁 Forward Prediction", "🔄 Backward Prediction", "⚙ Optimization"])

# ==============================
# TAB 1 – FORWARD
# ==============================
with tab1:
    st.header("Forward Prediction: Formulation → Responses")
    c1, c2, c3 = st.columns(3)
    with c1:
        gmo = st.number_input("GMO (%)", float(X.GMO.min()), float(X.GMO.max()), float(X.GMO.mean()))
    with c2:
        pol = st.number_input("Poloxamer (%)", float(X.Poloxamer.min()), float(X.Poloxamer.max()), float(X.Poloxamer.mean()))
    with c3:
        pt = st.number_input("Probe Time (min)", float(X.ProbeTime.min()), float(X.ProbeTime.max()), float(X.ProbeTime.mean()))

    user_X = pd.DataFrame([[gmo, pol, pt]], columns=X.columns)
    user_X_fe = user_X.copy()
    user_X_fe["GMO_x_ProbeTime"] = gmo * pt
    user_X_fe["Poloxamer_x_ProbeTime"] = pol * pt

    ps_ee = fwd_rf.predict(user_X)
    cdr = cdr_model.predict(user_X_fe)
    st.subheader("Predicted Responses")
    st.dataframe(pd.DataFrame([[ps_ee[0,0], ps_ee[0,1], cdr[0]]], columns=Y.columns), use_container_width=True)

    # 3D scatter & response surface
    st.subheader("3D Scatter & Response Surface")
    grid = pd.DataFrame([[g, p, t] 
         for g in np.linspace(X.GMO.min(), X.GMO.max(), 10)
         for p in np.linspace(X.Poloxamer.min(), X.Poloxamer.max(), 10)
         for t in np.linspace(X.ProbeTime.min(), X.ProbeTime.max(), 10)],
        columns=X.columns)
    grid_fe = grid.copy()
    grid_fe["GMO_x_ProbeTime"] = grid["GMO"] * grid["ProbeTime"]
    grid_fe["Poloxamer_x_ProbeTime"] = grid["Poloxamer"] * grid["ProbeTime"]
    ps_ee = fwd_rf.predict(grid)
    grid["ParticleSize"] = ps_ee[:,0]
    grid["Entrapment"] = ps_ee[:,1]
    grid["CDR"] = cdr_model.predict(grid_fe)
    grid["Score"] = -grid["ParticleSize"] + grid["Entrapment"] + grid["CDR"]
    best = grid.loc[grid.Score.idxmax()]

    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot(111, projection='3d')
    sc = ax.scatter(grid["GMO"], grid["Poloxamer"], grid["ProbeTime"], c=grid["Score"], cmap="viridis", s=50, alpha=0.6)
    ax.scatter(best["GMO"], best["Poloxamer"], best["ProbeTime"], color="red", s=120, label="Optimal", edgecolors='k')
    ax.set_xlabel("GMO"); ax.set_ylabel("Poloxamer"); ax.set_zlabel("ProbeTime")
    ax.set_title("Forward: 3D Grid (Red = Optimal)")
    ax.legend()
    fig.colorbar(sc, ax=ax, label="Score")
    st.pyplot(fig)

    # PDP
    st.subheader("Partial Dependence Plot (Particle Size vs Inputs)")
    fig, ax = plt.subplots(figsize=(8,6))
    PartialDependenceDisplay.from_estimator(fwd_rf.estimators_[0], X_train, ["GMO","Poloxamer","ProbeTime"], ax=ax)
    st.pyplot(fig)

    # Correlation Heatmap
    st.subheader("Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(6,5))
    sns.heatmap(df.corr(), annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)

    # ROC Curves
    st.subheader("ROC Curves (Forward Prediction)")
    fig, ax = plt.subplots(figsize=(6,5))
    plot_roc(Y_test["ParticleSize"], fwd_rf.predict(X_test)[:,0], ax=ax, label="ParticleSize")
    plot_roc(Y_test["Entrapment"], fwd_rf.predict(X_test)[:,1], ax=ax, label="Entrapment")
    plot_roc(Y_test["CDR"], cdr_model.predict(X_fe.loc[X_test.index]), ax=ax, label="CDR")
    st.pyplot(fig)

    # Random Forest Tree Visualization
    st.subheader("Random Forest Tree Visualization")
    target_col = st.selectbox("Select Output Tree (Forward)", ["ParticleSize", "Entrapment"])
    target_idx = {"ParticleSize":0, "Entrapment":1}[target_col]
    estimator = fwd_rf.estimators_[target_idx]
    tree_idx = st.slider("Select tree number", 0, len(estimator.estimators_)-1, 0)
    single_tree = estimator.estimators_[tree_idx]
    fig, ax = plt.subplots(figsize=(20,10))
    plot_tree(single_tree, feature_names=X.columns, filled=True, rounded=True, fontsize=10)
    ax.set_title(f"Random Forest Tree #{tree_idx} for {target_col} (Forward)")
    st.pyplot(fig)

# ==============================
# TAB 2 – BACKWARD
# ==============================
with tab2:
    st.header("Backward Prediction: Responses → Formulation")
    c1, c2, c3 = st.columns(3)
    with c1:
        ps = st.number_input("Particle Size", float(Y.ParticleSize.min()), float(Y.ParticleSize.max()), float(Y.ParticleSize.mean()))
    with c2:
        ent = st.number_input("Entrapment Efficiency", float(Y.Entrapment.min()), float(Y.Entrapment.max()), float(Y.Entrapment.mean()))
    with c3:
        cdr_val = st.number_input("CDR", float(Y.CDR.min()), float(Y.CDR.max()), float(Y.CDR.mean()))

    user_Y = pd.DataFrame([[ps, ent, cdr_val]], columns=Y.columns)
    pred_X = bwd_model.predict(user_Y)
    st.subheader("Predicted Formulation")
    st.dataframe(pd.DataFrame(pred_X, columns=X.columns), use_container_width=True)

    # 3D Scatter
    st.subheader("Backward: 3D Scatter & Response Surface")
    grid_Y = pd.DataFrame(
        [[ps_i, ent_i, cdr_i] 
         for ps_i in np.linspace(Y.ParticleSize.min(), Y.ParticleSize.max(),10)
         for ent_i in np.linspace(Y.Entrapment.min(), Y.Entrapment.max(),10)
         for cdr_i in np.linspace(Y.CDR.min(), Y.CDR.max(),10)],
        columns=Y.columns)
    pred_X_grid = bwd_model.predict(grid_Y)
    grid_Y["GMO"] = pred_X_grid[:,0]
    grid_Y["Poloxamer"] = pred_X_grid[:,1]
    grid_Y["ProbeTime"] = pred_X_grid[:,2]
    grid_Y["Score"] = grid_Y["GMO"] + grid_Y["Poloxamer"] + grid_Y["ProbeTime"]
    best_bwd = grid_Y.loc[grid_Y.Score.idxmax()]

    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot(111, projection='3d')
    sc = ax.scatter(grid_Y["GMO"], grid_Y["Poloxamer"], grid_Y["ProbeTime"], c=grid_Y["Score"], cmap="plasma", s=50, alpha=0.6)
    ax.scatter(best_bwd["GMO"], best_bwd["Poloxamer"], best_bwd["ProbeTime"], color="red", s=120, label="Optimal", edgecolors='k')
    ax.set_xlabel("GMO"); ax.set_ylabel("Poloxamer"); ax.set_zlabel("ProbeTime")
    ax.set_title("Backward: 3D Grid (Red = Optimal)")
    ax.legend()
    fig.colorbar(sc, ax=ax, label="Score")
    st.pyplot(fig)

    # PDP
    st.subheader("Partial Dependence Plot (Predicted GMO vs Inputs)")
    fig, ax = plt.subplots(figsize=(8,6))
    PartialDependenceDisplay.from_estimator(bwd_model.estimators_[0], Y_train, ["ParticleSize","Entrapment","CDR"], ax=ax)
    st.pyplot(fig)

    # Correlation Heatmap
    st.subheader("Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(6,5))
    sns.heatmap(df.corr(), annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)

    # ROC Curves
    st.subheader("ROC Curves (Backward Prediction)")
    fig, ax = plt.subplots(figsize=(6,5))
    plot_roc(X_test["GMO"], bwd_model.predict(Y_test)[:,0], ax=ax, label="GMO")
    plot_roc(X_test["Poloxamer"], bwd_model.predict(Y_test)[:,1], ax=ax, label="Poloxamer")
    plot_roc(X_test["ProbeTime"], bwd_model.predict(Y_test)[:,2], ax=ax, label="ProbeTime")
    st.pyplot(fig)

    # Random Forest Tree Visualization
    st.subheader("Random Forest Tree Visualization")
    target_col = st.selectbox("Select Output Tree (Backward)", ["GMO", "Poloxamer", "ProbeTime"])
    target_idx = {"GMO":0, "Poloxamer":1, "ProbeTime":2}[target_col]
    estimator = bwd_model.estimators_[target_idx]
    tree_idx = st.slider("Select tree number (Backward)", 0, len(estimator.estimators_)-1, 0)
    single_tree = estimator.estimators_[tree_idx]
    fig, ax = plt.subplots(figsize=(20,10))
    plot_tree(single_tree, feature_names=Y.columns, filled=True, rounded=True, fontsize=10)
    ax.set_title(f"Random Forest Tree #{tree_idx} for {target_col} (Backward)")
    st.pyplot(fig)

# ==============================
# TAB 3 – OPTIMIZATION
# ==============================
with tab3:
    st.header("Optimization: 🎯 Find Optimal Formulation")
    st.subheader("Forward Model Score Optimization")
    st.dataframe(best.to_frame("Optimal Value"), use_container_width=True)

    # 3D scatter
    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot(111, projection='3d')
    sc = ax.scatter(grid["GMO"], grid["Poloxamer"], grid["ProbeTime"], c=grid["Score"], cmap="viridis", s=50, alpha=0.6)
    ax.scatter(best["GMO"], best["Poloxamer"], best["ProbeTime"], color="red", s=120, label="Optimal", edgecolors='k')
    ax.set_xlabel("GMO"); ax.set_ylabel("Poloxamer"); ax.set_zlabel("ProbeTime")
    ax.set_title("Optimization: 3D Grid (Red = Optimal)")
    ax.legend()
    fig.colorbar(sc, ax=ax, label="Score")
    st.pyplot(fig)

    # Random Forest Tree Visualization (Forward)
    st.subheader("Random Forest Tree Visualization (Forward Model)")
    target_col = st.selectbox("Select Output Tree (Optimization)", ["ParticleSize", "Entrapment"])
    target_idx = {"ParticleSize":0, "Entrapment":1}[target_col]
    estimator = fwd_rf.estimators_[target_idx]
    tree_idx = st.slider("Select tree number (Optimization)", 0, len(estimator.estimators_)-1, 0)
    single_tree = estimator.estimators_[tree_idx]
    fig, ax = plt.subplots(figsize=(20,10))
    plot_tree(single_tree, feature_names=X.columns, filled=True, rounded=True, fontsize=10)
    ax.set_title(f"Random Forest Tree #{tree_idx} for {target_col} (Optimization)")
    st.pyplot(fig)

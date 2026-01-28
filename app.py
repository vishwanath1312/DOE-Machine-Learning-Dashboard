import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc
from sklearn.inspection import PartialDependenceDisplay
from sklearn.tree import plot_tree
from scipy.interpolate import griddata

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

def plot_3d_surface_scatter(grid, x_col, y_col, z_col, score_col, best_row, title):
    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot(111, projection='3d')
    
    # Scatter
    sc = ax.scatter(grid[x_col], grid[y_col], grid[z_col], c=grid[score_col], cmap="viridis", s=50, alpha=0.6)
    
    # Surface interpolation
    # Only works for 2D surface, we fix z_col (e.g., ProbeTime) to median
    z_fixed = grid[z_col].median()
    mask = np.isclose(grid[z_col], z_fixed)
    xi = np.linspace(grid[x_col].min(), grid[x_col].max(), 30)
    yi = np.linspace(grid[y_col].min(), grid[y_col].max(), 30)
    XI, YI = np.meshgrid(xi, yi)
    ZI = griddata((grid[x_col][mask], grid[y_col][mask]), grid[score_col][mask], (XI, YI), method='cubic')
    
    ax.plot_surface(XI, YI, ZI, alpha=0.3, cmap="viridis")
    
    # Highlight best point
    ax.scatter(best_row[x_col], best_row[y_col], best_row[z_col], color="red", s=120, label="Optimal", edgecolors='k')
    
    ax.set_xlabel(x_col); ax.set_ylabel(y_col); ax.set_zlabel(z_col)
    ax.set_title(title)
    ax.legend()
    fig.colorbar(sc, ax=ax, label="Score")
    st.pyplot(fig)

# -----------------------------
# FLOWCHART IMAGE
# -----------------------------
flowchart_path = "Flow Diagram.png"

# -----------------------------
# TABS
# -----------------------------
tab1, tab2, tab3, tab4 = st.tabs([
    "🔁 Forward Prediction",
    "🔄 Backward Prediction",
    "⚙ Optimization",
    "📈 Dashboard Flowchart"
])

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

    # 3D Scatter + Surface
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
    plot_3d_surface_scatter(grid, "GMO", "Poloxamer", "ProbeTime", "Score", best, "Forward Prediction: 3D Grid & Response Surface")

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

    # 3D Scatter + Surface
    st.subheader("3D Scatter & Response Surface")
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
    plot_3d_surface_scatter(grid_Y, "GMO", "Poloxamer", "ProbeTime", "Score", best_bwd, "Backward Prediction: 3D Grid & Response Surface")

# ==============================
# TAB 3 – OPTIMIZATION
# ==============================
with tab3:
    st.header("Optimization: Find Optimal Formulation")
    st.subheader("Forward Model Score Optimization")
    st.dataframe(best.to_frame("Optimal Value"), use_container_width=True)

    # 3D Scatter + Surface
    plot_3d_surface_scatter(grid, "GMO", "Poloxamer", "ProbeTime", "Score", best, "Optimization: 3D Grid & Response Surface")

# ==============================
# TAB 4 – FLOWCHART
# ==============================
with tab4:
    st.header("📊 DOE + ML Dashboard Flowchart")
    st.image(flowchart_path, use_column_width=True)

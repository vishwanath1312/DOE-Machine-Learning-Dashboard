import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
from sklearn.inspection import PartialDependenceDisplay
from scipy.spatial import cKDTree

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

def plot_roc(y_true, y_pred, title="ROC Curve"):
    thr = np.median(y_true)
    yt = (y_true >= thr).astype(int)
    fpr, tpr, _ = roc_curve(yt, y_pred)
    roc_auc = auc(fpr, tpr)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f"AUC={roc_auc:.2f}"))
    fig.add_trace(go.Scatter(x=[0,1], y=[0,1], mode='lines', line=dict(dash='dash', color='black')))
    fig.update_layout(title=title, xaxis_title='False Positive Rate', yaxis_title='True Positive Rate')
    st.plotly_chart(fig)

def plot_3d_surface_scatter(df, x_col, y_col, z_col, score_col, best_row, title):
    fig = go.Figure()
    fig.add_trace(go.Scatter3d(
        x=df[x_col], y=df[y_col], z=df[z_col],
        mode='markers',
        marker=dict(size=5, color=df[score_col], colorscale='Viridis', opacity=0.6),
        name='Grid Points'
    ))
    # Surface (nearest neighbor)
    xi = np.linspace(df[x_col].min(), df[x_col].max(), 30)
    yi = np.linspace(df[y_col].min(), df[y_col].max(), 30)
    XI, YI = np.meshgrid(xi, yi)
    tree = cKDTree(df[[x_col, y_col]].values)
    ZI = np.array([df[score_col].values[tree.query([xi, yi])[1]] for xi, yi in zip(np.ravel(XI), np.ravel(YI))])
    ZI = ZI.reshape(XI.shape)
    fig.add_trace(go.Surface(x=XI, y=YI, z=ZI, opacity=0.4, colorscale='Viridis', name='Response Surface'))
    # Highlight optimal
    fig.add_trace(go.Scatter3d(
        x=[best_row[x_col]], y=[best_row[y_col]], z=[best_row[z_col]],
        mode='markers',
        marker=dict(size=10, color='red'),
        name='Optimal'
    ))
    fig.update_layout(scene=dict(xaxis_title=x_col, yaxis_title=y_col, zaxis_title=z_col), title=title)
    st.plotly_chart(fig, use_container_width=True)

def plot_correlation_heatmap(df, title):
    fig, ax = plt.subplots()
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm', ax=ax)
    ax.set_title(title)
    st.pyplot(fig)

def plot_pdp(model, X_data, feature, target_name):
    fig, ax = plt.subplots()
    PartialDependenceDisplay.from_estimator(model, X_data, [feature], ax=ax)
    ax.set_title(f"PDP: {target_name} vs {feature}")
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
    with c1: gmo = st.number_input("GMO (%)", float(X.GMO.min()), float(X.GMO.max()), float(X.GMO.mean()))
    with c2: pol = st.number_input("Poloxamer (%)", float(X.Poloxamer.min()), float(X.Poloxamer.max()), float(X.Poloxamer.mean()))
    with c3: pt = st.number_input("Probe Time (min)", float(X.ProbeTime.min()), float(X.ProbeTime.max()), float(X.ProbeTime.mean()))

    user_X = pd.DataFrame([[gmo, pol, pt]], columns=X.columns)
    user_X_fe = user_X.copy()
    user_X_fe["GMO_x_ProbeTime"] = gmo * pt
    user_X_fe["Poloxamer_x_ProbeTime"] = pol * pt

    ps_ee = fwd_rf.predict(user_X)
    cdr = cdr_model.predict(user_X_fe)
    st.subheader("Predicted Responses")
    st.dataframe(pd.DataFrame([[ps_ee[0,0], ps_ee[0,1], cdr[0]]], columns=Y.columns), use_container_width=True)

    # 3D Grid & Surface
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

    plot_3d_surface_scatter(grid, "GMO", "Poloxamer", "ProbeTime", "Score", best, "Forward Prediction 3D Surface & Scatter")

    # PDPs for all three outputs
    st.subheader("Partial Dependence Plots (All Responses)")
    for target, model in zip(["ParticleSize", "Entrapment"], fwd_rf.estimators_):
        st.markdown(f"**PDPs for {target}**")
        for feat in X.columns:
            plot_pdp(model, X_train, feat, target)
    st.markdown(f"**PDPs for CDR**")
    for feat in X_fe.columns[:-2]:  # Original X features
        plot_pdp(cdr_model, X_fe, feat, "CDR")

    # Correlation heatmaps
    st.subheader("Correlation Heatmaps")
    plot_correlation_heatmap(X, "X Features (Formulation) Correlation")
    plot_correlation_heatmap(Y, "Y Features (Responses) Correlation")

    # ROC curves
    st.subheader("ROC Curves")
    plot_roc(Y_test["ParticleSize"], fwd_rf.predict(X_test)[:,0], "ParticleSize ROC")
    plot_roc(Y_test["Entrapment"], fwd_rf.predict(X_test)[:,1], "Entrapment ROC")
    plot_roc(Y_test["CDR"], cdr_model.predict(X_fe.loc[X_test.index]), "CDR ROC")

# ==============================
# TAB 2 – BACKWARD
# ==============================
with tab2:
    st.header("Backward Prediction: Responses → Formulation")
    c1, c2, c3 = st.columns(3)
    with c1: ps = st.number_input("Particle Size", float(Y.ParticleSize.min()), float(Y.ParticleSize.max()), float(Y.ParticleSize.mean()))
    with c2: ent = st.number_input("Entrapment Efficiency", float(Y.Entrapment.min()), float(Y.Entrapment.max()), float(Y.Entrapment.mean()))
    with c3: cdr_val = st.number_input("CDR", float(Y.CDR.min()), float(Y.CDR.max()), float(Y.CDR.mean()))

    user_Y = pd.DataFrame([[ps, ent, cdr_val]], columns=Y.columns)
    pred_X = bwd_model.predict(user_Y)
    st.subheader("Predicted Formulation")
    st.dataframe(pd.DataFrame(pred_X, columns=X.columns), use_container_width=True)

    # 3D Surface & Scatter
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

    plot_3d_surface_scatter(grid_Y, "GMO", "Poloxamer", "ProbeTime", "Score", best_bwd, "Backward Prediction 3D Surface & Scatter")

    # PDPs
    st.subheader("Partial Dependence Plots (Backward Outputs)")
    for target, model in zip(X.columns, bwd_model.estimators_):
        st.markdown(f"**PDPs for {target}**")
        for feat in Y.columns:
            plot_pdp(model, Y_train, feat, target)

    # Correlation heatmaps
    st.subheader("Correlation Heatmaps")
    plot_correlation_heatmap(X, "X Features (Formulation) Correlation")
    plot_correlation_heatmap(Y, "Y Features (Responses) Correlation")

    # ROC curves
    st.subheader("ROC Curves")
    pred_X_test = bwd_model.predict(Y_test)
    plot_roc(X_test["GMO"], pred_X_test[:,0], "GMO ROC")
    plot_roc(X_test["Poloxamer"], pred_X_test[:,1], "Poloxamer ROC")
    plot_roc(X_test["ProbeTime"], pred_X_test[:,2], "ProbeTime ROC")

# ==============================
# TAB 3 – OPTIMIZATION
# ==============================
with tab3:
    st.header("Optimization: Find Optimal Formulation")
    st.subheader("Forward Model Optimal Score")
    st.dataframe(best.to_frame("Optimal Value"), use_container_width=True)
    plot_3d_surface_scatter(grid, "GMO", "Poloxamer", "ProbeTime", "Score", best, "Optimization 3D Surface & Scatter")

    # PDPs for all outputs
    st.subheader("Partial Dependence Plots (All Outputs)")
    for target, model in zip(["ParticleSize", "Entrapment"], fwd_rf.estimators_):
        st.markdown(f"**PDPs for {target}**")
        for feat in X.columns:
            plot_pdp(model, X_train, feat, target)
    for feat in X_fe.columns[:-2]:
        plot_pdp(cdr_model, X_fe, feat, "CDR")

    # ROC curves
    st.subheader("ROC Curves")
    plot_roc(Y["ParticleSize"], ps_ee[:,0], "ParticleSize ROC (Grid)")
    plot_roc(Y["Entrapment"], ps_ee[:,1], "Entrapment ROC (Grid)")
    plot_roc(Y["CDR"], cdr_model.predict(grid_fe), "CDR ROC (Grid)")

# ==============================
# TAB 4 – FLOWCHART
# ==============================
with tab4:
    st.header("📊 DOE + ML Dashboard Flowchart")
    st.image(flowchart_path, use_column_width=True)

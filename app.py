# ---- Safe Matplotlib Import (Streamlit Cloud Fix) ----
try:
    import matplotlib
    matplotlib.use("Agg")   # IMPORTANT: Cloud-safe backend
    import matplotlib.pyplot as plt
except Exception as e:
    print("Matplotlib failed to load:", e)
    plt = None

# app.py
# ---- THEME TOGGLE (Dark / Light Mode) ----
import streamlit as st

# Add a sidebar switch for theme
theme = st.sidebar.selectbox(
    "Theme Mode",
    ["Light", "Dark"],
    index=0
)

# Apply theme using custom CSS
def apply_theme(mode):
    if mode == "Dark":
        dark_css = """
        <style>
        body, .stApp {
            background-color: #0e1117 !important;
            color: #ffffff !important;
        }
        .stSidebar, .css-1lcbmhc {
            background-color: #161a23 !important;
            color: white !important;
        }
        .stButton>button {
            background-color: #1f6feb !important;
            color: white !important;
            border-radius: 8px;
        }
        .stSelectbox, .stTextInput, .stFileUploader {
            background-color: #242831 !important;
            color: white !important;
        }
        </style>
        """
        st.markdown(dark_css, unsafe_allow_html=True)

    else:
        light_css = """
        <style>
        body, .stApp {
            background-color: #ffffff !important;
            color: #000000 !important;
        }
        .stSidebar {
            background-color: #f0f2f6 !important;
        }
        .stButton>button {
            background-color: #0066ff !important;
            color: white !important;
            border-radius: 8px;
        }
        </style>
        """
        st.markdown(light_css, unsafe_allow_html=True)

apply_theme(theme)


import streamlit as st
import pandas as pd
import numpy as np


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.metrics import (
    r2_score, mean_squared_error,
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report,
    roc_curve, auc
)
from sklearn.inspection import permutation_importance

st.set_page_config(page_title="Model Hub Pro", layout="wide", initial_sidebar_state="expanded")

# ---- Header ----
st.markdown(
    """
    <style>
    .big-title { font-size:28px; font-weight:700; }
    .muted { color: #6c757d; }
    </style>
    """, unsafe_allow_html=True
)
st.markdown('<div class="big-title">Model Hub Pro — ML playground(Rishi jain)</div>', unsafe_allow_html=True)
st.markdown('<div class="muted">Upload CSV → preprocess → train models → visualize insights (regression & classification)</div>', unsafe_allow_html=True)
st.write("---")

# ---- Sidebar: Upload & Config ----
with st.sidebar:
    st.header("Upload & Config")
    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
    sample_btn = st.button("Load demo dataset (Iris)")
    st.markdown("---")
    st.subheader("Preprocessing")
    do_label_encode = st.checkbox("Auto label-encode categorical", value=True)
    scaling_option = st.selectbox("Scaling", ["None", "StandardScaler", "MinMaxScaler"], index=1)
    st.markdown("---")
    st.subheader("Split & Random seed")
    test_size = st.slider("Test size (%)", 5, 50, 20)
    random_state = st.number_input("Random state", value=42, step=1)
    st.markdown("---")
    st.subheader("Models to train")
    train_lr = st.checkbox("Linear Regression", value=True)
    train_log = st.checkbox("Logistic Regression", value=True)
    train_dt = st.checkbox("Decision Tree", value=True)
    st.markdown("---")
    st.subheader("Advanced")
    show_feature_importance = st.checkbox("Show feature importance (for tree)", value=True)
    show_permutation_importance = st.checkbox("Permutation importance (slower)", value=False)

# ---- load dataset ----
if uploaded_file is None and not sample_btn:
    st.info("Upload a CSV or click 'Load demo dataset (Iris)' to try.")
    st.stop()

if sample_btn:
    from sklearn.datasets import load_iris
    iris = load_iris(as_frame=True)
    df = iris.frame.copy()
    # iris.frame has 'target' numeric; map to names to simulate categorical target
    df['target'] = iris.target
    df['target'] = df['target'].map(dict(enumerate(iris.target_names)))
else:
    try:
        df = pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"Failed to read CSV: {e}")
        st.stop()

st.sidebar.success(f"Dataset loaded — {df.shape[0]} rows, {df.shape[1]} cols")
st.subheader("Dataset preview")
st.dataframe(df.head(200))

# ---- Target & Features selection ----
cols = df.columns.tolist()
with st.expander("Choose target and features (auto selects all other cols)"):
    target_col = st.selectbox("Target column", options=cols, index=len(cols)-1)
    default_features = [c for c in cols if c != target_col]
    features = st.multiselect("Features (select at least one)", options=default_features, default=default_features)

if not features:
    st.error("Select at least one feature.")
    st.stop()

X = df[features].copy()
y = df[target_col].copy()

# ---- Preprocessing ----
le_dict = {}

# Label-encode X categorical columns
if do_label_encode:
    for col in X.columns:
        if X[col].dtype == "object" or X[col].dtype.name == "category":
            le = LabelEncoder()
            X[col] = X[col].fillna("___MISSING___")
            X[col] = le.fit_transform(X[col].astype(str))
            le_dict[col] = le

# Keep original y for potential inverse transform mapping
y_original = y.copy()

# If y is categorical/object and user allows, label-encode it now (we'll ensure y is a Series later)
if do_label_encode:
    if y.dtype == "object" or y.dtype.name == "category":
        le_y = LabelEncoder()
        y = y.fillna("___MISSING___")
        y = le_y.fit_transform(y.astype(str))
        # store encoder
        le_dict[target_col] = le_y

# numeric conversion and missing handling for X
for col in X.columns:
    if pd.api.types.is_numeric_dtype(X[col]):
        X[col] = X[col].fillna(X[col].median())
    else:
        try:
            X[col] = pd.to_numeric(X[col])
            X[col] = X[col].fillna(X[col].median())
        except:
            le = LabelEncoder()
            X[col] = X[col].fillna("___MISSING___")
            X[col] = le.fit_transform(X[col].astype(str))
            le_dict[col] = le

# scaling
if scaling_option != "None":
    scaler = StandardScaler() if scaling_option == "StandardScaler" else MinMaxScaler()
    X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

# ---- Task detection & override ----
# Auto-detect
detected_task = None
if pd.api.types.is_numeric_dtype(y):
    unique = np.unique(y[~pd.isnull(y)]) if isinstance(y, (pd.Series, np.ndarray)) else np.unique(y)
    detected_task = "classification" if len(unique) <= 20 else "regression"
else:
    detected_task = "classification"

# Provide override so user can force regression/classification
task_choice = st.selectbox("Task type (auto-detected or force):", options=["Auto-detect", "Regression", "Classification"], index=0)
if task_choice == "Auto-detect":
    task = detected_task
else:
    task = task_choice.lower()

st.markdown(f"**Detected task:** `{detected_task}` → **Using:** `{task}`")

# If user forced regression but y is non-numeric, encode y so regression can run
if task == "regression":
    if not pd.api.types.is_numeric_dtype(y):
        st.warning("Target is non-numeric but you forced Regression — label-encoding target for regression.")
        le_y_force = LabelEncoder()
        y = le_y_force.fit_transform(y.astype(str))
        le_dict[target_col] = le_y_force

# Ensure y is a pandas Series with a name (fixes .rename error)
if isinstance(y, np.ndarray):
    y = pd.Series(y, name=target_col)
elif isinstance(y, pd.Series):
    y = y.rename(target_col)
else:
    # fallback
    y = pd.Series(y, name=target_col)

# ---- Train/Test split ----
try:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size/100.0, random_state=int(random_state),
        stratify=y if task == "classification" else None
    )
except Exception as e:
    st.error(f"Train/test split failed: {e}")
    st.stop()

# ---- Train Models ----
models = {}
metrics = {}

def fmt(x): return float(f"{x:.4f}")

# Linear Regression (only when task==regression)
if train_lr and task == "regression":
    try:
        lr = LinearRegression()
        lr.fit(X_train, y_train)
        y_pred = lr.predict(X_test)
        models["LinearRegression"] = lr
        metrics["LinearRegression"] = {"R2": fmt(r2_score(y_test, y_pred)), "MSE": fmt(mean_squared_error(y_test, y_pred))}
    except Exception as e:
        st.warning(f"Linear Regression failed: {e}")

# Logistic Regression (only when task==classification)
if train_log and task == "classification":
    try:
        solver = "lbfgs" if len(np.unique(y_train)) > 2 else "liblinear"
        log = LogisticRegression(max_iter=2000, solver=solver)
        log.fit(X_train, y_train)
        y_pred = log.predict(X_test)
        models["LogisticRegression"] = log
        metrics["LogisticRegression"] = {
            "Accuracy": fmt(accuracy_score(y_test, y_pred)),
            "Precision": fmt(precision_score(y_test, y_pred, average="weighted", zero_division=0)),
            "Recall": fmt(recall_score(y_test, y_pred, average="weighted", zero_division=0)),
            "F1": fmt(f1_score(y_test, y_pred, average="weighted", zero_division=0))
        }
    except Exception as e:
        st.warning(f"Logistic Regression failed: {e}")

# Decision Tree
if train_dt:
    if task == "classification":
        try:
            dtc = DecisionTreeClassifier(random_state=int(random_state))
            dtc.fit(X_train, y_train)
            y_pred = dtc.predict(X_test)
            models["DecisionTreeClassifier"] = dtc
            metrics["DecisionTreeClassifier"] = {
                "Accuracy": fmt(accuracy_score(y_test, y_pred)),
                "Precision": fmt(precision_score(y_test, y_pred, average="weighted", zero_division=0)),
                "Recall": fmt(recall_score(y_test, y_pred, average="weighted", zero_division=0)),
                "F1": fmt(f1_score(y_test, y_pred, average="weighted", zero_division=0))
            }
        except Exception as e:
            st.warning(f"Decision Tree Classifier failed: {e}")
    else:
        try:
            dtr = DecisionTreeRegressor(random_state=int(random_state))
            dtr.fit(X_train, y_train)
            y_pred = dtr.predict(X_test)
            models["DecisionTreeRegressor"] = dtr
            metrics["DecisionTreeRegressor"] = {"R2": fmt(r2_score(y_test, y_pred)), "MSE": fmt(mean_squared_error(y_test, y_pred))}
        except Exception as e:
            st.warning(f"Decision Tree Regressor failed: {e}")

# ---- Top-level Metrics Cards ----
st.write("### Model summary")
if not metrics:
    st.warning("No models trained — enable models in sidebar or change task selection.")
else:
    cols = st.columns(len(metrics))
    for c, (mname, mets) in zip(cols, metrics.items()):
        primary = list(mets.items())[0]
        # show numeric metric nicely
        c.metric(label=mname, value=primary[1])
        for k, v in mets.items():
            c.write(f"- {k}: **{v}**")

st.write("---")

# ---- Visuals & Insights ----
left, right = st.columns([2, 1])

with left:
    st.subheader("Detailed model visuals")
    if models:
        model_name = st.selectbox("Pick model to inspect", options=list(models.keys()))
        model = models[model_name]
        st.markdown(f"**Inspecting:** `{model_name}`")

        # classification visuals
        if task == "classification":
            y_pred = model.predict(X_test)
            st.markdown("#### Confusion Matrix")
            try:
                cm = confusion_matrix(y_test, y_pred)
                fig, ax = plt.subplots(figsize=(5, 4))
                im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
                ax.set_title("Confusion matrix")
                ax.set_xlabel("Predicted")
                ax.set_ylabel("True")
                plt.colorbar(im, ax=ax)
                labels = np.unique(np.concatenate([y_test.values, y_pred]))
                ax.set_xticks(np.arange(len(labels))); ax.set_yticks(np.arange(len(labels)))
                ax.set_xticklabels(labels); ax.set_yticklabels(labels)
                for i in range(cm.shape[0]):
                    for j in range(cm.shape[1]):
                        ax.text(j, i, cm[i, j], ha="center", va="center", color="black")
                st.pyplot(fig)
            except Exception as e:
                st.warning(f"Confusion matrix failed: {e}")

            # classification report
            try:
                st.markdown("#### Classification report")
                report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
                report_df = pd.DataFrame(report).transpose()
                st.dataframe(report_df.style.format("{:.4f}"))
            except Exception as e:
                st.warning(f"Classification report failed: {e}")

            # ROC (binary)
            if len(np.unique(y_test)) == 2:
                try:
                    st.markdown("#### ROC Curve (binary)")
                    y_prob = model.predict_proba(X_test)[:, 1]
                    fpr, tpr, _ = roc_curve(y_test, y_prob)
                    roc_auc = auc(fpr, tpr)
                    fig2, ax2 = plt.subplots()
                    ax2.plot(fpr, tpr, lw=2, label=f'AUC = {roc_auc:.4f}')
                    ax2.plot([0, 1], [0, 1], '--', lw=1)
                    ax2.set_xlabel('False Positive Rate')
                    ax2.set_ylabel('True Positive Rate')
                    ax2.set_title('ROC curve')
                    ax2.legend(loc="lower right")
                    st.pyplot(fig2)
                except Exception as e:
                    st.warning(f"ROC plot failed: {e}")

            # feature importance for tree
            if "DecisionTree" in model_name and show_feature_importance:
                try:
                    st.markdown("#### Feature importance (Decision Tree)")
                    importances = model.feature_importances_
                    feat_imp = pd.Series(importances, index=X.columns).sort_values(ascending=False)
                    fig3, ax3 = plt.subplots(figsize=(6, 4))
                    feat_imp.plot.bar(ax=ax3)
                    ax3.set_ylabel("Importance")
                    st.pyplot(fig3)

                    if show_permutation_importance:
                        st.markdown("Permutation importance (may take time)...")
                        perm = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=0)
                        perm_imp = pd.Series(perm.importances_mean, index=X.columns).sort_values(ascending=False)
                        fig4, ax4 = plt.subplots(figsize=(6, 4))
                        perm_imp.plot.bar(ax=ax4)
                        st.pyplot(fig4)
                except Exception as e:
                    st.warning(f"Feature importance error: {e}")

        # regression visuals
        else:
            try:
                y_pred = model.predict(X_test)
                st.markdown("#### Regression metrics")
                st.write(f"- R² = **{r2_score(y_test, y_pred):.4f}**")
                st.write(f"- MSE = **{mean_squared_error(y_test, y_pred):.4f}**")

                st.markdown("#### Residual plot")
                residuals = y_test - y_pred
                fig5, ax5 = plt.subplots()
                ax5.scatter(y_pred, residuals, alpha=0.6)
                ax5.axhline(0, linestyle='--', color='red')
                ax5.set_xlabel("Predicted")
                ax5.set_ylabel("Residuals")
                ax5.set_title("Residuals vs Predicted")
                st.pyplot(fig5)

                if "DecisionTree" in model_name and show_feature_importance:
                    st.markdown("#### Feature importance (Decision Tree Regressor)")
                    importances = model.feature_importances_
                    feat_imp = pd.Series(importances, index=X.columns).sort_values(ascending=False)
                    fig6, ax6 = plt.subplots(figsize=(6, 4))
                    feat_imp.plot.bar(ax=ax6)
                    ax6.set_ylabel("Importance")
                    st.pyplot(fig6)
            except Exception as e:
                st.warning(f"Regression visuals failed: {e}")
    else:
        st.info("Train models to see visuals.")

with right:
    st.subheader("Quick data & model diagnostics")
    st.markdown("**Data types**")
    st.write(df[features].dtypes)
    st.markdown("**Train / Test sizes**")
    st.write(f"- Train: {X_train.shape[0]} rows\n- Test: {X_test.shape[0]} rows")
    st.markdown("**Feature snapshot**")
    st.dataframe(X.head(50))

    st.markdown("**Download processed dataset**")
    # ensure y is a Series with name, then concat
    if isinstance(y, np.ndarray):
        y_for_save = pd.Series(y, name=target_col)
    else:
        y_for_save = y.rename(target_col) if isinstance(y, pd.Series) else pd.Series(y, name=target_col)
    processed_csv = pd.concat([X, y_for_save], axis=1).to_csv(index=False)
    st.download_button("Download processed CSV", data=processed_csv, file_name="processed_dataset.csv", mime="text/csv")

    # show sample predictions for chosen model
    st.markdown("---")
    st.subheader("Sample predictions")
    if models:
        chosen = st.selectbox("Model for sample predictions", options=list(models.keys()))
        mdl = models[chosen]
        try:
            preds = mdl.predict(X_test)
            out = X_test.reset_index(drop=True).copy()
            out[f"true_{target_col}"] = y_test.reset_index(drop=True)
            out[f"pred_{chosen}"] = preds
            st.dataframe(out.head(100))
        except Exception as e:
            st.warning(f"Prediction preview failed: {e}")
    else:
        st.info("No model trained yet.")

# ---- Footer tips ----
st.write("---")
with st.expander("Tips & next steps"):
    st.markdown("""
    - If you force Regression but your target was categorical, it gets label-encoded to numeric for training.
    - If you want to preserve original labels for classification metrics, keep 'Auto label-encode' checked (we keep the encoder in memory).
    - Want one-hot encoding, pipelines, CV, hyperparameter tuning, SHAP, or model export? Say the word and I'll add.
    """)


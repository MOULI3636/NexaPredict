import warnings
from io import BytesIO

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.ensemble import (
    RandomForestClassifier,
    RandomForestRegressor,
    VotingClassifier,
    VotingRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid")

try:
    from xgboost import XGBClassifier, XGBRegressor
except Exception:
    XGBClassifier = None
    XGBRegressor = None

try:
    from catboost import CatBoostClassifier, CatBoostRegressor
except Exception:
    CatBoostClassifier = None
    CatBoostRegressor = None

try:
    import shap
except Exception:
    shap = None


st.set_page_config(
    page_title="Insight Predict",
    page_icon="data",
    layout="wide",
)


def init_state() -> None:
    if "df" not in st.session_state:
        st.session_state.df = pd.DataFrame()
    if "original_df" not in st.session_state:
        st.session_state.original_df = pd.DataFrame()


def read_uploaded_file(uploaded_file) -> pd.DataFrame:
    if uploaded_file.name.lower().endswith(".csv"):
        return pd.read_csv(uploaded_file)
    return pd.read_excel(uploaded_file)


def load_sample_dataframe() -> pd.DataFrame:
    iris = load_iris(as_frame=True)
    sample_df = iris.frame.copy()
    sample_df["species"] = sample_df["target"].map(dict(enumerate(iris.target_names)))
    sample_df = sample_df.drop(columns=["target"])
    return sample_df


def infer_problem_type(target: pd.Series) -> str:
    if pd.api.types.is_numeric_dtype(target) and target.nunique(dropna=True) > 10:
        return "regression"
    return "classification"


def encode_target(y: pd.Series, problem_type: str):
    encoder = None
    if problem_type == "classification":
        encoder = LabelEncoder()
        y = pd.Series(encoder.fit_transform(y.astype(str)), index=y.index, name=y.name)
    return y, encoder


def prepare_feature_matrix(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    X = df[features].copy()

    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = [col for col in X.columns if col not in numeric_cols]

    if numeric_cols:
        num_imputer = SimpleImputer(strategy="median")
        X[numeric_cols] = num_imputer.fit_transform(X[numeric_cols])

    if categorical_cols:
        cat_imputer = SimpleImputer(strategy="most_frequent")
        X[categorical_cols] = cat_imputer.fit_transform(X[categorical_cols])
        X = pd.get_dummies(X, columns=categorical_cols, drop_first=False)

    return X


def get_model(model_name: str, problem_type: str):
    if model_name == "Linear / Logistic Regression":
        if problem_type == "classification":
            return LogisticRegression(max_iter=2000)
        return LinearRegression()

    if model_name == "Random Forest":
        if problem_type == "classification":
            return RandomForestClassifier(random_state=42, n_estimators=200)
        return RandomForestRegressor(random_state=42, n_estimators=200)

    if model_name == "Voting Ensemble":
        if problem_type == "classification":
            estimators = [
                ("lr", LogisticRegression(max_iter=2000)),
                ("rf", RandomForestClassifier(random_state=42, n_estimators=200)),
            ]
            if XGBClassifier is not None:
                estimators.append(
                    ("xgb", XGBClassifier(random_state=42, eval_metric="mlogloss"))
                )
            voting = "soft" if len(estimators) > 1 else "hard"
            return VotingClassifier(estimators=estimators, voting=voting)

        estimators = [
            ("lr", LinearRegression()),
            ("rf", RandomForestRegressor(random_state=42, n_estimators=200)),
        ]
        if XGBRegressor is not None:
            estimators.append(("xgb", XGBRegressor(random_state=42)))
        return VotingRegressor(estimators=estimators)

    if model_name == "XGBoost":
        if XGBClassifier is None or XGBRegressor is None:
            raise ImportError("xgboost is not installed.")
        if problem_type == "classification":
            return XGBClassifier(random_state=42, eval_metric="mlogloss")
        return XGBRegressor(random_state=42)

    if model_name == "CatBoost":
        if CatBoostClassifier is None or CatBoostRegressor is None:
            raise ImportError("catboost is not installed.")
        if problem_type == "classification":
            return CatBoostClassifier(random_state=42, verbose=False)
        return CatBoostRegressor(random_state=42, verbose=False)

    raise ValueError(f"Unsupported model: {model_name}")


def draw_confusion_matrix(cm: np.ndarray) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")
    fig.tight_layout()
    return fig


def draw_regression_plot(y_true: pd.Series, y_pred: np.ndarray) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(y_true, y_pred, alpha=0.7, edgecolor="black")
    line_start = min(float(np.min(y_true)), float(np.min(y_pred)))
    line_end = max(float(np.max(y_true)), float(np.max(y_pred)))
    ax.plot([line_start, line_end], [line_start, line_end], linestyle="--", color="red")
    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    ax.set_title("Actual vs Predicted")
    fig.tight_layout()
    return fig


def draw_feature_importance(model, feature_names: list[str]) -> plt.Figure | None:
    if hasattr(model, "feature_importances_"):
        values = np.asarray(model.feature_importances_)
    elif hasattr(model, "coef_"):
        coef = np.asarray(model.coef_)
        values = np.abs(coef[0] if coef.ndim > 1 else coef)
    else:
        return None

    order = np.argsort(values)[-15:]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh(np.array(feature_names)[order], values[order], color="#2563eb")
    ax.set_title("Top Feature Importance")
    ax.set_xlabel("Importance")
    fig.tight_layout()
    return fig


def draw_correlation(df: pd.DataFrame) -> plt.Figure | None:
    numeric_df = df.select_dtypes(include=[np.number])
    if numeric_df.empty:
        return None
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(numeric_df.corr(), cmap="coolwarm", center=0, ax=ax)
    ax.set_title("Correlation Matrix")
    fig.tight_layout()
    return fig


def draw_pca(df: pd.DataFrame, n_components: int, standardize: bool):
    numeric_df = df.select_dtypes(include=[np.number]).dropna()
    if numeric_df.empty or len(numeric_df.columns) < 2:
        return None, None, None

    values = numeric_df.values
    if standardize:
        values = StandardScaler().fit_transform(values)

    pca = PCA(n_components=n_components)
    transformed = pca.fit_transform(values)

    scree_fig, scree_ax = plt.subplots(figsize=(6, 4))
    scree_ax.plot(
        range(1, len(pca.explained_variance_ratio_) + 1),
        pca.explained_variance_ratio_,
        marker="o",
    )
    scree_ax.set_title("Scree Plot")
    scree_ax.set_xlabel("Principal Component")
    scree_ax.set_ylabel("Explained Variance Ratio")
    scree_fig.tight_layout()

    scatter_fig = None
    if transformed.shape[1] >= 2:
        scatter_fig, scatter_ax = plt.subplots(figsize=(6, 4))
        scatter_ax.scatter(transformed[:, 0], transformed[:, 1], alpha=0.7)
        scatter_ax.set_title("PCA Projection")
        scatter_ax.set_xlabel("PC1")
        scatter_ax.set_ylabel("PC2")
        scatter_fig.tight_layout()

    components_df = pd.DataFrame(
        pca.components_,
        columns=numeric_df.columns,
        index=[f"PC{i + 1}" for i in range(len(pca.components_))],
    )
    return pca, scree_fig, scatter_fig, components_df


def build_shap_plot(model, X_train: pd.DataFrame, X_test: pd.DataFrame):
    if shap is None:
        raise ImportError("shap is not installed.")

    sample = X_test.head(min(len(X_test), 200))
    explainer = shap.Explainer(model, X_train)
    shap_values = explainer(sample)

    plt.figure()
    shap.summary_plot(shap_values, sample, show=False)
    fig = plt.gcf()
    fig.tight_layout()
    return fig


def dataframe_download(df: pd.DataFrame) -> bytes:
    buffer = BytesIO()
    df.to_csv(buffer, index=False)
    return buffer.getvalue()


def apply_missing_value_action(df: pd.DataFrame, action: str) -> pd.DataFrame:
    updated = df.copy()
    if action == "Drop rows with missing values":
        return updated.dropna()

    numeric_cols = updated.select_dtypes(include=[np.number]).columns.tolist()
    object_cols = [col for col in updated.columns if col not in numeric_cols]

    if action == "Fill numeric with mean":
        for col in numeric_cols:
            updated[col] = updated[col].fillna(updated[col].mean())
        for col in object_cols:
            if updated[col].isna().any():
                updated[col] = updated[col].fillna(updated[col].mode().iloc[0])

    if action == "Fill numeric with median":
        for col in numeric_cols:
            updated[col] = updated[col].fillna(updated[col].median())
        for col in object_cols:
            if updated[col].isna().any():
                updated[col] = updated[col].fillna(updated[col].mode().iloc[0])

    if action == "Fill all missing with 0":
        updated = updated.fillna(0)

    return updated


def main() -> None:
    init_state()

    st.title("Insight Predict")
    st.caption("Upload a dataset, clean it, visualize it, and run machine learning models in the browser.")

    with st.sidebar:
        st.header("Dataset")
        uploaded_file = st.file_uploader("Upload CSV or Excel", type=["csv", "xlsx"])

        if uploaded_file is not None:
            try:
                loaded_df = read_uploaded_file(uploaded_file)
                st.session_state.df = loaded_df.copy()
                st.session_state.original_df = loaded_df.copy()
                st.success(f"Loaded {uploaded_file.name}")
            except Exception as exc:
                st.error(f"Could not read file: {exc}")

        col1, col2 = st.columns(2)
        if col1.button("Reset data", use_container_width=True):
            if not st.session_state.original_df.empty:
                st.session_state.df = st.session_state.original_df.copy()
        if col2.button("Sample data", use_container_width=True):
            st.session_state.df = load_sample_dataframe()
            st.session_state.original_df = st.session_state.df.copy()

    df = st.session_state.df.copy()

    if df.empty:
        st.info("Upload a dataset from the sidebar to begin.")
        return

    with st.expander("Data preparation", expanded=True):
        prep_col1, prep_col2, prep_col3 = st.columns(3)

        missing_action = prep_col1.selectbox(
            "Missing values",
            [
                "Keep as-is",
                "Drop rows with missing values",
                "Fill numeric with mean",
                "Fill numeric with median",
                "Fill all missing with 0",
            ],
        )

        object_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
        columns_to_encode = prep_col2.multiselect("Convert text columns to category codes", object_cols)

        available_columns = df.columns.tolist()
        default_features = available_columns[:-1] if len(available_columns) > 1 else available_columns
        default_target = available_columns[-1] if available_columns else None

        if prep_col3.button("Apply prep", use_container_width=True):
            df = apply_missing_value_action(df, missing_action)
            for col in columns_to_encode:
                df[col] = df[col].astype("category").cat.codes
            st.session_state.df = df.copy()
            st.success("Preparation changes applied.")

    df = st.session_state.df.copy()

    overview_col1, overview_col2 = st.columns([2, 1])
    with overview_col1:
        st.subheader("Dataset preview")
        st.dataframe(df.head(20), use_container_width=True)
    with overview_col2:
        st.subheader("Dataset summary")
        st.write(f"Rows: {df.shape[0]}")
        st.write(f"Columns: {df.shape[1]}")
        st.write(f"Missing values: {int(df.isna().sum().sum())}")
        st.download_button(
            "Download current CSV",
            data=dataframe_download(df),
            file_name="prepared_dataset.csv",
            mime="text/csv",
            use_container_width=True,
        )

    st.subheader("Column info")
    info_df = pd.DataFrame(
        {
            "dtype": df.dtypes.astype(str),
            "missing": df.isna().sum(),
            "unique": df.nunique(dropna=False),
        }
    )
    st.dataframe(info_df, use_container_width=True)

    feature_col, target_col, mode_col = st.columns(3)
    features = feature_col.multiselect(
        "Features",
        df.columns.tolist(),
        default=default_features,
    )
    target = target_col.selectbox("Target", df.columns.tolist(), index=df.columns.get_loc(default_target))
    inferred_type = infer_problem_type(df[target].dropna())
    problem_type = mode_col.selectbox(
        "Problem type",
        ["classification", "regression"],
        index=0 if inferred_type == "classification" else 1,
    )

    if target in features:
        features = [col for col in features if col != target]

    viz_tab, model_tab, advanced_tab = st.tabs(["Visualizations", "Models", "Advanced"])

    with viz_tab:
        left, right = st.columns(2)

        with left:
            viz_type = st.selectbox(
                "Chart type",
                [
                    "Correlation Matrix",
                    "Histogram",
                    "Box Plot",
                    "Scatter Plot",
                    "Bar Chart",
                ],
            )

        with right:
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            selected_x = st.selectbox("X axis", df.columns.tolist(), index=0)
            selected_y = st.selectbox("Y axis", numeric_cols if numeric_cols else df.columns.tolist(), index=0)

        if viz_type == "Correlation Matrix":
            fig = draw_correlation(df)
            if fig is None:
                st.warning("No numeric columns available for a correlation matrix.")
            else:
                st.pyplot(fig, clear_figure=True)

        elif viz_type == "Histogram":
            fig, ax = plt.subplots(figsize=(8, 4))
            sns.histplot(df[selected_x].dropna(), kde=True, ax=ax, color="#2563eb")
            ax.set_title(f"Histogram of {selected_x}")
            fig.tight_layout()
            st.pyplot(fig, clear_figure=True)

        elif viz_type == "Box Plot":
            fig, ax = plt.subplots(figsize=(8, 4))
            sns.boxplot(x=df[selected_x], ax=ax, color="#60a5fa")
            ax.set_title(f"Box Plot of {selected_x}")
            fig.tight_layout()
            st.pyplot(fig, clear_figure=True)

        elif viz_type == "Scatter Plot":
            fig, ax = plt.subplots(figsize=(8, 4))
            if target in df.columns and problem_type == "classification":
                sns.scatterplot(data=df, x=selected_x, y=selected_y, hue=target, ax=ax)
            else:
                sns.scatterplot(data=df, x=selected_x, y=selected_y, ax=ax)
            ax.set_title(f"{selected_x} vs {selected_y}")
            fig.tight_layout()
            st.pyplot(fig, clear_figure=True)

        elif viz_type == "Bar Chart":
            counts = df[selected_x].astype(str).value_counts().head(20)
            fig, ax = plt.subplots(figsize=(8, 4))
            sns.barplot(x=counts.values, y=counts.index, ax=ax, color="#2563eb")
            ax.set_title(f"Top values in {selected_x}")
            ax.set_xlabel("Count")
            ax.set_ylabel(selected_x)
            fig.tight_layout()
            st.pyplot(fig, clear_figure=True)

    with model_tab:
        if not features:
            st.warning("Select at least one feature.")
        else:
            model_name = st.selectbox(
                "Model",
                [
                    "Linear / Logistic Regression",
                    "Random Forest",
                    "Voting Ensemble",
                    "XGBoost",
                    "CatBoost",
                ],
            )
            test_size = st.slider("Test size", 0.1, 0.4, 0.2, 0.05)

            if st.button("Train model", type="primary"):
                try:
                    model_df = df[features + [target]].dropna(subset=[target]).copy()
                    X = prepare_feature_matrix(model_df, features)
                    y, label_encoder = encode_target(model_df[target], problem_type)

                    X_train, X_test, y_train, y_test = train_test_split(
                        X,
                        y,
                        test_size=test_size,
                        random_state=42,
                        stratify=y if problem_type == "classification" else None,
                    )

                    model = get_model(model_name, problem_type)
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)

                    metric_col1, metric_col2 = st.columns(2)
                    if problem_type == "classification":
                        acc = accuracy_score(y_test, y_pred)
                        report = classification_report(y_test, y_pred, output_dict=True)
                        metric_col1.metric("Accuracy", f"{acc:.4f}")
                        metric_col2.metric("Classes", str(int(pd.Series(y).nunique())))
                        st.dataframe(pd.DataFrame(report).transpose(), use_container_width=True)
                        st.pyplot(draw_confusion_matrix(confusion_matrix(y_test, y_pred)), clear_figure=True)
                    else:
                        mse = mean_squared_error(y_test, y_pred)
                        r2 = r2_score(y_test, y_pred)
                        metric_col1.metric("MSE", f"{mse:.4f}")
                        metric_col2.metric("R2", f"{r2:.4f}")
                        st.pyplot(draw_regression_plot(y_test, y_pred), clear_figure=True)

                    importance_fig = draw_feature_importance(model, X.columns.tolist())
                    if importance_fig is not None:
                        st.pyplot(importance_fig, clear_figure=True)

                    st.session_state.last_training = {
                        "model": model,
                        "X_train": X_train,
                        "X_test": X_test,
                        "problem_type": problem_type,
                        "label_encoder": label_encoder,
                    }
                except Exception as exc:
                    st.error(f"Training failed: {exc}")

    with advanced_tab:
        adv1, adv2 = st.columns(2)

        with adv1:
            st.markdown("### PCA")
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if len(numeric_cols) < 2:
                st.info("PCA needs at least two numeric columns.")
            else:
                max_components = min(len(numeric_cols), max(2, len(df.dropna())))
                pca_components = st.slider("Components", 2, max_components, min(3, max_components))
                standardize = st.checkbox("Standardize before PCA", value=True)
                if st.button("Run PCA"):
                    result = draw_pca(df[numeric_cols], pca_components, standardize)
                    if result[0] is None:
                        st.warning("PCA could not be computed for this dataset.")
                    else:
                        pca_model, scree_fig, scatter_fig, components_df = result
                        st.write(
                            "Explained variance:",
                            [round(value, 4) for value in pca_model.explained_variance_ratio_],
                        )
                        st.pyplot(scree_fig, clear_figure=True)
                        if scatter_fig is not None:
                            st.pyplot(scatter_fig, clear_figure=True)
                        st.dataframe(components_df, use_container_width=True)

        with adv2:
            st.markdown("### SHAP")
            if "last_training" not in st.session_state:
                st.info("Train a model first to enable SHAP.")
            elif shap is None:
                st.info("Install `shap` to enable SHAP analysis.")
            else:
                if st.button("Generate SHAP summary"):
                    try:
                        payload = st.session_state.last_training
                        shap_fig = build_shap_plot(
                            payload["model"],
                            payload["X_train"],
                            payload["X_test"],
                        )
                        st.pyplot(shap_fig, clear_figure=True)
                    except Exception as exc:
                        st.error(f"SHAP failed: {exc}")


if __name__ == "__main__":
    main()

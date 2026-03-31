import io

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
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
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor


st.set_page_config(page_title="Insight Predict", layout="wide")


@st.cache_data
def load_data(uploaded_file):
    if uploaded_file.name.lower().endswith(".csv"):
        return pd.read_csv(uploaded_file)
    return pd.read_excel(uploaded_file)


def infer_problem_type(series: pd.Series) -> str:
    if series.dtype == "object" or str(series.dtype).startswith("category"):
        return "Classification"
    if series.nunique(dropna=True) <= 10:
        return "Classification"
    return "Regression"


def prepare_dataframe(
    dataframe: pd.DataFrame,
    fill_numeric: str,
    fill_categorical: str,
    drop_duplicates: bool,
    standardize_numeric: bool,
):
    df = dataframe.copy()

    if drop_duplicates:
        df = df.drop_duplicates()

    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = df.select_dtypes(exclude=np.number).columns.tolist()

    for col in numeric_cols:
        if df[col].isna().any():
            if fill_numeric == "Mean":
                df[col] = df[col].fillna(df[col].mean())
            elif fill_numeric == "Median":
                df[col] = df[col].fillna(df[col].median())
            elif fill_numeric == "Zero":
                df[col] = df[col].fillna(0)

    for col in categorical_cols:
        if df[col].isna().any():
            if fill_categorical == "Mode":
                mode_values = df[col].mode(dropna=True)
                fill_value = mode_values.iloc[0] if not mode_values.empty else "Unknown"
            else:
                fill_value = "Unknown"
            df[col] = df[col].fillna(fill_value)

    if standardize_numeric and numeric_cols:
        scaler = StandardScaler()
        df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

    return df


def encode_features(x_train, x_test):
    x_train_encoded = pd.get_dummies(x_train, drop_first=False)
    x_test_encoded = pd.get_dummies(x_test, drop_first=False)
    x_train_encoded, x_test_encoded = x_train_encoded.align(x_test_encoded, join="left", axis=1, fill_value=0)
    return x_train_encoded, x_test_encoded


def plot_confusion_matrix(y_true, y_pred):
    fig, ax = plt.subplots(figsize=(6, 4))
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")
    st.pyplot(fig)


def plot_feature_importance(model, feature_names):
    if not hasattr(model, "feature_importances_"):
        return

    importance = pd.Series(model.feature_importances_, index=feature_names).sort_values(ascending=False).head(15)
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(x=importance.values, y=importance.index, ax=ax, palette="viridis")
    ax.set_title("Top Feature Importances")
    ax.set_xlabel("Importance")
    ax.set_ylabel("Feature")
    st.pyplot(fig)


st.title("Insight Predict")
st.caption("Streamlit version of your ML data analyzer for easy upload and sharing.")

with st.sidebar:
    st.header("Upload")
    uploaded_file = st.file_uploader("Upload CSV or Excel file", type=["csv", "xlsx", "xls"])

    st.header("Preparation")
    fill_numeric = st.selectbox("Numeric missing values", ["Mean", "Median", "Zero"])
    fill_categorical = st.selectbox("Categorical missing values", ["Mode", "Unknown"])
    drop_duplicates = st.checkbox("Drop duplicate rows", value=False)
    standardize_numeric = st.checkbox("Standardize numeric columns", value=False)


if uploaded_file is None:
    st.info("Upload your dataset from the sidebar to start.")
    st.stop()

try:
    raw_df = load_data(uploaded_file)
except Exception as exc:
    st.error(f"Failed to load file: {exc}")
    st.stop()

df = prepare_dataframe(
    raw_df,
    fill_numeric=fill_numeric,
    fill_categorical=fill_categorical,
    drop_duplicates=drop_duplicates,
    standardize_numeric=standardize_numeric,
)

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["Overview", "Prepare Data", "Visualize", "Correlation", "Model"]
)

with tab1:
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Rows", df.shape[0])
    col2.metric("Columns", df.shape[1])
    col3.metric("Numeric Columns", len(df.select_dtypes(include=np.number).columns))
    col4.metric("Missing Values", int(df.isna().sum().sum()))

    st.subheader("Preview")
    st.dataframe(df.head(20), use_container_width=True)

    buffer = io.StringIO()
    df.info(buf=buffer)
    st.subheader("Dataset Info")
    st.text(buffer.getvalue())

with tab2:
    st.subheader("Prepared Dataset")
    st.write("The dataset below reflects the options selected in the sidebar.")
    st.dataframe(df, use_container_width=True)

    st.subheader("Missing Values")
    missing_df = df.isna().sum().reset_index()
    missing_df.columns = ["Column", "Missing Count"]
    st.dataframe(missing_df, use_container_width=True)

    csv_bytes = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download Prepared CSV",
        data=csv_bytes,
        file_name="prepared_dataset.csv",
        mime="text/csv",
    )

with tab3:
    st.subheader("Visualizations")
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    all_cols = df.columns.tolist()

    chart_type = st.selectbox(
        "Chart Type",
        ["Histogram", "Box Plot", "Scatter Plot", "Line Plot", "Count Plot"],
    )

    if chart_type in ["Histogram", "Box Plot", "Line Plot"] and not numeric_cols:
        st.warning("No numeric columns available for this chart type.")
    elif chart_type == "Histogram":
        col = st.selectbox("Select numeric column", numeric_cols)
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.histplot(df[col], kde=True, ax=ax, color="#2c7fb8")
        ax.set_title(f"Histogram of {col}")
        st.pyplot(fig)

    elif chart_type == "Box Plot":
        col = st.selectbox("Select numeric column", numeric_cols)
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.boxplot(y=df[col], ax=ax, color="#41b6c4")
        ax.set_title(f"Box Plot of {col}")
        st.pyplot(fig)

    elif chart_type == "Scatter Plot":
        if len(numeric_cols) < 2:
            st.warning("Need at least two numeric columns for a scatter plot.")
        else:
            x_col = st.selectbox("X-axis", numeric_cols, key="scatter_x")
            y_col = st.selectbox("Y-axis", numeric_cols, key="scatter_y", index=min(1, len(numeric_cols) - 1))
            fig, ax = plt.subplots(figsize=(8, 4))
            sns.scatterplot(data=df, x=x_col, y=y_col, ax=ax, color="#225ea8")
            ax.set_title(f"{x_col} vs {y_col}")
            st.pyplot(fig)

    elif chart_type == "Line Plot":
        col = st.selectbox("Select numeric column", numeric_cols, key="line_col")
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(df.index, df[col], color="#253494")
        ax.set_title(f"Line Plot of {col}")
        ax.set_xlabel("Index")
        ax.set_ylabel(col)
        st.pyplot(fig)

    elif chart_type == "Count Plot":
        col = st.selectbox("Select categorical column", all_cols, key="count_col")
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.countplot(data=df, x=col, ax=ax, color="#7fcdbb")
        ax.set_title(f"Count Plot of {col}")
        ax.tick_params(axis="x", rotation=45)
        st.pyplot(fig)

with tab4:
    st.subheader("Correlation Matrix")
    numeric_df = df.select_dtypes(include=np.number)
    if numeric_df.empty:
        st.warning("No numeric columns available for correlation analysis.")
    else:
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
        ax.set_title("Correlation Heatmap")
        st.pyplot(fig)

with tab5:
    st.subheader("Model Training")
    if df.shape[1] < 2:
        st.warning("Need at least two columns to train a model.")
    else:
        target_col = st.selectbox("Target column", df.columns.tolist(), index=len(df.columns) - 1)
        feature_cols = st.multiselect(
            "Feature columns",
            [col for col in df.columns if col != target_col],
            default=[col for col in df.columns if col != target_col],
        )

        if not feature_cols:
            st.warning("Select at least one feature column.")
        else:
            problem_type = infer_problem_type(df[target_col])
            st.write(f"Detected problem type: **{problem_type}**")

            if problem_type == "Classification":
                model_name = st.selectbox(
                    "Choose model",
                    ["Logistic Regression", "Decision Tree", "Random Forest"],
                )
            else:
                model_name = st.selectbox(
                    "Choose model",
                    ["Linear Regression", "Decision Tree Regressor", "Random Forest Regressor"],
                )

            test_size = st.slider("Test size", min_value=0.1, max_value=0.4, value=0.2, step=0.05)

            if st.button("Train Model", type="primary"):
                model_df = df[feature_cols + [target_col]].dropna().copy()
                x = model_df[feature_cols]
                y = model_df[target_col]

                if problem_type == "Classification":
                    label_encoder = LabelEncoder()
                    y = label_encoder.fit_transform(y.astype(str))

                x_train, x_test, y_train, y_test = train_test_split(
                    x,
                    y,
                    test_size=test_size,
                    random_state=42,
                )

                x_train_encoded, x_test_encoded = encode_features(x_train, x_test)

                if model_name == "Logistic Regression":
                    model = LogisticRegression(max_iter=1000)
                elif model_name == "Decision Tree":
                    model = DecisionTreeClassifier(random_state=42)
                elif model_name == "Random Forest":
                    model = RandomForestClassifier(random_state=42)
                elif model_name == "Linear Regression":
                    model = LinearRegression()
                elif model_name == "Decision Tree Regressor":
                    model = DecisionTreeRegressor(random_state=42)
                else:
                    model = RandomForestRegressor(random_state=42)

                model.fit(x_train_encoded, y_train)
                predictions = model.predict(x_test_encoded)

                if problem_type == "Classification":
                    st.metric("Accuracy", f"{accuracy_score(y_test, predictions):.4f}")
                    st.text("Classification Report")
                    st.text(classification_report(y_test, predictions))
                    plot_confusion_matrix(y_test, predictions)
                else:
                    rmse = mean_squared_error(y_test, predictions) ** 0.5
                    st.metric("RMSE", f"{rmse:.4f}")
                    st.metric("R2 Score", f"{r2_score(y_test, predictions):.4f}")

                plot_feature_importance(model, x_train_encoded.columns)

# Insight Predict Streamlit App

This project contains a Streamlit version of the original desktop ML dashboard.

## Main file

`streamlit_app.py`

## Local run

```powershell
python -m pip install -r requirements.txt
streamlit run streamlit_app.py
```

## Optional advanced features

Install these only if you want the heavier model extras:

```powershell
python -m pip install -r requirements-advanced.txt
```

This enables:

- `XGBoost`
- `CatBoost`
- `SHAP`

## Streamlit Community Cloud

1. Push this folder to GitHub.
2. Open Streamlit Community Cloud.
3. Create a new app from your repository.
4. Set the main file path to `streamlit_app.py`.
5. Deploy.

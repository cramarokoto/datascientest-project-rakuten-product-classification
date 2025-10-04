# Rakuten E-commerce Product Classification

This repository contains the code and app for a Rakuten product classification project developed during the DataScientest training.

## Environment (Conda, Python 3.11)

Create and activate a dedicated environment:

```bash
conda create -n py311 python=3.11 -y
conda activate py311
```

Install dependencies from the root requirements file:

```bash
pip install -r requirements.txt
```

## Notebooks

- Place notebooks under `notebooks/`.
- Ensure the environment is activated before running Jupyter or IPython.

```bash
ipython
```

## Streamlit App

The Streamlit app lives independently in `streamlit_app/`.

Run it with its own environment:

```bash
cd streamlit_app
conda create -n streamlit-app python=3.11 -y
conda activate streamlit-app
streamlit run app.py
```

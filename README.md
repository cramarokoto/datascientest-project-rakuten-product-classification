# Rakuten E-commerce Product Classification

This repository contains the code and app for a product classification project developed during the DataScientest training.

## Environment (Conda, Python 3.11)

Create and activate a dedicated environment:

```bash
conda create -n streamlit-app python=3.11 -y
conda activate streamlit-app
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

The Streamlit app lives in `streamlit_app/`.

Run it with the same environment:

```bash
streamlit run streamlit_app/app.py
```

If you deploy or work with separate environments, prefer a single dependency source (the root `requirements.txt`). The legacy `streamlit_app/requirements.txt` pins very old versions and can be removed or ignored.

## Project Structure

- `requirements.txt`: project-wide dependencies
- `streamlit_app/`: Streamlit application (tabs, assets, config)
- `data/`: datasets (git-ignored if large/sensitive)
- `notebooks/`: analysis and experiments

## Team

Add your team members in `streamlit_app/config.py` and update this section with names and links.

# Streamlit App

This directory contains the Streamlit UI for the Rakuten product classification project.

## Quickstart (Conda, Python 3.11)

Create and activate the environment at the project root (recommended):

```bash
conda create -n streamlit-app python=3.11 -y
conda activate streamlit-app
pip install -r ../requirements.txt
```

Run the app from the project root or from this directory:

```bash
# from project root
streamlit run streamlit_app/app.py

# or from this directory
streamlit run app.py
```

## Configuration

- Edit `config.py` to set `TITLE`, `PROMOTION`, and `TEAM_MEMBERS`.
- Team member cards are defined in `member.py`.
- Tabs are in `tabs/` (`intro.py`, `second_tab.py`, `third_tab.py`). Add new tabs and register them in `app.py`.

## Assets and Styling

- Static assets live under `assets/`.
- Global styles are in `style.css` and injected by `app.py`.

## Dependencies

Prefer using the root `requirements.txt` for a single source of dependencies. The `requirements.txt` in this folder was part of the template and pins old versions; it can be removed or ignored if you standardize on the root file.

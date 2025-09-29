import streamlit as st


title = "Interprétabilité"
sidebar_name = "Interprétabilité"


def run():
    st.image("assets/analysis.svg")

    st.title(title)
    st.markdown("---")
    st.write("Expliquez ici les méthodes d'interprétabilité (SHAP, LIME, etc.).")

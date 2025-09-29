import streamlit as st


title = "Modélisation"
sidebar_name = "Modélisation"


def run():
    st.image("assets/processing.svg")

    st.title(title)
    st.markdown("---")
    st.write("Présentez ici l'entraînement, l'évaluation et la comparaison des modèles.")

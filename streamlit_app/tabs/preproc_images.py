import streamlit as st


title = "Préprocessing des images"
sidebar_name = "Préprocessing des images"


def run():
    st.image("assets/processing.svg")

    st.title(title)
    st.markdown("---")
    st.write("Décrivez et appliquez ici vos étapes de prétraitement d'images.")

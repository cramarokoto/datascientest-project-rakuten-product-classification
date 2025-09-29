import streamlit as st


title = "Conclusion"
sidebar_name = "Conclusion"


def run():
    st.image("assets/analysis.svg")

    st.title(title)
    st.markdown("---")
    st.write("Résumez ici vos résultats, limites et pistes de travail.")

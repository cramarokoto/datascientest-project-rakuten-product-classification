import streamlit as st


title = "Exploration des données"
sidebar_name = "Exploration des données"


def run():
    # Fit image to the zone with flexbox
    st.image("assets/statistics.svg", width="stretch")

    st.title(title)
    st.markdown("---")
    st.write("Ajoutez ici vos visualisations et statistiques descriptives.")

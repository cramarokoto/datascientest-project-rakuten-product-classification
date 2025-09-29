import streamlit as st


title = "Prétraitement"
sidebar_name = "Prétraitement"


def run():
    st.image("assets/processing.svg")
    st.title(title)

    tab1, tab2 = st.tabs(["Prétraitement des données textuelles", "Prétraitement des données graphiques"])

    with tab1:
        st.write("Prétraitement des données textuelles")

    with tab2:
        st.write("Prétraitement des données graphiques")

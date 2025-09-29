import streamlit as st
import config


title = "Équipe"
sidebar_name = "Équipe"


def run():
    st.image("assets/team.svg")

    st.title(title)
    st.markdown("---")
    st.subheader("Membres de l'équipe")
    for member in config.TEAM_MEMBERS:
        st.markdown(member.sidebar_markdown(), unsafe_allow_html=True)

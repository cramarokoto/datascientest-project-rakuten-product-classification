import streamlit as st


title = "Introduction"
sidebar_name = "Introduction"


def run():

    st.image("assets/shopping.svg")

    st.title(title)

    st.markdown("---")

    st.header("Présentation du projet")
    st.write("""
    Ce projet propose une classification automatique des produits du catalogue Rakuten France, 
    utilisant à la fois les données textuelles (titre et description) et les images fournies pour chaque article.
    """)

    st.header("Objectif")
    st.write("""
    L'enjeu est de prédire le type de chaque produit parmi de nombreuses catégories, 
    en relevant des défis tels que la diversité, le bruit des données réelles de e-commerce, 
    et le fort déséquilibre des classes. Nous avons expérimenté différentes approches de machine learning pour résoudre ce problème en exploitant l'ensemble des données disponibles.
    """)

    st.header("En savoir plus")
    cols = st.columns([1, 3])
    with cols[0]:
        st.image("assets/RIT_logo_big.jpg")
    with cols[1]:
        st.markdown("""
        Pour consulter le détail du challenge et du jeu de données :  
        [Rakuten France Multimodal Product Data Classification — ENS Challenge](https://challengedata.ens.fr/challenges/35)
        """)

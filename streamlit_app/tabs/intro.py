import streamlit as st


title = "Introduction"
sidebar_name = "Introduction"


def run():

    st.image("assets/shopping.svg")

    st.title(title)

    st.markdown("---")

    st.header("Présentation du projet")
    st.write("""
    Ce projet propose une application de classification automatique des produits du catalogue Rakuten France, 
    utilisant à la fois les données textuelles (titre et description) et les images fournies pour chaque article.
    """)

    st.header("Objectif")
    st.write("""
    L’enjeu est de prédire le type de chaque produit parmi de nombreuses catégories, 
    en relevant des défis tels que la diversité, le bruit des données réelles de e-commerce, 
    et la forte déséquilibre des classes. La solution s’appuie sur des modèles de deep learning 
    pour exploiter l’ensemble des données disponibles et optimiser l’organisation du catalogue.
    """)

    st.header("En savoir plus")
    st.markdown("""
    Pour consulter le détail du challenge et du jeu de données :  
    [Rakuten France Multimodal Product Data Classification — ENS Challenge](https://challengedata.ens.fr/challenges/35)
    """)

import streamlit as st


title = "Interprétabilité"
sidebar_name = "Interprétabilité"


def run():
    st.image("assets/analysis.svg")

    st.title(title)
    
    tab1, tab2 = st.tabs(["Modèles de classification des textes", "Modèles de classification des images"])

    with tab1:
        st.markdown("#### Préprocessing avec SVD")
        st.markdown("##### Logistic Regression")
        st.markdown("##### Random Forest")
        st.markdown("##### XGBoost")

        st.markdown("#### Préprocessing avec SVD")
        st.markdown("##### Logistic Regression")
        st.markdown("##### Random Forest")
        st.markdown("##### XGBoost")
    with tab2:
        st.markdown("#### Grad cam du fine tuned Resnet")
        st.markdown(
            """
            Afin de vérifier la capacité du modèle à classifier les images en utilisant des données utiles, nous étudions l'interprétabilité du modèle via un grad cam.

            Nous observons les limitations du modèle :
            - certains objets sont reconnus en tant que tel mais leur catégorie est mauvaise car ils peuvent appartenir à plusieurs catégories naturellement
            - certains objets sont mal représentés dans les images (par exemple lot de cartouches de jeu trop dense)
            - certaines catégories se basent plus sur l'environnement que sur l'objet (par exemple pour les articles de jardinage), ce qui n'est pas stable et forcément discriminant
            """
        )
        st.image("assets/grad_cam/gc1.png")
        st.image("assets/grad_cam/gc2.png")
        st.image("assets/grad_cam/gc3.png")
        st.image("assets/grad_cam/gc4.png")
        st.image("assets/grad_cam/gc5.png")
        st.image("assets/grad_cam/gc6.png")
        st.image("assets/grad_cam/gc7.png")
        st.image("assets/grad_cam/gc8.png")
        st.image("assets/grad_cam/gc9.png")
        st.image("assets/grad_cam/gc10.png")
        st.image("assets/grad_cam/gc11.png")
        st.image("assets/grad_cam/gc12.png")
        st.image("assets/grad_cam/gc13.png")
        st.image("assets/grad_cam/gc14.png")
        st.image("assets/grad_cam/gc15.png")
        st.image("assets/grad_cam/gc16.png")
        st.image("assets/grad_cam/gc17.png")
        st.image("assets/grad_cam/gc18.png")
        st.image("assets/grad_cam/gc19.png")
        st.image("assets/grad_cam/gc20.png")
        st.image("assets/grad_cam/gc21.png")
        st.image("assets/grad_cam/gc22.png")
        st.image("assets/grad_cam/gc23.png")
        st.image("assets/grad_cam/gc24.png")
        st.image("assets/grad_cam/gc25.png")
        st.image("assets/grad_cam/gc26.png")
        st.image("assets/grad_cam/gc27.png")



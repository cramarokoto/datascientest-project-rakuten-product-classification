import streamlit as st


title = "Interprétabilité"
sidebar_name = "Interprétabilité"


def run():
    st.image("assets/analysis.svg")

    st.title(title)
    
    tab1, tab2 = st.tabs(["Modèles de classification des textes", "Modèles de classification des images"])

    with tab1:
        st.markdown("#### Préprocessing avec SVD")
        st.markdown(
           """
            Afin de mieux comprendre les prédictions effectuées par les modèles et d’obtenir une vision plus précise de l’importance des variables, nous avons appliqué SHapley Additive exPlanations (SHAP) sur trois modèles utilisés pour la classification de textes.
            
            Pour les modèles entrainés sur toutes les données d’entrainement, nous avons rencontré une difficulté : le prétraitement des données avec SVD a conduit à une perte d’information sur la vectorisation TF-IDF, et donc sur l’interprétation de l’importance des mots utilisés dans la classification des articles.

           """ 
        )
        st.markdown("##### Logistic Regression")
        st.markdown(
            """
            Summary plot - classe avec le score F1 le plus élevé (2905 - jeux en téléchargement):
            """
        )
        st.image("assets/shap/svd/lr_sum_plot.png")

        st.markdown("##### Random Forest")
        st.markdown(
            """
            Summary plot - classe avec le score F1 le plus élevé (2905 - jeux en téléchargement):
            """
        )
        st.image("assets/shap/svd/rfc_sum_plot.png", width=500)

        st.markdown("##### XGBoost")
        st.markdown(
            """
            Summary plot - classe avec le score F1 le plus élevé (2905 - jeux en téléchargement):
            """
        )
        st.image("assets/shap/svd/xgb_sum_plot.png", width=500)

        st.markdown("#### Préprocessing sans SVD")
        st.markdown(
           """
            Afin d’obtenir une meilleure compréhension de la contribution des mots à la sortie de classification, nous avons entraîné les trois modèles en utilisant un ensemble des données qui :
            -	sont prétraités sans réduction de dimension avec SVD
            -	contient 2000 articles pour l’ensemble d’entraînement et de test
            
            avant de poursuivre l’analyse de SHAP.

           """ 
        )
        st.markdown("##### Logistic Regression")

        st.markdown("""
        <p style='font-size:13px; line-height:1.2'>
        <span>La classe prédite : 60 - consoles de jeux video</span><br>
        <span>La classe réelle : 60 - consoles de jeux video</span>
        </p>
        """, unsafe_allow_html=True)
        st.image("assets/shap/tfidf/lr_forceplot_1.png", width=700)

        st.markdown("""
        <p style='font-size:13px; line-height:1.2'>
        <span>La classe prédite : 1281 - jeux de société</span><br>
        <span>La classe réelle : 1320 - accessoires petite enfance</span>
        </p>
        """, unsafe_allow_html=True)
        st.image("assets/shap/tfidf/lr_forceplot_2.png", width=700)

        st.markdown(
            """
            Summary plot - classe avec le score F1 le plus élevé (1301 - accesoires et jeux pour petits enfants/bébés):
            """
        )
        st.image("assets/shap/tfidf/lr_sumplot.png", width=500)

        st.markdown("##### Random Forest") 
        st.markdown("""
        <p style='font-size:13px; line-height:1.2'>
        <span>La classe prédite : 60 - consoles de jeux video</span><br>
        <span>La classe réelle : 60 - consoles de jeux video</span>
        </p>
        """, unsafe_allow_html=True)
        st.image("assets/shap/tfidf/rfc_forceplot_1.png", width=700)

        st.markdown("""
        <p style='font-size:13px; line-height:1.2'>
        <span>La classe prédite : 1940 - nourriture</span><br>
        <span>La classe réelle : 1320 - accessoires petite enfance</span>
        </p>
        """, unsafe_allow_html=True)
        st.image("assets/shap/tfidf/rfc_forceplot_2.png", width=700)

        st.markdown(
            """
            Summary plot - classe avec le score F1 le plus élevé (2905 - jeux en téléchargement):
            """
        )
        st.image("assets/shap/tfidf/rfc_sumplot.png", width=500)

        st.markdown("##### XGBoost")
        st.markdown("""
        <p style='font-size:13px; line-height:1.2'>
        <span>La classe prédite : 1160 - cartes collectionables</span><br>
        <span>La classe réelle : 1160 - cartes collectionables</span>
        </p>
        """, unsafe_allow_html=True)
        st.image("assets/shap/tfidf/xgb_forceplot_1.png", width=700)

        st.markdown("""
        <p style='font-size:13px; line-height:1.2'>
        <span>La classe prédite : 1301 - accesoires et jeux pour petits enfants/bébés</span><br>
        <span>La classe réelle : 1300 - voitures miniatures et maquettes</span>
        </p>
        """, unsafe_allow_html=True)
        st.image("assets/shap/tfidf/xgb_forceplot_2.png", width=700)

        st.markdown(
            """
            Summary plot - classe avec le score F1 le plus élevé (2905 - jeux en téléchargement):
            """
        )
        st.image("assets/shap/tfidf/xgb_sumplot.png", width=500)

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



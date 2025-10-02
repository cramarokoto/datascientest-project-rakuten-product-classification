import streamlit as st
import pandas as pd


title = "Prétraitement"
sidebar_name = "Prétraitement"


def run():
    st.image("assets/processing.svg")
    st.title(title)

    tab1, tab2, tab3 = st.tabs(["Séparation des données", "Prétraitement des données textuelles", "Prétraitement des données graphiques"])

    with tab1:
        st.markdown("#### Chargement des données et séparation des features et target")

        st.markdown("""
        Avant tout préprocessing, on établit la séparation des données de train et de test afin de limiter la possibilité de data leak entre les jeux de données de train et de test.  
        Nous avons choisi le ratio standard suivant : **80% de train, 20% de test**.
        """)

        st.markdown("""
        On sait que la classe **2583 (accessoires de piscines)** est sur représentée dans le jeu de données.  
        Nous avons décidé de :
        - faire de l'undersampling pour diminuer la présence de la classe majoritaire
        - faire de l'oversampling aléatoire pour augmenter la présence des classes minoritaires

        Cela permet d'équilibrer toutes les classes à **3.7%** et d'améliorer les performances de l'apprentissage.

        ➡️ Le jeu de test, lui, reste inchangé et déséquilibré comme dans la réalité.
        """)

        st.markdown("#### Proportion de chaque classe dans les jeux de données")

        # Données des classes dans un DataFrame pour st.dataframe scrollable
        data = {
            "Classe": [2583, 1560, 1300, 2060, 2522, 1280, 2403, 2280, 1920, 1160, 1320, 10, 2705, 1140, 2582, 40, 2585, 1302, 1281, 50, 2462, 2905, 60, 2220, 1301, 1940, 1180],
            "Train avant resampling": [12.022316, 5.973621, 5.941235, 5.879409, 5.874993, 5.735147, 5.621798, 5.605606, 5.066832, 4.654655, 3.817052, 3.669846, 3.251781, 3.145793, 3.048637, 2.952953, 2.939704, 2.933816, 2.437732, 1.979921, 1.673733, 1.027498, 0.980392, 0.970088, 0.950951, 0.945063, 0.899429],
            "Train après resampling": [3.703704]*27,
            "Test": [12.023081, 5.976213, 5.940886, 5.882007, 5.876119, 5.734809, 5.622939, 5.605276, 5.069477, 4.657325, 3.815356, 3.668158, 3.250118, 3.144136, 3.049929, 2.955723, 2.938059, 2.932171, 2.437588, 1.978333, 1.672162, 1.024494, 0.977390, 0.971503, 0.947951, 0.947951, 0.900848]
        }
        df_classes = pd.DataFrame(data)

        st.dataframe(df_classes, height=250)

    with tab2:
        st.markdown("#### Création de la variable `has_description`")
        st.markdown("""
        La présence ou absence de description n'est pas homogène parmi les catégories de produits.  
        Nous créons donc une variable `has_description` :
        - **1** si la description est présente
        - **0** sinon
        """)

        st.markdown("#### Fusion des variables `designation` et `description` en `full_description`")

        st.markdown("""
        Ces deux champs textuels apportent la même information sémantique.  
        Comme `description` peut être nulle, nous les fusionnons dans une variable `full_description`.
        """)

        st.markdown("#### Nettoyage, tokenisation et prétraitement de `full_description`")
        st.markdown("""
        Étapes prévues :
        - Suppression des stop words (FR & EN)
        - Suppression ponctuation & balises HTML
        - Tokenisation
        - (abandon de la stemmatisation et lemmatisation pour limiter la perte d'information multilingue)
        """)

        st.markdown("#### Suppression des variables à faible variance")
        st.markdown("Nous supprimons `productid` et `imageid`.")

        st.markdown("#### Application de TF-IDF")
        st.markdown("""
        Choix :
        - 10 000 features
        - n-gram (1,1)
        - min 10 documents
        - réduction dimensionnelle par **TruncatedSVD (200 composantes)**
        """)

        st.markdown("#### Sauvegarde")
        st.markdown("Les données textuelles prétraitées sont sauvegardées au format **pkl** via joblib.")

    with tab3:
        st.markdown("""
        Les différents modèles prenant en entrée des formats d'images différents, nous avons adapté le prétraitement pour chaque modèle.
                    
        On considère la Régression logistique, un CNN personnalisé avec Keras et un CNN fine-tuned (Resnet18).
        """)
    
        st.markdown("#### Transformation des images")
        st.markdown("""
        **Régression Logistique :**
        - Passage de **500×500 en couleurs** ➝ **200×200 en niveaux de gris**  
        - Réduction mémoire : 750 000 valeurs ➝ 40 000 valeurs par image
        - Perte d'information (couleur, détails) compensée par gain en performance et coût calcul
                    
        **CNN personnalisé :**
        - Passage de **500×500 en couleurs** ➝ **224×224 en niveaux de gris**
                    
        **Resnet18 :**
        - Passage de **500×500 en couleurs** ➝ **224×224 en couleurs**
        """)

        st.markdown("#### Alignement avec les données textuelles")
        st.markdown("""
        Le **même rééquilibrage (oversampling/undersampling)** est appliqué aux images pour assurer une cohérence multimodale.
        """)

        st.markdown("#### Encodage des classes cibles")
        st.markdown("Classes encodées de **0 à n-1** comme pour les données textuelles.")

        st.markdown("#### Découpage du jeu de données")
        st.markdown("""
        Même séparation train/test que pour les données textuelles afin de garantir la validité des comparaisons.
        """)

        st.markdown("#### Réduction de dimension")
        st.markdown("""
        Régression Logistique :
        - **PCA incrémental**: 40 000 features ➝ 256 features
        """)

        st.markdown("#### Conclusion")
        st.markdown("""
        Le preprocessing des images et textes vise à :
        - Réduire la complexité computationnelle 
        - Aligner données textuelles et visuelles
        - Préserver autant que possible l’information utile
        """)
 
        st.markdown("#### Alternatives envisagées")
        st.markdown("""
        **Régression Logistique :**
        - Conserver les couleurs 
        - Appliquer un PCA sur des données non réduites
        - Comparer **info en gris avec plus de pixels** vs **info en couleurs avec moins de pixels**
                    
        **CNN personnalisé :**
        - Comparer **info en gris** vs **info en couleurs**
        """)
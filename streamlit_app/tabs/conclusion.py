import streamlit as st


title = "Conclusion"
sidebar_name = "Conclusion"


def run():
    st.image("assets/analysis.svg")

    st.title(title)
    st.markdown("---")
    st.markdown(
        """
        #### Résultats
        - Classement optimal avec **textes** (XGBoost) grâce aux désignations riches.
        - **Images seules** (ResNet) insuffisantes → similarité visuelle + manque de détails.
        - Pas d'amélioration par fusion → complémentarité faible - texte plus informatif

        #### Limites techniques
        - Ressources limitées (RAM/GPU) → temps d’entraînement longs.
        - Tests et exploration d’hyperparamètres restreints → compromis performance/faisabilité.

        #### Perspectives
        - **Modèle multimodal end-to-end** (texte + image, type Transformers) :
            - Croisement des signaux visuels et sémantiques
            - Robustesse face aux données manquantes ou bruitées
            - Meilleure gestion des catégories ambiguës et nouveaux produits
        """
    )

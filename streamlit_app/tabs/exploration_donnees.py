import streamlit as st

title = "Exploration des données"
sidebar_name = "Exploration des données"

def run():
    st.image("assets/statistics.svg")
    st.title(title)

    tab1, tab2, tab3 = st.tabs(["Description globale", "Visualisation des données textuelles", "Visualisation des données graphiques"])

    with tab1:
        st.markdown("""
        Les jeux de données fournis par le challenge sont les suivants :
        - un fichier csv `X_train` contient toutes les données que nous avons pour l'entraînement (84916 entrées) ET les tests : elles sont associées aux valeurs de `y_train`.
        - un fichier csv `y_train` contient les valeurs de la classification réelle pour l'entraînement.
        - un fichier csv `X_test` contient uniquement les données à classifier avec notre modèle afin d'être évalué dans le cadre du challenge Rakuten.
        - une archive `images` : chaque entrée de `X_train` et `X_test` est associée à une image.
        """)

        st.markdown("## Description de X_train")
        # Description synthétique des jeux de données (d'après le notebook EDA)
        st.markdown(
            """
            | Nom de la colonne | Description                                                                                                       | Disponibilité | Type informatique | Taux de NA | Distribution des valeurs                                       |
            |-------------------|-------------------------------------------------------------------------------------------------------------------|---------------|-------------------|------------|----------------------------------------------------------------|
            | designation       | L’appelation du produit : comme une petite description qui donne l’essentiel sur le produit et donc sa catégorie. | Toujours      | chaîne            | 0,00 %     | Chaîne ni quantitative ni catégorielle : variable descriptive. |
            | description       | Description plus détaillée du produit, de son état, de son utilisation. Regorge d’informations et de mots clés.   | Optionnelle   | chaîne            | 35.09%     | Chaîne ni quantitative ni catégorielle : variable descriptive. |
            | productid         | L’index du produit                                                                                                | Toujours      | int64             | 0,00 %     | /                                                              |
            | imageid           | L’index de l’image                                                                                                | Toujours      | int64             | 0,00 %     | /                                                              |
            """
        )

        st.markdown("## Description de y_train")
        st.markdown(
            """
            | Nom de la colonne | Description          | Disponibilité | Type informatique | Taux de NA | Distribution des valeurs    |
            |-------------------|----------------------|---------------|-------------------|------------|-----------------------------|
            | prdtypecode       | Catégorie du produit | Toujours      | numérique         | 0,00 %     | Variable catégorielle cible |
            """
        )

        st.markdown("## Description des images")
        st.markdown(
            """
            Les images sont stockées dans l'archive `images` dans des dossiers `image_train` et `image_test`.
            Elles sont nommées avec le pattern `image_<imageid>_product_<productid>.jpg`, ce qui permet de les lier aux données textuelles.
            Elles sont en couleur et de taille 500 × 500 pixels. 
            """
        )

    with tab2:
        st.write("Visualisation des données textuelles")
    with tab3:
        st.write("Visualisation des données graphiques")
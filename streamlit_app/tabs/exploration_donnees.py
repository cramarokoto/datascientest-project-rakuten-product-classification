import streamlit as st
import pandas as pd
from PIL import Image, ImageOps

title = "Exploration des données"
sidebar_name = "Exploration des données"

@st.cache_data
def load_data():
    """
    Load the selected data from the csv files
    """
    X_train = pd.read_csv("data/selected_X_train.csv")
    y_train = pd.read_csv("data/selected_y_train.csv")
    return X_train, y_train

def show_image_with_gray_border(path, caption=None):
    image = Image.open(path)
    bordered_image = ImageOps.expand(image, border=2, fill="gray")
    st.image(bordered_image, caption=caption)

def run():
    st.image("assets/statistics.svg")
    st.title(title)

    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Description", "Visualisation des données textuelles", "Visualisation des données graphiques", "Méthodologie", "Échantillon"])

    with tab1:
        st.markdown("""
        Les jeux de données fournis par le challenge sont les suivants :
        - un fichier csv `X_train` contient toutes les données que nous avons pour l'entraînement (84916 entrées) ET les tests : elles sont associées aux valeurs de `y_train`.
        - un fichier csv `y_train` contient les valeurs de la classification réelle pour l'entraînement.
        - un fichier csv `X_test` contient uniquement les données à classifier avec notre modèle afin d'être évalué dans le cadre du challenge Rakuten.
        - une archive `images` : chaque entrée de `X_train` et `X_test` est associée à une image.
        """)

        st.markdown("#### Description de X_train")
        # Description synthétique des jeux de données (d'après le notebook EDA)
        st.markdown(
            """
            | Nom de la colonne | Description                                                                                                       | Disponibilité | Type informatique | Taux de NA | Distribution des valeurs                                       |
            |-------------------|-------------------------------------------------------------------------------------------------------------------|---------------|-------------------|------------|----------------------------------------------------------------|
            | designation       | L'appelation du produit : comme une petite description qui donne l'essentiel sur le produit et donc sa catégorie. | Toujours      | chaîne            | 0,00 %     | Chaîne ni quantitative ni catégorielle : variable descriptive. |
            | description       | Description plus détaillée du produit, de son état, de son utilisation. Regorge d'informations et de mots clés.   | Optionnelle   | chaîne            | 35.09%     | Chaîne ni quantitative ni catégorielle : variable descriptive. |
            | productid         | L'index du produit                                                                                                | Toujours      | int64             | 0,00 %     | /                                                              |
            | imageid           | L'index de l'image                                                                                                | Toujours      | int64             | 0,00 %     | /                                                              |
            """
        )

        st.markdown("#### Description des images")
        st.markdown(
            """
            Les images sont stockées dans l'archive `images` dans des dossiers `image_train` et `image_test`.
            Elles sont nommées avec le pattern `image_<imageid>_product_<productid>.jpg`, ce qui permet de les lier aux données textuelles.
            Elles sont en couleur et de taille 500 × 500 pixels. 
            """
        )

        st.markdown("#### Description de y_train")
        st.markdown(
            """
            | Nom de la colonne | Description          | Disponibilité | Type informatique | Taux de NA | Distribution des valeurs    |
            |-------------------|----------------------|---------------|-------------------|------------|-----------------------------|
            | prdtypecode       | Catégorie du produit | Toujours      | numérique         | 0,00 %     | Variable catégorielle cible |
            """
        )
        st.markdown(
            """
            Il existe 27 classes de produits distinctes dans le jeu de données d'apprentissage.
            Pour faciliter la compréhension des données, nous avons décidé de les nommer nous même.

            | prdtypecode |                                              label |
            |------------:|---------------------------------------------------:|
            |          10 |                                             livres |
            |          40 |                                         jeux video |
            |          50 |                              accesoires jeux video |
            |          60 |                             consoles de jeux video |
            |        1140 |                    produits derivés de jeux vidéos |
            |        1160 |                             cartes collectionables |
            |        1180 |     figurines collectionnables pour jeu de société |
            |        1280 |                          jouets, peluches, poupées |
            |        1281 |                                    jeux de société |
            |        1300 |                   voitures miniatures et maquettes |
            |        1301 |       accesoires et jeux pour petits enfants/bébés |
            |        1302 |                                   jeux d'exterieur |
            |        1320 |                         accessoires petite enfance |
            |        1560 |               mobilier de rangement pour la maison |
            |        1920 |                                    linge de maison |
            |        1940 |                                         nourriture |
            |        2060 |               accesoires de décoration pour maison |
            |        2220 |              accessoires pour animaux de compagnie |
            |        2280 |                                          magazines |
            |        2403 |                                      livres et bds |
            |        2462 |   consoles de jeux vidéo et jeux vidéos d'occasion |
            |        2522 |          produits de papeterie et rangement bureau |
            |        2582 |                mobilier d'extérieur et accessoires |
            |        2583 |                             accessoires de piscine |
            |        2585 | outillage et accessoires pour travaux ou jardinage |
            |        2705 |                                             livres |
            |        2905 |                             jeux en téléchargement |
            """
        )
        st.image("assets/prdtypecode_distribution.png")
        st.markdown(
            """
            Le jeu de données n'est pas équilibré :
            - la classe 2583 (accessoires de piscine) est plus représentée que les autres avec 12% du jeu de données
            - les classes 1560 (mobilier de rangement pour la maison), 1300 (voitures miniatures et maquettes), 2060 (accesoires de décoration pour maison), 2522 (produits de papeterie et rangement bureau), 1280 (jouets, peluches, poupées), 2403 (livres et bds), 2280 (magazines) suivent ensuite avec 6% chacune seulement
            - les classes les moins représentées 2905 (jeux en téléchargement), 60 (consoles de jeux video), 2220 (accessoires pour animaux de compagnie), 1301 (accesoires et jeux pour petits enfants/bébés), 1940 (nourriture), 1180 (figurines collectionnables pour jeu de société) représentent quant à elles 1% chacune
            """
        )

    with tab2:
        st.markdown("#### Analyse de la variable designation")
        st.markdown("##### Longueur de la variable designation")
        cols = st.columns([3, 1])
        with cols[0]:
            st.image("assets/designation_length.png")
        with cols[1]:
            st.write(
                """
                | count | 84916        |
                |-------|--------------|
                | mean  | 70.16        |
                | std   | 36.79        |
                | min   | 11.00        |
                | 25%   | 43.00        |
                | 50%   | 64.00        |
                | 75%   | 90.00        |
                | max   | 250.00       |
                """
            )
        st.markdown(
            """
            Nous observons que la longueur de designation est très variable avec un min de 11 caractères et un max de 250 caractères.
            Sa longueur moyenne est de 70 caractères et la médiane se situe à 64 caractères.
            D'après le boxplot, il existe beaucoup d'outliers au dela de 160 caractères.
            """
        )
        st.markdown("##### Distribution de la longueur de designation par catégorie de produit")
        st.image("assets/designation_per_category.png")
        st.markdown(
            """
            La distribution de la longueur de Designation par catégorie de produit n'est pas homogène.
            """
        )

        st.markdown("#### Analyse de la variable description")
        st.markdown("##### Longueur de la variable description")
        cols = st.columns([3, 1])
        with cols[0]:
            st.image("assets/description_length.png")
        with cols[1]:
            st.write(
                """
                | count | 55116        |
                |-------|--------------|
                | mean  | 808.17       |
                | std   | 805.46       |
                | min   | 1.00         |
                | 25%   | 259.00       |
                | 50%   | 626.00       |
                | 75%   | 1034.00      |
                | max   | 12451.00     |
                """
            )
        st.markdown(
            """
            Nous observons que la longueur de description est très variable avec un min de 1 caractères et un max de 12451 caractères.
            Sa longueur moyenne est de 808 caractères et la médiane se situe à 626 caractères.
            D'après le boxplot, il existe beaucoup d'outliers au dela de 2000 caractères.
            """
        )
        st.markdown("##### Distribution de catégorie de produit selon la présence de description")
        st.image("assets/prdtypecode_by_description_presence.png")
        st.markdown(
            """
            La répartition de produits avec ou sans description n'est pas homogène entre les catégories de produit par rapport au ratio moyen du jeu de données (65% avec et 35% sans description).
            Certains produits comme 2403 contiennent majoritairement des produits non décrits alors que d'autres comme 1560 contiennent majoritairement des produits décrits.
            """
        )
        st.markdown("##### Wordclouds de la variable description")
        st.markdown(
            """
            Nous avons décidé d'analyser plus en détail le contenu de la variable description pour chaque catégorie de produit.
            Pour cela nous avons tokenisé, nettoyé, supprimé les stop words et généré des wordclouds.
            """
        )
        cols = st.columns(2)
        with cols[0]:
            st.image("assets/wordclouds/wc1.png")
            st.image("assets/wordclouds/wc3.png")
            st.image("assets/wordclouds/wc5.png")
            st.image("assets/wordclouds/wc7.png")
        with cols[1]:
            st.image("assets/wordclouds/wc2.png")
            st.image("assets/wordclouds/wc4.png")
            st.image("assets/wordclouds/wc6.png")
            st.image("assets/wordclouds/wc8.png")
        st.markdown(
            """
            Les 4 (premiers) worclouds des catégories de produits avec le plus de descriptions et les 4 (derniers) wordclouds de celles avec le moins de descriptions mettent clairement en avant certains termes plus que d'autres.

            Cela nous permet d'identifier la catégorie de produit d'une manière sémantique à la lecture des termes mis en exergue.
            """
        )
    with tab3:
        st.markdown("##### Format")
        st.image("assets/images_format.png")
        st.markdown(
            """
            Nous observons que toutes les images sont en couleur et de taille 500 × 500 pixels.
            """
        )

        st.markdown("##### Content box")
        cols = st.columns(2)
        with cols[0]:
            show_image_with_gray_border("assets/before_content_dim.png", caption="Avant la détection de la bounding box")
        with cols[1]:
            show_image_with_gray_border("assets/after_content_dim.png", caption="Après la détection de la bounding box")
        st.write("Nous déterminons la zone utile de l'image en détectant la bounding box du contenu non blanc. Nous appliquons cette analyse à un échantillon aléatoire du jeu de données pour des raisons de temps de calcul.")
        st.image("assets/content_dim_is_null.png")
        st.markdown(
            """
            Quand une image est entièrement blanche, alors le content_dim est nul. Ce n'est pas le cas ici alors aucune de nos images (donc aucune des images liées aux produits) n'est "nulle".

            Il ne s'agit que d'un échantillon, même s'il a été choisi aléatoirement, cela ne nous donne pas la proportion exacte. Nous supposerons ainsi qu'aucune image n'est entièrement blanche.
            """
        )
        st.image("assets/uniformity.png")
        st.markdown(
            """
            52,5% des images n’ont aucune dimension égale à 500, ce qui signifie qu’elles n’occupent pas tout l’espace disponible. Cette hétérogénéité réduit l’efficacité de la zone utile pour l’apprentissage automatique, car la quantité d’information varie fortement selon les images. Pour harmoniser les données, on pourrait appliquer des techniques de Computer Vision, comme l’agrandissement par masques ou filtres jusqu’aux bords. Ainsi, un modèle s’entraînerait sur des images plus cohérentes, évitant de comparer des données pleines d’informations à d’autres majoritairement vides (pixels blancs).

            La visualisation de tous les formats de Content box serait trop dispersée, mais un heatmap de luminance permettrait d’analyser la répartition des pixels et donc les dimensions des images. Une autre piste d’étude intéressante concerne le rapport longueur/largeur, qui renseignerait sur la forme générale des images.
            """
        )
        st.markdown("##### Luminance")
        st.markdown(
            """
            Nous transformons nos images en nuances de gris pour effectuer une heatmap de la luminance et ainsi avoir une idée de la répartition de l'information utile sur les images.
            """
        )

        st.image("assets/heatmap2.png")
        st.markdown(    
            """
            Les valeurs vont de 0.56 (noir), 0.96 (blanc)
            La répartition montre que les pixels blancs se concentrent surtout sur les coins et les bords, tandis que le contenu se situe plutôt au centre. Le contraste global reste faible, oscillant seulement sur une partie des valeurs possibles.
            
            En analysant la heatmap autour des intensités moyennes, on distingue une tâche sombre centrale, un carré orange bien défini, des côtés éclaircis et des coins presque entièrement blancs. Cela révèle que beaucoup d’images sont centrées mais mal dimensionnées, et qu’un certain nombre occupent tout l’espace en laissant néanmoins des bords ou coins vides selon leur proportion longueur/largeur.
            """
        )

        st.markdown("##### Distribution du ratio largeur/hauteur")
        st.markdown(
            """
            Pour étudier le ratio longueur/largeur, il faut considérer la véritable taille des images sans les bords vides, ce qui correspond à la Content Box. Celle-ci définit le plus petit rectangle englobant tous les pixels non-blancs.
            """
        )
        st.image("assets/width_length_ratio.png")
        st.markdown(
            """           
            La distribution des ratios est globalement symétrique autour de 1, entre 0 et 2. Toutefois, quelques valeurs extrêmes jusqu’à 11 rompent cette symétrie, bien qu’elles deviennent de plus en plus rares en s’éloignant de 1.

            La distribution du ratio longueur/largeur pourrait s’approcher d’une loi lognormale à faible écart-type, plutôt qu’une loi normale à cause des valeurs extrêmes. En moyenne, les images sont aussi longues que larges, sans différence notable entre celles orientées en largeur ou en hauteur.
            """
        )
        st.markdown("##### Conclusion")
        st.markdown(
            """
            Les images sont centrées, les rebords sont globalement bien plus blancs car les images ne prennent pas la totalité de l'espaces, et les coins sont les moins représentés car ils sont laissés de côté à la fois par les images plus petites, mais aussi par les images dont les dimensions sont asymétriques.

            Les images de dimensions asymétriques sont plutôt équitablement réparties entre les images plus longues que larges et les images plus larges que longues.
            """
        )
    with tab4:
        st.markdown("#### Stratégie MLFlow proposée")

        st.markdown("##### Préparation des données")
        st.markdown("""
        - Diviser **X_train** et **y_train** en ensembles d'entraînement et de test avec la même répartition.  
        - Décider si les descriptions nulles doivent être remplacées par des descriptions générées (ex. à partir de la désignation ou de mots-clés).
        """)

        st.markdown("##### Modélisation texte")
        st.markdown("""
        - Construire un modèle de classification des produits basé sur le texte.  
        - Intégrer des traitements NLP sur la désignation et la description.
        """)

        st.markdown("##### Modélisation images")
        st.markdown("""
        - Appliquer des techniques de Computer Vision pour harmoniser le dataset d'images.  
        - Transformer les images en vecteurs de pixels associés aux produits.  
        - Construire un modèle de classification des produits à partir des images.
        """)

        st.markdown("##### Évaluation et amélioration")
        st.markdown("""
        - Évaluer les performances des modèles texte et image.  
        - Ajuster et optimiser ces modèles pour améliorer les performances.
        """)

        st.markdown("##### Fusion des modèles")
        st.markdown("""
        - Définir des pondérations pour les modèles en fonction de leurs performances respectives.  
        - Créer une fonction pour calculer une moyenne pondérée des probabilités des classes (comme en Bagging).  
        - Évaluer la classification finale basée sur cette moyenne pondérée pendant la phase de test.  
        - Optimiser les pondérations pour améliorer les résultats.
        """)

        st.markdown("##### Conclusion")
        st.markdown("""
        - Comparer les performances du modèle combiné avec celles des meilleurs modèles individuels.  
        - Conclure sur la solution la plus efficace entre modèle seul et combinaison.
        """)
    with tab5:
        st.write("Voici un échantillon de données pour chaque catégorie de produit : nous affichons la catégorie, la designation, la description et l'image.")

        X_train, y_train = load_data()
        for index, row in X_train.iterrows():
            st.markdown("---")
            cols = st.columns(2)
            with cols[0]:
                st.image(f"data/images/image_{row['imageid']}_product_{row['productid']}.jpg")
            with cols[1]:
                st.markdown(
                    f"""
                    **Catégorie de produit :** {y_train.loc[index, 'prdtypecode_label']}

                    **Product ID :** {row['productid']}

                    **Designation :** {row['designation']}

                    **Description :**
                    """,
                )
                st.html(row['description'])
                

import streamlit as st


title = "Modélisation"
sidebar_name = "Modélisation"


def run():
    st.image("assets/processing.svg")

    st.title(title)
    
    tab1, tab2, tab3 = st.tabs(["Modèles de classification des textes", "Modèles de classification des images", "Modèles multimodaux"])

    with tab1:
        st.markdown(
            """
            Pour l’analyse de texte, nous avons testé trois classificateurs du plus simple au plus complexe (régression logistique, Random Forest et XGBoost) et optimisé leurs hyperparamètres via différentes stratégies de recherche  afin d’améliorer l’entraînement tout en limitant les ressources.
            """
        )
        st.markdown("#### Logistic Regression")
        st.markdown(
            """
            **Recherche d’hyperparamètres** : GridSearchCV (score F1 pondéré, validation croisée 3 plis)  
            **Nombre total de fits** : 30 sur 78 min (4676 s) 

            | Paramètre | Scope de recherche          | Valeur optimale |
            |-----------|----------------------------|----------------|
            | solver    | ["lbfgs", "saga"]          | "saga"         |
            | C         | [0.01, 0.1, 1, 10, 100]   | 100            |

            """
        )
        logistic_regression_report_acc_data = {
            "metric": ["accuracy", "macro avg", "weighted avg"],
            "precision": [None, 0.70, 0.74],
            "recall": [None, 0.72, 0.72],
            "f1-score": [0.72, 0.70, 0.73],
            "support": [16984, 16984, 16984]
        }
        st.dataframe(logistic_regression_report_acc_data)
        with st.expander("Performance"):
            st.markdown("**Classification report**")
            logistic_regression_report_main_data = {
                "class": [10, 40, 50, 60, 1140, 1160, 1180, 1280, 1281, 1300, 1301, 1302, 1320,
                1560, 1920, 1940, 2060, 2220, 2280, 2403, 2462, 2522, 2582, 2583, 2585, 2705, 2905],
                "precision": [0.33, 0.60, 0.67, 0.80, 0.68, 0.90, 0.40, 0.63, 0.43, 0.83, 0.84, 0.59, 0.65,
                            0.77, 0.85, 0.61, 0.74, 0.60, 0.76, 0.80, 0.73, 0.86, 0.59, 0.96, 0.54, 0.73, 0.93],
                "recall": [0.64, 0.54, 0.70, 0.86, 0.73, 0.86, 0.59, 0.41, 0.51, 0.84, 0.86, 0.65, 0.62,
                        0.67, 0.87, 0.94, 0.64, 0.81, 0.71, 0.63, 0.74, 0.81, 0.69, 0.88, 0.69, 0.70, 0.97],
                "f1-score": [0.44, 0.57, 0.69, 0.83, 0.71, 0.88, 0.47, 0.49, 0.47, 0.84, 0.85, 0.62, 0.63,
                            0.72, 0.86, 0.74, 0.69, 0.69, 0.73, 0.71, 0.74, 0.84, 0.64, 0.92, 0.60, 0.71, 0.95],
                "support": [623, 502, 336, 166, 534, 791, 153, 974, 414, 1009, 161, 498, 648,
                            1015, 861, 161, 999, 165, 952, 955, 284, 998, 518, 2042, 499, 552, 174]
            }
            st.dataframe(logistic_regression_report_main_data, height=250)

            st.markdown("**Confusion Matrix**")
            st.image("assets/heatmaps/logistic_regression_text_confusion_matrix.png")
        
        st.markdown("#### Random Forest")
        st.markdown(
            """
            **Recherche d’hyperparamètres** : RandomizedSearchCV (score F1 pondéré, validation croisée à 3 plis) 
            **Nombre total de fits** : 18 sur 248 min (14876 s)

            | Paramètre     | Scope de recherche     | Valeur optimale |
            |---------------|------------------------|-----------------|
            | n_estimators  | [50, 100, 200]        | 200             |
            | max_depth     | [10, 20, 30]          | 30              |
            """
        )
        random_forest_report_acc_data = {
            "metric": ["accuracy", "macro avg", "weighted avg"],
            "precision": [None, 0.76, 0.76],
            "recall": [None, 0.74, 0.75],
            "f1-score": [0.75, 0.74, 0.75],
            "support": [16984, 16984, 16984]
        }
        st.dataframe(random_forest_report_acc_data)
        with st.expander("Performance"):
            st.markdown("**Classification report**")
            random_forest_report_main_data = {
                "class": [10, 40, 50, 60, 1140, 1160, 1180, 1280, 1281, 1300, 1301, 1302, 1320,
                1560, 1920, 1940, 2060, 2220, 2280, 2403, 2462, 2522, 2582, 2583, 2585, 2705, 2905],
                "precision": [0.38, 0.69, 0.73, 0.92, 0.71, 0.89, 0.75, 0.65, 0.58, 0.82, 0.95, 0.78, 0.69,
                            0.74, 0.87, 0.81, 0.76, 0.75, 0.67, 0.74, 0.78, 0.77, 0.74, 0.96, 0.60, 0.73, 0.97],
                "recall": [0.56, 0.56, 0.75, 0.81, 0.76, 0.90, 0.59, 0.52, 0.50, 0.87, 0.82, 0.63, 0.67,
                        0.76, 0.89, 0.87, 0.74, 0.70, 0.76, 0.68, 0.76, 0.85, 0.70, 0.91, 0.72, 0.70, 0.98],
                "f1-score": [0.45, 0.62, 0.74, 0.86, 0.73, 0.89, 0.66, 0.57, 0.54, 0.84, 0.88, 0.70, 0.68,
                            0.75, 0.88, 0.84, 0.75, 0.72, 0.71, 0.71, 0.77, 0.81, 0.72, 0.94, 0.66, 0.71, 0.97],
                "support": [623, 502, 336, 166, 534, 791, 153, 974, 414, 1009, 161, 498, 648,
                            1015, 861, 161, 999, 165, 952, 955, 284, 998, 518, 2042, 499, 552, 174]
            }
            st.dataframe(random_forest_report_main_data, height=250)
            st.markdown("**Confusion Matrix**")
            st.image("assets/heatmaps/random_forest_confusion_matrix.png")
        
        st.markdown("#### XGBoost")
        st.markdown(
            """
            **Recherche hyperparamètres** : HalvingGridSearchCV (3-fold CV, F1 pondéré)  
            **Temps d'exécution** : 31,8 min (1908 s)  
            
            | Paramètre      | Scope de recherche  | Valeur optimale |
            |----------------|---------------------|-----------------|
            | n_estimators   | [100, 200, 300]     | 300             |
            | max_depth      | [3, 5, 6, 7]        | 3               |
            | learning_rate  | [0.01, 0.1, 0.2]    | 0.2             |
            """
        )
        xgboost_report_acc_data = {
            'Class': ['accuracy', 'macro avg', 'weighted avg'],
            'precision': [ None, 0.74, 0.75],
            'recall': [None, 0.74, 0.75],
            'f1-score': [.75, 0.74, 0.75],
            'support': [16984, 16984, 16984]
        }
        st.dataframe(xgboost_report_acc_data)
        with st.expander("Performance"):
            st.markdown("**Classification report**")
            xgboost_report_main_data = {
                'Class': ['10', '40', '50', '60', '1140', '1160', '1180', '1280', '1281', '1300', '1301', '1302', '1320', '1560', '1920', '1940', '2060', '2220', '2280', '2403', '2462', '2522', '2582', '2583', '2585', '2705', '2905'],
                'precision': [0.38, 0.62, 0.72, 0.89, 0.71, 0.89, 0.63, 0.61, 0.51, 0.82, 0.96, 0.64, 0.66, 0.76, 0.86, 0.75, 0.74, 0.75, 0.75, 0.74, 0.75, 0.85, 0.70, 0.97, 0.58, 0.68, 0.97],
                'recall': [0.53, 0.60, 0.78, 0.82, 0.74, 0.89, 0.58, 0.47, 0.51, 0.87, 0.83, 0.69, 0.70, 0.72, 0.89, 0.89, 0.70, 0.74, 0.73, 0.67, 0.77, 0.83, 0.69, 0.91, 0.73, 0.71, 0.98],
                'f1-score': [0.45, 0.61, 0.75, 0.86, 0.73, 0.89, 0.60, 0.53, 0.51, 0.84, 0.89, 0.67, 0.68, 0.74, 0.88, 0.82, 0.72, 0.74, 0.74, 0.70, 0.76, 0.84, 0.69, 0.94, 0.65, 0.69, 0.97],
                'support': [623, 502, 336, 166, 534, 791, 153, 974, 414, 1009, 161, 498, 648, 1015, 861, 161, 999, 165, 952, 955, 284, 998, 518, 2042, 499, 552, 174]
            }
            st.dataframe(xgboost_report_main_data, height=250)

            st.markdown("**Confusion Matrix**")
            st.image("assets/heatmaps/xgboost_confusion_matrix.png")

    with tab2:
        st.markdown(
            """
            Pour la classification des images, nous avons testé trois modèles en partant du plus naïf au plus complexe : une régression logistique, un modèle CNN personnalisé et un modèle Fine tuned Resnet.
            """
        )
        st.markdown("#### Régression logistique")
        st.markdown("##### Réduction de dimension par PCA incrémental")
        st.markdown("""
        - Utilisation d’un PCA incrémental (batch 512 images)  
        - Réduction : **40 000 ➝ 256 composantes**
        - Normalisation [0,1] en float32
                    
        Cette réduction de dimension permet de diminuer drastiquement le temps d’entraînement de la régression logistique. Cependant, nous avions déjà eu une perte d'information en passant de 500x500 en couleurs à 200x200 en niveaux de gris.
                    
        Pour des résultats optimaux en évitant une double perte d'information, nous aurions pu appliquer ce même PCA sur les images 500x500 en couleurs mais le PCA aussi est coûteux. La logreg étant surtout à but exploratoire, nous avons conservé cette approche.
        """)
        st.markdown(
            """
            ##### Hyperparamètres
            Nous avons exploré les hyperparamètres suivants avec HalvingGridSearchCV (score F1 pondéré, validation croisée 3 plis) :

            | Paramètre | Ensemble 1   | Ensemble 2           | Valeur optimale   |
            |-----------|--------------|----------------------|-------------------|
            | C         | [0.1, 1, 10] | [0.1, 1, 10]         | 0.1               |
            | solver    | "lbfgs"      | "saga"               | "saga"            |
            | penalty   | "l2"         | ["l1", "elasticnet"] | "elasticnet"      |
            | l1_ratio  | -            | 0.5                  | 0.5               |
            """
        )
        image_log_reg_report_acc_data = {
            "Metric": ["Accuracy", "Macro avg", "Weighted avg"],
            "Precision": [None, 0.22, 0.27],
            "Recall": [None, 0.25, 0.24],
            "F1-score": [0.24, 0.22, 0.24],
            "Support": [16984, 16984, 16984]
        }
        st.dataframe(image_log_reg_report_acc_data)
        st.markdown("La régression logistique n'est pas assez performante pour la classification des images. Nous décidons donc d'entraîner un modèle CNN personnalisé au vu du nombre de données disponibles raisonnable.")

        with st.expander("Performance"):
            st.markdown("**Classification report**")
            image_log_reg_report_main_data = {
                'Class': ['10', '40', '50', '60', '1140', '1160', '1180', '1280', '1281', '1300', '1301', '1302', '1320', '1560', '1920', '1940', '2060', '2220', '2280', '2403', '2462', '2522', '2582', '2583', '2585', '2705', '2905'],
                "Precision": [0.31, 0.24, 0.08, 0.04, 0.17, 0.49, 0.04, 0.13, 0.07, 0.18,
                            0.05, 0.09, 0.19, 0.19, 0.48, 0.05, 0.28, 0.04, 0.45, 0.29,
                            0.13, 0.28, 0.11, 0.47, 0.09, 0.44, 0.56],
                "Recall": [0.39, 0.21, 0.13, 0.18, 0.15, 0.55, 0.15, 0.01, 0.03, 0.07,
                        0.25, 0.06, 0.17, 0.13, 0.51, 0.24, 0.19, 0.15, 0.38, 0.26,
                        0.29, 0.31, 0.10, 0.21, 0.10, 0.69, 0.91],
                "F1-score": [0.35, 0.22, 0.10, 0.07, 0.16, 0.52, 0.07, 0.02, 0.04, 0.10,
                            0.08, 0.08, 0.18, 0.16, 0.49, 0.09, 0.23, 0.06, 0.41, 0.28,
                            0.18, 0.30, 0.10, 0.29, 0.09, 0.54, 0.70],
                "Support": [623, 502, 336, 166, 534, 791, 153, 974, 414, 1009,
                            161, 498, 648, 1015, 861, 161, 999, 165, 952, 955,
                            284, 998, 518, 2042, 499, 552, 174]
            }
            st.dataframe(image_log_reg_report_main_data, height=300)
            st.markdown("**Matrice de confusion**")
            st.image("assets/heatmaps/logistic_regression_image_confusion_matrix.png")

        st.markdown("#### CNN Keras")
        st.markdown(
            """
            ##### Méthodologie  
            - Prétraitement : resize (224×224), normalisation, encodage labels  
            - Split : 80% train / 20% val, test séparé  
            - Dataset : `tf.data` (batch 32, prefetch, AUTOTUNE)  

            ##### Architecture  
            | Layer | Détails |
            |-------|---------|
            | Conv2D (32, 3×3, ReLU) + MaxPool(2) | Extraction bas-niveau |
            | Conv2D (64, 3×3, ReLU) + MaxPool(2) | Features intermédiaires |
            | Conv2D (128, 3×3, ReLU) + MaxPool(2) | Features complexes |
            | Flatten | - |
            | Dense (128, ReLU) | Fully connected |
            | Dropout (0.5) | Régularisation |
            | Dense (num_classes, Softmax) | Classification finale |

            ##### Entraînement  
            | Paramètre | Valeur |
            |-----------|--------|
            | Optimiseur | Adam |
            | Loss | SparseCategoricalCrossentropy |
            | Batch size | 32 |
            | Epochs | 10 |
            | Validation split | 20% |

            """
        )
        st.markdown(
            """
            Le modèle CNN personnalisé que nous avons entraîné est malheureusement un échec : on observe un collapse rendant le modèle inutilisable. (Classe: outillage et accessoires pour travaux ou jardinage)
            Nous décidons donc d'entraîner un modèle Fine tuned Resnet avec une base de reconnaissance d'images déjà solide.
            """
        )
        with st.expander("Matrice de confusion"):
            st.image("assets/heatmaps/cnn_image_confusion_matrix.png")

        st.markdown(
            """
            #### Fine tuned Resnet

            Le modèle Resnet18 est un modèle de reconnaissance d'images pré-entraîné sur le dataset ImageNet. Nous l'avons fine tuned sur notre dataset Rakuten afin de l'adapter à notre problématique.

            Nous avons décidé de dégeler le dernier bloc et de fournir notre propre couche fully connected pour la classification à 27 classes.
            """
        )
        resnet_report_acc_data = {
            "Metric": ["Accuracy", "Macro avg", "Weighted avg"],
            "Precision": [None, 0.49, 0.51],
            "Recall": [None, 0.43, 0.50],
            "F1-score": [0.50, 0.44, 0.49],
            "Support": [16984, 16984, 16984]
        }
        st.dataframe(resnet_report_acc_data)
        st.markdown(
            """
            Le modèle obtient une performance satisfaisante sachant que certaines images sont difficiles à classifier car certains types d'objets peuvent appartenir à plusieurs classes. Par exemple les figurines peuvent être classées comme des jouets ou des objets de collection de jeu vidéo.
            """
        )
        with st.expander("Performance"):
            st.markdown("**Rapport de classification**")
            resnet_report_main_data = {
                "Class": [10, 40, 50, 60, 1140, 1160, 1180, 1280, 1281, 1300,
                1301, 1302, 1320, 1560, 1920, 1940, 2060, 2220, 2280,
                2403, 2462, 2522, 2582, 2583, 2585, 2705, 2905],
                "Precision": [0.62, 0.58, 0.32, 0.59, 0.55, 0.84, 0.36, 0.28, 0.36, 0.56,
                            0.37, 0.41, 0.35, 0.48, 0.73, 0.48, 0.45, 0.51, 0.53, 0.39,
                            0.49, 0.56, 0.34, 0.61, 0.27, 0.62, 0.49],
                "Recall": [0.24, 0.32, 0.19, 0.58, 0.37, 0.80, 0.15, 0.43, 0.07, 0.58,
                        0.38, 0.20, 0.41, 0.48, 0.72, 0.59, 0.36, 0.13, 0.71, 0.71,
                        0.30, 0.54, 0.36, 0.73, 0.20, 0.64, 0.49],
                "F1-score": [0.35, 0.41, 0.24, 0.58, 0.45, 0.82, 0.21, 0.34, 0.11, 0.57,
                            0.38, 0.27, 0.38, 0.48, 0.73, 0.53, 0.40, 0.20, 0.61, 0.51,
                            0.37, 0.55, 0.35, 0.67, 0.23, 0.63, 0.49],
                "Support": [623, 502, 336, 166, 534, 791, 153, 974, 414, 1009,
                            161, 498, 648, 1015, 861, 161, 999, 165, 952, 955,
                            284, 998, 518, 2042, 499, 552, 174]
            }
            st.dataframe(resnet_report_main_data, height=300)

            st.markdown("**Matrice de confusion**")
            st.image("assets/heatmaps/resnet_confusion_matrix.png")

    with tab3:
        st.markdown("""
        Nous avons testé deux modèles multimodaux simples : le late fusion et le stacking.
        """)
        st.markdown("##### Late Fusion")
        st.markdown("""
        Le modèle de late fusion est un modèle qui combine les prédictions des modèles de classification des textes et des images.
        Il combine les prédictions du XGBoost textuel et du ResNet image en faisant la moyenne des probabilités des deux modèles.

        Malgré notre intuition initiale, cette tentative atteinte un score de 0.75 égale à celui du modèle XGBoost textuel seul. On en déduit qu'il n'y a pas de gain significatif à combiner les modèles textuels et visuels si leurs prédictions ne se complètent pas.
        """)
        
        late_fusion_report_acc_data = {
            "metric": ["accuracy", "macro avg", "weighted avg"],
            "precision": [None, 0.71, 0.76],
            "recall": [None, 0.75, 0.75],
            "f1-score": [0.75, 0.72, 0.75],
            "support": [16984, 16984, 16984]
        }
        st.dataframe(late_fusion_report_acc_data)
        with st.expander("Performance"):
            st.markdown("**Rapport de classification**")
            late_fusion_report_main_data = {
                "class": [10, 40, 50, 60, 1140, 1160, 1180, 1280, 1281, 1300, 1301, 1302, 1320, 1560,
                        1920, 1940, 2060, 2220, 2280, 2403, 2462, 2522, 2582, 2583, 2585, 2705, 2905],
                "precision": [0.58, 0.69, 0.60, 0.85, 0.80, 0.91, 0.60, 0.68, 0.46, 0.78, 0.61, 0.68,
                            0.73, 0.82, 0.87, 0.58, 0.69, 0.65, 0.82, 0.79, 0.64, 0.78, 0.53, 0.97,
                            0.48, 0.78, 0.77],
                "recall": [0.62, 0.67, 0.75, 0.81, 0.74, 0.95, 0.58, 0.41, 0.49, 0.86, 0.90, 0.67,
                        0.62, 0.62, 0.85, 0.90, 0.69, 0.74, 0.72, 0.76, 0.79, 0.83, 0.70, 0.88,
                        0.71, 0.90, 0.98],
                "f1-score": [0.60, 0.68, 0.66, 0.83, 0.77, 0.93, 0.59, 0.51, 0.48, 0.82, 0.73, 0.68,
                            0.67, 0.71, 0.86, 0.71, 0.69, 0.69, 0.77, 0.78, 0.71, 0.81, 0.61, 0.92,
                            0.57, 0.83, 0.86],
                "support": [624, 501, 336, 166, 534, 791, 153, 974, 414, 1009, 161, 498, 649, 1015,
                            861, 160, 999, 165, 952, 955, 284, 998, 518, 2042, 499, 552, 174]
            }
            st.dataframe(late_fusion_report_main_data, height=250)
            st.markdown("**Matrice de confusion**")
            st.image("assets/heatmaps/late_fusion_confusion_matrix.png")
        st.markdown("##### Stacking")
        st.markdown("""
        Le modèle de stacking ajoute un méta-classifieur (régression logistique) qui combine les probabilités issues des modèles texte et image pour améliorer la classification.

        Idem, ce modèle atteint un score de 0.72 en dessous de celui du modèle XGBoost textuel seul. Cela peut s'expliquer par le fait que les modèles de classification des textes et des images ne sont pas assez performants pour se compléter et/ou que nos paramètres ne sont pas optimisés pour le méta classifieur.
        """)
        stacking_report_acc_data = {
            "metric": ["accuracy", "macro avg", "weighted avg"],
            "precision": [None, 0.72, 0.73],
            "recall": [None, 0.72, 0.72],
            "f1-score": [0.72, 0.72, 0.72],
            "support": [16984, 16984, 16984]
        }
        st.dataframe(stacking_report_acc_data)
        with st.expander("Performance"):
            st.markdown("**Rapport de classification**")
            stacking_report_main_data = {
                "class": [10, 40, 50, 60, 1140, 1160, 1180, 1280, 1281, 1300, 1301, 1302, 1320, 1560,
                1920, 1940, 2060, 2220, 2280, 2403, 2462, 2522, 2582, 2583, 2585, 2705, 2905],
                "precision": [0.34, 0.58, 0.67, 0.93, 0.69, 0.90, 0.67, 0.57, 0.48, 0.78, 0.95, 0.67,
                            0.63, 0.77, 0.87, 0.80, 0.72, 0.87, 0.57, 0.70, 0.71, 0.87, 0.64, 0.98,
                            0.59, 0.60, 0.96],
                "recall": [0.46, 0.56, 0.75, 0.78, 0.75, 0.83, 0.58, 0.50, 0.51, 0.79, 0.89, 0.69,
                        0.69, 0.74, 0.87, 0.82, 0.72, 0.73, 0.58, 0.65, 0.74, 0.79, 0.71, 0.89,
                        0.72, 0.63, 0.95],
                "f1-score": [0.39, 0.57, 0.71, 0.85, 0.72, 0.86, 0.62, 0.53, 0.49, 0.78, 0.92, 0.68,
                            0.66, 0.76, 0.87, 0.81, 0.72, 0.80, 0.57, 0.68, 0.73, 0.83, 0.67, 0.93,
                            0.65, 0.62, 0.96],
                "support": [624, 501, 336, 166, 534, 791, 153, 974, 414, 1009, 161, 498, 649, 1015,
                            861, 160, 999, 165, 952, 955, 284, 998, 518, 2042, 499, 552, 174]
            }
            st.dataframe(stacking_report_main_data, height=250)
            st.markdown("**Matrice de confusion**")
            st.image("assets/heatmaps/stacking_confusion_matrix.png")

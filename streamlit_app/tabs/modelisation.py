import streamlit as st


title = "Modélisation"
sidebar_name = "Modélisation"


def run():
    st.image("assets/processing.svg")

    st.title(title)
    
    tab1, tab2 = st.tabs(["Modèles de classification des textes", "Modèles de classification des images"])

    with tab1:
        st.markdown(
            """
            Pour l’analyse de texte, nous avons testé trois classificateurs du plus simple au plus complexe (régression logistique, Random Forest et XGBoost) et optimisé leurs hyperparamètres via différentes stratégies de recherche  afin d’améliorer l’entraînement tout en limitant les ressources.
            """
        )
        st.markdown("#### Logistic Regression")
        st.markdown(
            """
            """
        )
        st.markdown("#### Random Forest")
        st.markdown(
            """
            """
        )
        st.markdown("#### XGBoost")
        st.markdown(
            """
            **Recherche hyperparamètres** : HalvingGridSearchCV (3-fold CV, F1 pondéré)  
            
            | Paramètre      | Scope de recherche  | Valeur optimale |
            |----------------|---------------------|-----------------|
            | n_estimators   | [100, 200, 300]     | 300             |
            | max_depth      | [3, 5, 6, 7]        | 3               |
            | learning_rate  | [0.01, 0.1, 0.2]    | 0.2             |
            
            **Temps d’exécution** : 31,8 min (1908 s)  
            
            **Classification report** :  
            """
        )
        data = {
            'Class': ['10', '40', '50', '60', '1140', '1160', '1180', '1280', '1281', '1300', '1301', '1302', '1320', '1560', '1920', '1940', '2060', '2220', '2280', '2403', '2462', '2522', '2582', '2583', '2585', '2705', '2905'],
            'precision': [0.38, 0.62, 0.72, 0.89, 0.71, 0.89, 0.63, 0.61, 0.51, 0.82, 0.96, 0.64, 0.66, 0.76, 0.86, 0.75, 0.74, 0.75, 0.75, 0.74, 0.75, 0.85, 0.70, 0.97, 0.58, 0.68, 0.97],
            'recall': [0.53, 0.60, 0.78, 0.82, 0.74, 0.89, 0.58, 0.47, 0.51, 0.87, 0.83, 0.69, 0.70, 0.72, 0.89, 0.89, 0.70, 0.74, 0.73, 0.67, 0.77, 0.83, 0.69, 0.91, 0.73, 0.71, 0.98],
            'f1-score': [0.45, 0.61, 0.75, 0.86, 0.73, 0.89, 0.60, 0.53, 0.51, 0.84, 0.89, 0.67, 0.68, 0.74, 0.88, 0.82, 0.72, 0.74, 0.74, 0.70, 0.76, 0.84, 0.69, 0.94, 0.65, 0.69, 0.97],
            'support': [623, 502, 336, 166, 534, 791, 153, 974, 414, 1009, 161, 498, 648, 1015, 861, 161, 999, 165, 952, 955, 284, 998, 518, 2042, 499, 552, 174]
        }
        st.dataframe(data, height=300)
        data = {
            'Class': ['accuracy', 'macro avg', 'weighted avg'],
            'precision': [ .75, 0.74, 0.75],
            'recall': [None, 0.74, 0.75],
            'f1-score': [None, 0.74, 0.75],
            'support': [16984, 16984, 16984]
        }
        st.dataframe(data)
        st.markdown("**Confusion Matrix**")
        st.image("assets/heatmaps/xgboost_confusion_matrix.png")
    with tab2:
        st.markdown("#### Custom CNN")
        st.markdown("#### Fine tuned Resnet")

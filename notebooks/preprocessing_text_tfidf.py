# Utils
import sys
import os
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(PROJECT_ROOT)

from IPython.display import display

import pandas as pd
import numpy as np
from collections import Counter

from sklearn.model_selection import train_test_split
from sklearn.decomposition import TruncatedSVD

from imblearn.pipeline import Pipeline
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler

from scipy.sparse import hstack, csr_matrix

# Traitement des variables textuelles
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
nltk.download('punkt_tab')
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.snowball import FrenchStemmer

import joblib

from scripts.utils import load_data

X, y, _ = load_data()

# Split train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("Train avant over et undersampling :")
print(y_train['prdtypecode'].value_counts(normalize=True) * 100)

# Calcul des effectifs
counts = Counter(y_train['prdtypecode'])
n_total = len(y_train)
target_ratio = 0.06

# Construction d'une sampling_strategy d'over et undersampling
undersampling_strategy = {
    2583: int(n_total * target_ratio)
}
oversampling_strategy = {}
for cls, count in counts.items():
    current_ratio = count / n_total
    if current_ratio < target_ratio:
        oversampling_strategy[cls] = int(n_total * target_ratio)

# Application de l'over et undersampling avec un pipeline
pipeline = Pipeline(steps=[
    ('under', RandomUnderSampler(sampling_strategy=undersampling_strategy, random_state=42)),
    ('over', RandomOverSampler(sampling_strategy=oversampling_strategy, random_state=42))
])

X_train, y_train = pipeline.fit_resample(X_train, y_train)

print("\nTrain après over et undersampling :")
print(y_train['prdtypecode'].value_counts(normalize=True) * 100)

print("\nTest reste inchangé :")
print(y_test['prdtypecode'].value_counts(normalize=True) * 100)

# Stopwords

html_stopwords = [
    'html', 'head', 'body', 'div', 'span', 'p', 'br', 'a', 'img', 'ul', 'li', 'ol', 'table',
    'tr', 'td', 'th', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'b', 'i', 'u', 'strong', 'em',
    'eacute', 'agrave'
]
punctuation_words = [",", ".", "``", "@", "*", "(", ")", "...", "!", "?", "-", 
                  "_", ">", "<", ":", "/", "=", "--", "©", "~", ";", "\\", "\\\\"]
final_stopwords = stopwords.words('english') + stopwords.words('french') + html_stopwords + punctuation_words

# Stemming and processing

stemmer = FrenchStemmer()

def stemming(mots) :
    sortie = []
    for string in mots :
        radical = stemmer.stem(string)
        if (radical not in sortie) : sortie.append(radical)
    return sortie


def preprocessing(text, with_stemming=False):
    text = text.lower()
    tokens = word_tokenize(text)
    result = [word for word in tokens if word not in final_stopwords]
    if with_stemming:
        result = stemming(result)
    return ' '.join(result)

# Création de la variable has_description

# Pour X_train
X_train['has_description'] = X_train['description'].notnull() & (X_train['description'].str.strip() != "")
X_train['has_description'] = X_train['has_description'].astype(int)

# Pour X_test
X_test['has_description'] = X_test['description'].notnull() & (X_test['description'].str.strip() != "")
X_test['has_description'] = X_test['has_description'].astype(int)

# Fusion des variables `designation` et `description` dans `full_description` 

# Pour X_train
X_train['full_description'] = (
    X_train['designation'] + " " + X_train['description'].fillna('')
).str.strip()
# Nettoyage des colonnes inutiles
X_train = X_train.drop(columns=['designation', 'description'])

# Pour X_test
X_test['full_description'] = (
    X_test['designation'] + " " + X_test['description'].fillna('')
).str.strip()
# Nettoyage des colonnes inutiles
X_test = X_test.drop(columns=['designation', 'description'])

# Nettoyage, tokenisation et stemmatisation de la variable `full_description`

result = pd.DataFrame(columns=['preprocessed_full_description', 'full_description'])

for i in X_train.head(10).index:
    current_description = X_train.loc[i, 'full_description']
    result.loc[i, 'full_description'] = current_description
    result.loc[i, 'preprocessed_full_description'] = preprocessing(current_description, with_stemming=True)

display(result)
# Le résultat n'est pas satisfaisant avec la stemmatisation donc on ne la conserve
# pas pour la suite du preprocessing

X_train['preprocessed_full_description'] = ""
X_test['preprocessed_full_description'] = ""

for i in X_train.index:
    X_train.loc[i, 'preprocessed_full_description'] = preprocessing(X_train.loc[i, 'full_description'])

for i in X_test.index:
    X_test.loc[i, 'preprocessed_full_description'] = preprocessing(X_test.loc[i, 'full_description'])

display(X_train[["full_description", "preprocessed_full_description"]].head(10))
display(X_test[["full_description", "preprocessed_full_description"]].head(10))

# TFIDF

# Initialisation du vecteur TF-IDF
tfidf = TfidfVectorizer(
    max_features=10000,
    ngram_range=(1,1),
    min_df=10
)

# Apprentissage sur X_train et transformation
X_train_tfidf = tfidf.fit_transform(X_train['preprocessed_full_description'])
X_test_tfidf = tfidf.transform(X_test['preprocessed_full_description'])


# Extract feature names

feature_names = tfidf.get_feature_names_out()

# Reduction du nombre des données à 2000 lignes

X_train_small, _, y_train_small, _ = train_test_split(X_train_tfidf, y_train,
                                                      train_size = 2000/X_train_tfidf.shape[0],
                                                      stratify = y_train,
                                                      random_state = 42)

X_test_small, _, y_test_small, _ = train_test_split(X_test_tfidf, y_test,
                                                      train_size = 2000/X_test_tfidf.shape[0],
                                                      stratify = y_test,
                                                      random_state = 42)


# Sauvegarde de vectorizer, feature_names et les datasets tfidf dans des fichiers PKL

joblib.dump(tfidf, "./data/tfidf/tfidf_vectorizer.pkl")
joblib.dump(feature_names, "./data/tfidf/feature_names.pkl")

joblib.dump(X_train_small, "./data/tfidf/X_train_tfidf.pkl")
joblib.dump(y_train_small, "./data/tfidf/y_train_tfidf.pkl")
joblib.dump(X_test_small, "./data/tfidf/X_test_tfidf.pkl")
joblib.dump(y_test_small, "./data/tfidf/y_test_tfidf.pkl")
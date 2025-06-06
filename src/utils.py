import re
from pathlib import Path

import pandas as pd
from scipy.sparse import hstack, csr_matrix
import joblib

ENGLISH_STOP_WORDS = set([
    'the', 'and', 'is', 'in', 'to', 'it', 'of', 'for', 'on', 'with', 'as', 'this', 'that', 'by',
])


def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = text.split()
    tokens = [t for t in tokens if t not in ENGLISH_STOP_WORDS]
    return ' '.join(tokens)


def preprocess_input(df, vectorizer, scaler):
    # Limpiar texto
    df['clean_note'] = df['Clinical Note'].apply(clean_text)
    # Transformar texto
    X_text = vectorizer.transform(df['clean_note'])
    # Mapear sexo (F=0, M=1)
    df['Sex'] = df['Sex'].map({'F': 0, 'M': 1})
    # Escalar edad
    X_age = scaler.transform(df[['Age']])
    # Concatenar edad y sexo
    X_tabular = hstack([csr_matrix(X_age), csr_matrix(df[['Sex']].values)])
    # Concatenar texto + tabular
    X_final = hstack([X_text, X_tabular])
    return X_final


def load_models():
    vectorizer = joblib.load(Path(__file__).resolve().parent.parent / 'models' / 'vectorizer.pkl')
    scaler = joblib.load(Path(__file__).resolve().parent.parent / 'models' / 'scaler.pkl')
    clf = joblib.load(Path(__file__).resolve().parent.parent / 'models' / 'modelo_random_forest_BrainTumorTreatment.pkl')
    return vectorizer, scaler, clf

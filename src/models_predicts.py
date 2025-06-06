from utils import preprocess_input, load_models

def predict(df_new):
    # df_new ya es DataFrame, no hacer pd.DataFrame de nuevo

    vectorizer, scaler, clf = load_models()

    X_final = preprocess_input(df_new, vectorizer, scaler)

    prediction = clf.predict(X_final)

    return prediction

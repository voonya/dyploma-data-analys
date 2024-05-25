import os
import pickle

from utils import *

def load_models(folder, model_name):
    models = {}
    models_files = [file for file in os.listdir(folder) if file.endswith(".pkl")]
    folder_path_converters = folder + '/converters'

    vectorizer = None
    tfidfconverter = None 

    with open(f"{folder_path_converters}/vectorizer.pkl", 'rb') as file:
        vectorizer = pickle.load(file)
    
    with open(f"{folder_path_converters}/tfidfconverter.pkl", 'rb') as file:
        tfidfconverter = pickle.load(file)

    for model_file in models_files:
        current_model_name = model_file.split(".", 1)[0]
        if(current_model_name != model_name):
            continue
        with open(f"{folder}/{model_file}", 'rb') as file:
            models[model_name] = pickle.load(file)

    return models, vectorizer, tfidfconverter


def predict(msgs, classifier, vectorizer, tfidfconverter):
    text = vectorizer.transform(msgs).toarray()
    text = tfidfconverter.transform(text).toarray()
    labels = classifier.predict(text)
    return labels


def is_propaganda(texts):
    cleared_texts = (clear_text(text) for text in texts)
    models, vectorizer, tfidfconverter = load_models('./models/old_tg_data_15000_features', 'Naive Bayes')
    result = predict(cleared_texts, models['Naive Bayes'], vectorizer, tfidfconverter)


    return {'is_propaganda': [bool(isp) for isp in result]}
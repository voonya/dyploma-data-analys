from itertools import zip_longest

from utils import *

from get_topics_db import get_current_topics_db

MODEL_FOLDER = './models/topics-new-lemma'

def load_models(folder):
    import os
    import pickle

    models = {}
    models_files = [file for file in os.listdir(folder) if file.endswith(".pkl")]
    folder_path_converters = folder + '/converters'

    vectorizer = None

    with open(f"{folder_path_converters}/vectorizer.pkl", 'rb') as file:
        vectorizer = pickle.load(file)

    for model_file in models_files:
        model_name = model_file.split(".", 1)[0]
        with open(f"{folder}/{model_file}", 'rb') as file:
            models[model_name] = pickle.load(file)

    return models, vectorizer


def predict(msgs, classifier, vectorizer):
    tf = vectorizer.transform(msgs)
    topic_num = classifier.transform(tf).argmax(axis=1)

    return topic_num

def get_topic_db_id(topic_words, topic_id):
    return topic_words['id'][topic_id]



import pyLDAvis.lda_model

models, vectorizer = load_models(MODEL_FOLDER)

data = get_pandas_df('./data/new_tg_cleared_v3.csv')

lda_visualization = pyLDAvis.lda_model.prepare(models['lda'], vectorizer.transform(data['message_lemmatized']), vectorizer)

def get_topics(texts):
    try:
        cleared_texts = [clear_text(text) for text in texts]
        result = predict(cleared_texts, models['lda'], vectorizer)

        topics = get_current_topics_db()
        topic_ids = [get_topic_db_id(topics, topic_id) for topic_id in result]
        res = [{'topicId': topicId, 'clearedText': clearedText} for topicId, clearedText in zip_longest(topic_ids, cleared_texts)]
        
        return res
    except Exception as err:
        return {'error': f"Unexpected {err=}, {type(err)=}"}

def get_vis():
    return pyLDAvis.prepared_data_to_html(lda_visualization)
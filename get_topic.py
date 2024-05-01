import re
import json
import sys


import spacy

nlp = spacy.load("ru_core_news_lg")
import ru_core_news_lg
nlp = ru_core_news_lg.load()
lemmatizer = nlp.get_pipe("lemmatizer")

def extract_lemmas(text):
    doc = nlp(text)
    lemmas = [token.lemma_ for token in doc]
    return " ".join(lemmas)

MODEL_FOLDER = './models/topics-new-lemma'


def remove_emojis(text):
    # Define regex pattern for emojis
    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"  # emoticons
                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
                               u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                               u"\U00002500-\U00002BEF"  # chinese char
                               u"\U00002702-\U000027B0"
                               u"\U00002702-\U000027B0"
                               u"\U000024C2-\U0001F251"
                               u"\U0001f926-\U0001f937"
                               u"\U00010000-\U0010ffff"
                               u"\u2640-\u2642"
                               u"\u2600-\u2B55"
                               u"\u200d"
                               u"\u23cf"
                               u"\u23e9"
                               u"\u231a"
                               u"\ufe0f"  # dingbats
                               u"\u3030"
                               "]+", flags=re.UNICODE)
    # Remove emojis from the text
    return emoji_pattern.sub(r'', text)

def remove_link_mails_hashtags_tags(text):
    return ' '.join(re.sub("(\S+@\S+\.\S+)|(@[\_A-Za-z0-9]+)|(\w+:\/\/\S+)|", "", text).split())

def remove_card_numbers(text):
    return ' '.join(re.sub("(\d{16})", "", text).split())

def remove_phone_numbers(text):
    return ' '.join(re.sub("((\+\d{1,3})?\s?\(?\d{1,4}\)?[\s.-]?\d{3}[\s.-]?\d{4})", "", text).split())

def remove_punctuation(text):
    return ' '.join(re.sub("([^\w\s]+)", " ", text).split())

def remove_english_text(text):
    return ' '.join(re.sub("([A-Za-z]{2,})", " ", text).split())

def remove_cut_words(text):
    return ' '.join(re.sub("((\s|[\.]{0,})[а-яёА-яЁ]\.)", " ", text).split())

def remove_number_abbr(text):
    return ' '.join(re.sub("(\d+\-[а-яёА-яЁ]+)", " ", text).split())

def remove_numbers(text):
    return ' '.join(re.sub("([0-9]+)", " ", text).split())

def remove_single_characters(text):
    return ' '.join(re.sub("(^|\s+)([а-яёА-яЁa-zA-Z](\s+|$))+", " ", text).split())

def to_lower(text):
    return text.lower()

def remove_template_phrases(text):
    return ' '.join(re.sub("(подпишись|подписывайтесь|подписывайся|(наш\sчат)|(просим поддержать репостами)|подписаться|архангел спецназа|прислать нам).*", " ", text).split())

def clear_text(text):
    from functools import reduce
    pipeline = [remove_emojis, remove_link_mails_hashtags_tags, remove_card_numbers, remove_phone_numbers, remove_number_abbr, remove_punctuation, remove_numbers, remove_english_text, to_lower, remove_single_characters, remove_template_phrases, extract_lemmas]

    return reduce(lambda text, func: func(text), pipeline, text)

def get_pandas_df(filename_path, sep = ','):
    import pandas as pd

    return pd.read_csv(filename_path, sep=sep, encoding='utf-8')

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

def get_topic_words(topic_words, topic_id):
    return topic_words['topic'][topic_id]

def get_topics(texts):
    try:
        cleared_texts = [clear_text(text) for text in texts]
        #print("*******Text cleared*******")
        models, vectorizer = load_models(MODEL_FOLDER)
        #print("*******Models loaded*******")
        result = predict(cleared_texts, models['lda'], vectorizer)

        topics_words = get_pandas_df(MODEL_FOLDER + '/data/topics.csv')
        #print(topics_words['topic'][result])
        #print("*******Prediction done*******")
        current_topic_words = [get_topic_words(topics_words, topic_id) for topic_id in result]
        return {'topic_ids': [int(topic_id) for topic_id in result], 'topic_words': current_topic_words, 'cleared_texts': cleared_texts}
    except Exception as err:
        return {'error': f"Unexpected {err=}, {type(err)=}"}
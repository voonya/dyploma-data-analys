import re
import spacy

nlp = spacy.load("ru_core_news_lg")
import ru_core_news_lg
nlp = ru_core_news_lg.load()
lemmatizer = nlp.get_pipe("lemmatizer")

def extract_lemmas(text):
    doc = nlp(text)
    lemmas = [token.lemma_ for token in doc]
    return " ".join(lemmas)

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
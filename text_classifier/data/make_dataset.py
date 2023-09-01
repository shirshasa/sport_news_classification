# for text preprocessing
from pymorphy2 import MorphAnalyzer
try:
    from nltk.corpus import stopwords
    STOP_WORDS = stopwords.words("russian")
except ImportError:
    STOP_WORDS = ['и', 'в', 'во', 'не', 'что', 'он', 'на', 'я', 'с', 'со', 'как', 'а', 'то', 'все', 'она', 'так', 'его', 'но', 'да', 'ты', 'к', 'у', 'же', 'вы', 'за', 'бы', 'по', 'только', 'ее', 'мне', 'было', 'вот', 'от', 'меня', 'еще', 'нет', 'о', 'из', 'ему', 'теперь', 'когда', 'даже', 'ну', 'вдруг', 'ли', 'если', 'уже', 'или', 'ни', 'быть', 'был', 'него', 'до', 'вас', 'нибудь', 'опять', 'уж', 'вам', 'ведь', 'там', 'потом', 'себя', 'ничего', 'ей', 'может', 'они', 'тут', 'где', 'есть', 'надо', 'ней', 'для', 'мы', 'тебя', 'их', 'чем', 'была', 'сам', 'чтоб', 'без', 'будто', 'чего', 'раз', 'тоже', 'себе', 'под', 'будет', 'ж', 'тогда', 'кто', 'этот', 'того', 'потому', 'этого', 'какой', 'совсем', 'ним', 'здесь', 'этом', 'один', 'почти', 'мой', 'тем', 'чтобы', 'нее', 'сейчас', 'были', 'куда', 'зачем', 'всех', 'никогда', 'можно', 'при', 'наконец', 'два', 'об', 'другой', 'хоть', 'после', 'над', 'больше', 'тот', 'через', 'эти', 'нас', 'про', 'всего', 'них', 'какая', 'много', 'разве', 'три', 'эту', 'моя', 'впрочем', 'хорошо', 'свою', 'этой', 'перед', 'иногда', 'лучше', 'чуть', 'том', 'нельзя', 'такой', 'им', 'более', 'всегда', 'конечно', 'всю', 'между']

from text_classifier.common.preprocessing import preprocess_text
from sklearn.model_selection import GroupShuffleSplit
import pandas as pd
import numpy as np
from typing import Iterable, Union


class Preprocessor:

    def __init__(self):
        self.stopwords_ru = STOP_WORDS
        self.morph = MorphAnalyzer()

    def __call__(self, text: Union[str, Iterable]):
        if isinstance(text, str):
            return preprocess_text(text, morph=self.morph, stopwords=self.stopwords_ru)
        else:
            processed = [preprocess_text(x, morph=self.morph, stopwords=self.stopwords_ru) for x in text]
            return processed


def get_cleaned_dataset(df: pd.DataFrame) -> pd.DataFrame:
    stopwords_ru = stopwords.words("russian")
    morph = MorphAnalyzer()
    df["text_clean"] = df["text"].apply(lambda x: preprocess_text(x, morph, stopwords_ru))
    return df


def get_splits(df: pd.DataFrame) -> tuple[pd.DataFrame, np.array, pd.DataFrame, np.array]:
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=0)
    train_index, test_index = next(gss.split(df, df.category, groups=df.oid))

    df_train = df.iloc[train_index]
    df_test = df.iloc[test_index]

    y_train = df_train["category"].values
    y_test = df_test["category"].values

    return df_train, y_train, df_test, y_test

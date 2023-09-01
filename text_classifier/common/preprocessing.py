import re
from pymorphy2 import MorphAnalyzer


def preprocess_text(text: str, morph: MorphAnalyzer = None, stopwords: list = None) -> str:
    """
    Preprocess a string.
    parameter
        :param text: (str) text to be transformed
        :param morph: morphological analyzer for getting normal forms from russian tokens
        :param stopwords: (list) list of stopwords to remove
    :return
        cleaned text (str)
    """

    # clean (convert to lowercase and remove punctuations, english chars and then strip)
    text = re.sub(r'[^ЁёА-я0-9\s]', '', str(text).lower().strip())
    # naive numbers masking
    text = re.sub(r'(^| )[0-9]+($| )', ' число ', str(text).lower().strip())

    # Tokenize (convert from string to list)
    lst_text = text.split()

    # remove Stopwords
    if stopwords is not None:
        lst_text = [token for token in lst_text if token not in stopwords]

    if morph:
        lst_text = [morph.normal_forms(token)[0] for token in lst_text]

    text = " ".join(lst_text)

    return text

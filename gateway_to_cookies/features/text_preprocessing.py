import re
import nltk
import string
import gensim
from nltk.corpus import stopwords
from itertools import chain
nltk.download('stopwords')
nltk.download('punkt')

stop_words = set(stopwords.words('english') +
                 list(string.punctuation) +
                 ['\\n'] + ['quot'])

regex_str = ["http[s]?://(?:[a-z]|[0-9]|[$-_@.&+]|"
             r"[!*\(\),](?:%[0-9a-f][0-9a-f]))+",
             r"(?:\w+-\w+){2}",
             r"(?:\w+-\w+)",
             r"(?:\\\+n+)",
             r"(?:@[\w_]+)",
             "<[^>]+>",
             r"(?:\w+'\w)",
             r"(?:[\w_]+)",
             r"(?:\S)"
             ]

# Create the tokenizer which will be case insensitive and will ignore space.
tokens_re = re.compile(r'(' + '|'.join(regex_str) + ')',
                       re.VERBOSE | re.IGNORECASE)


def tokenize_document(text, flatten=False):
    """Preprocess a whole raw document.

    Args:
        text (str): Raw string of text.
        flatten (bool): Whether to flatten out sentences

    Return:
        List: preprocessed and tokenized documents

    #UTILS
    """

    if flatten:
        return list(chain(*[clean_and_tokenize(sentence)
                            for sentence in nltk.sent_tokenize(text)]))
    else:
        return [clean_and_tokenize(sentence)
                for sentence in nltk.sent_tokenize(text)]


def clean_and_tokenize(text):
    """Preprocess a raw string/sentence of text.

    Args:
       text (str): Raw string of text.

    Return:
       list of str: Preprocessed tokens.

    #UTILS
    """

    tokens = tokens_re.findall(text)
    _tokens = [t.lower() for t in tokens]
    filtered_tokens = [token.replace('-', '_') for token in _tokens
                       if len(token) > 2
                       and token not in stop_words
                       and not any(x in token for x in string.digits)
                       and any(x in token for x in string.ascii_lowercase)]
    return filtered_tokens


def build_ngrams(documents, n=2, **kwargs):
    """Create ngrams using Gensim's phrases.

    Args:
        documents (list of token lists): List of preprocessed and
                                                tokenized documents
        n (int): The `n` in n-gram.

    Return:
        List: bigrams

    #UTILS
    """
    # Check whether "level" was passed as an argument
    if "level" not in kwargs:
        level = 2
    else:
        level = kwargs["level"]

    def _build_ngrams(documents, n, level):
        """ Create ngrams using Gensim's phrases for a given level """
        phrases = gensim.models.Phrases(documents,  min_count=2, threshold=1, delimiter=b'_')
        bigram = gensim.models.phrases.Phraser(phrases)
        docs_bi = [bigram[doc] for doc in documents]

        if level == n:  # If finished
            return docs_bi
        else:  # Otherwise, keep processing until n-grams satisfied
            return build_ngrams(docs_bi, n=n, level=level+1)

    docs_bi = _build_ngrams(documents, n, level)

    return docs_bi

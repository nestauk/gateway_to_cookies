import os
import logging
import gensim
import numpy as np

logger = logging.getLogger(__name__)

def train_w2v(docs):
    """Train Word2Vec model

    Args:

    Returns:
        gensim.models.Word2Vec
    """

    if 'PYTHONHASHSEED' in os.environ.keys():
        PYTHONHASHSEED = os.environ['PYTHONHASHSEED']
        logger.info(f'Using environment variable PYTHONHASHSEED={PYTHONHASHSEED}')
    else:
        logger.warning(f'No environment variable PYTHONHASHSEED - results not reproducible')

    w2v = gensim.models.Word2Vec(
            docs, size=50, window=10, min_count=2, iter=20, workers=1, seed=0)

    return w2v


def document_vector(word2vec_model, doc):
    """Construct and return document vectors from word embeddings

    Args:
        word2vec_model (gensim.models.Word2Vec):
            Word2Vec model

        doc (list of str):
            List of tokenised documents to construct a document vector for.

    Returns:
        numpy.array

    #UTILS
    """
    # remove out-of-vocabulary words
    doc = [word for word in doc if word in word2vec_model.wv.vocab]
    if doc:
        return np.mean(word2vec_model[doc], axis=0)
    else:
        return np.zeros(word2vec_model.trainables.layer1_size,)

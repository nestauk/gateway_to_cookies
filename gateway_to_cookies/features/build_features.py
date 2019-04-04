# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path


import ast
from pandas import read_csv, DataFrame
from gateway_to_cookies.features.w2v import train_w2v, document_vector


@click.command()
def main():
    """ Runs data processing scripts to turn cleaned data from (../processed) into
        features ready to train models (saved in ../processed).
    """
    logger = logging.getLogger(__name__)

    logger.info('load gateway to research data')
    docs = (read_csv(f"{project_dir}/data/processed/gtr_tokenised.csv",
            usecols=['processed_documents'])
            .processed_documents
            .apply(ast.literal_eval))

    logger.info('making gateway to research word embeddings')
    w2v = train_w2v(docs)
    w2v_out = f"{project_dir}/models/gtr_w2v"
    logger.info(f'saving gateway to research word embeddings to {w2v_out}')
    w2v.wv.save(w2v_out)

    logger.info('making gateway to research document vectors')
    doc_vecs = (DataFrame([document_vector(w2v, doc) for doc in docs],
                index=docs.index,
                columns=[f'dim_{i}' for i in range(w2v.vector_size)])
                )

    docs_out = f"{project_dir}/data/processed/gtr_embedding.csv"
    logger.info(f'saving gateway to research document vectors to {docs_out}')
    doc_vecs.to_csv(docs_out)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # Define project base directory
    project_dir = Path(__file__).resolve().parents[2]

    main()

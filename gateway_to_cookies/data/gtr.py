import logging
from pandas import read_csv
from gateway_to_cookies.features.text_preprocessing import tokenize_document

logger = logging.getLogger(__name__)


def make_gtr(data_dir, usecols, nrows, min_length):
    """Clean and tokenise gateway to research abstract texts


    Args:
        data_dir (str): data directory
        usecols (list[str]): Columns to keep
        nrows (int): number of rows to use
        min_length (int): Minimum token length

    """

    fin = f"{data_dir}/raw/gtr_projects.csv"
    fout = f"{data_dir}/processed/gtr_tokenised.csv"

    msg = ('making gateway to research data set '
           f'from raw data in {fin} ({nrows} rows)')
    logger.info(msg)

    (read_csv(fin, nrows=nrows, usecols=usecols)
     .pipe(clean_gtr)  # Clean
     .pipe(transform_gtr, min_length)  # Tokenise
     .to_csv(fout)  # Save
     )
    logger.info(f'Produced gateway to research data: {fout}')


def clean_gtr(gtr_df):
    """Remove duplicate project_id's and missing values

    Args:
        gtr_df (pandas.DataFrame): Gateway to research data

    Returns:
        pandas.DataFrame
    """

    return (gtr_df
            .drop_duplicates('project_id')
            .dropna()
            )


def transform_gtr(gtr_df, min_length):
    """Tokenise Gateway to Research abstract texts

    Tokens added to `processed_documents` column.

    Args:
        gtr_df (pandas.DataFrame): Gateway to research data
        min_length (int): Minimum token length

    Returns:
        pandas.DataFrame
            Tokenised dataset
    """

    processed = (gtr_df.abstract_texts
                 .apply(tokenize_document, min_length, flatten=True)
                 .to_frame('processed_documents')
                 # Keep only documents with tokenised terms:
                 .assign(is_doc=lambda x: x.processed_documents.apply(len) > 0)
                 .query("is_doc > 0")
                 )

    return (gtr_df
            .join(processed)
            .dropna()
            )

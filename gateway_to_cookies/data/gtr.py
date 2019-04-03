from pandas import read_csv
from gateway_to_cookies.features.text_preprocessing import tokenize_document


def make_gtr(data_dir, nrows=None):
    """Clean and tokenise gateway to research abstract texts


    Args:
        data_dir (str): data directory
        nrows (int, optional): number of rows to use, defaults to None.

    """

    (read_csv(f"{data_dir}/raw/gtr_projects.csv", nrows=nrows)
     .pipe(clean_gtr)
     .pipe(transform_gtr)
     .to_csv(f"{data_dir}/processed/gtr_tokenised.csv")
     )


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


def transform_gtr(gtr_df):
    """Tokenise Gateway to Research abstract texts

    Args:
        gtr_df (pandas.DataFrame): Gateway to research data

    Returns:
        pandas.DataFrame

        Tokenised dataset
    """

    processed = (gtr_df.abstract_texts
                 .apply(tokenize_document, flatten=True)
                 .to_frame('processed_documents')
                 # Keep only documents with tokenised terms:
                 .assign(is_doc=lambda x: x.processed_documents.apply(len) > 0)
                 .query("is_doc > 0")
                 )

    return (gtr_df
            .join(processed)
            .dropna()
            )

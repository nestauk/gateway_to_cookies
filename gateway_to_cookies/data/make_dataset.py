# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
from gateway_to_cookies.data.gtr import make_gtr


@click.command()
@click.option('--nrows', '-n', default=1000, type=int)
def main(nrows):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).

    Args:
        nrows (int, optional): Number of rows of dataset to make. Defaults to 1000.
    """
    logger = logging.getLogger(__name__)

    logger.info(f'making gateway to research data set from raw data ({nrows} rows)')
    make_gtr(project_dir / 'data', nrows)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # Define project base directory
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    # NOTE: not used in this stub but often useful
    load_dotenv(find_dotenv())

    main()

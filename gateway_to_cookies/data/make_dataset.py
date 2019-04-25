import logging
import yaml
from pathlib import Path
from gateway_to_cookies.data.gtr import make_gtr

logger = logging.getLogger(__name__)


def main():
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    with open(project_dir / 'model_config.yaml', 'rt') as f:
        config = yaml.safe_load(f.read())['data']
    logger.info(f'Loaded data parameters: {config}')

    logger.info('Making gtr...')
    make_gtr(project_dir / 'data', config['gtr']['usecols'],
             config['gtr']['nrows'], config['gtr']['min_length'])


if __name__ == '__main__':
    # Define project base directory
    project_dir = Path(__file__).resolve().parents[2]

    try:
        main()
    except (Exception, KeyboardInterrupt) as e:
        logger.exception(e, stack_info=True)
        raise e

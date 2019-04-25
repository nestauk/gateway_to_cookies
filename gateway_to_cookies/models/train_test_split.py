import logging
import yaml
import gateway_to_cookies
from pathlib import Path

from pandas import read_csv
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)


def main():
    """ Performs test-train split """

    # Input datasets
    target_fin = f"{project_dir}/data/processed/gtr_tokenised.csv"
    embedding_fin = f"{project_dir}/data/processed/gtr_embedding.csv"
    # Training set output
    train_fout = f"{project_dir}/data/processed/gtr_train.csv"
    # Test set output
    test_fout = f"{project_dir}/data/processed/gtr_test.csv"

    with open(project_dir / 'model_config.yaml', 'rt') as f:
        config = yaml.safe_load(f.read())['split']
    target = config['target']
    logger.info(f'Loaded train-test split parameters: {config}')

    logger.info('load gateway to research data')
    Xy = (read_csv(target_fin, usecols=[target])
          .join(read_csv(embedding_fin, index_col=0))
          )

    msg = 'Building train-test split'
    logger.info(msg)

    X_train, X_test, y_train, y_test = train_test_split(
            Xy.drop(target, 1), Xy[target], **config['split'])

    X_train.join(y_train).to_csv(train_fout)
    msg = f'Saved training set to {train_fout}'
    logger.info(msg)

    X_test.join(y_test).to_csv(test_fout)
    msg = f'Saved test set to {test_fout}'
    logger.info(msg)


if __name__ == '__main__':
    # Define project base directory
    project_dir = Path(__file__).resolve().parents[2]

    try:
        main()
    except (Exception, KeyboardInterrupt) as e:
        logging.exception(e, stack_info=True)
        raise e

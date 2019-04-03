# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path

import pandas as pd
import joblib
from sklearn.metrics import accuracy_score


@click.command()
@click.option('--random_state', type=int, default=0)
@click.option('--target', type=str, default='funder_name')
def main(random_state, target):
    """Evaluates model metrics on test set.

    Outputs metrics to `models/metrics.txt`

    Args:
        random_state (int, RandomState instance or None, optional):
            If int, random_state is the seed used by the random number generator;
            If RandomState instance, random_state is the random number generator;
            If None, the random number generator is the RandomState instance used

        target (str, optional):
            The Gateway to Research column   by `np.random`. Defaults to 0.
    """

    logger = logging.getLogger(__name__)

    Xy = pd.read_csv(f"{project_dir}/data/processed/gtr_test.csv", index_col=0)
    logger.info(f"Loaded test data")
    X, y = Xy.drop(target, 1), Xy[target]

    clf = joblib.load(f"{project_dir}/models/gtr_forest.pkl")
    logger.info(f"Loaded model")

    accuracy = accuracy_score(y, clf.predict(X))
    logger.info(f"Test Accuracy: {accuracy}")

    metrics_fout = f"{project_dir}/models/metrics.txt"
    with open(metrics_fout, 'w') as f:
        f.write(f"{'gtr_clf'} accuracy: {accuracy:4f}\n")
    logger.info(f"Saved classifier to {metrics_fout}")


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # Define project base directory
    project_dir = Path(__file__).resolve().parents[2]

    main()

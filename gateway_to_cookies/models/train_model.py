# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path

import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

logger = logging.getLogger(__name__)

@click.command()
@click.option('--random_state', type=int, default=0)
@click.option('--target', type=str, default='funder_name')
def main(random_state, target):
    """ Runs model

    Args:

        random_state (int, RandomState instance or None, optional):
            If int, random_state is the seed used by the random number generator;
            If RandomState instance, random_state is the random number generator;
            If None, the random number generator is the RandomState instance used
            by `np.random`. Defaults to 0.

        target (str, optional):
            The Gateway to Research column name to use as a target
    """

    Xy = pd.read_csv(f"{project_dir}/data/processed/gtr_train.csv", index_col=0)
    logger.info(f"Loaded train data")
    X, y = Xy.drop(target, 1), Xy[target]

    logger.info(f"Training classifier...")
    clf = RandomForestClassifier(n_estimators=100, random_state=random_state)
    clf.fit(X, y)
    logger.info(f"Train Accuracy: {accuracy_score(y, clf.predict(X))}")

    clf_fout = f"{project_dir}/models/gtr_forest.pkl"
    with open(clf_fout, 'wb') as fd:
        joblib.dump(clf, fd)
    logger.info(f"Saved classifier to {clf_fout}")


if __name__ == '__main__':

    # Define project base directory
    project_dir = Path(__file__).resolve().parents[2]

    main()

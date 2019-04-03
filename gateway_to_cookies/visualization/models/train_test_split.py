# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path

from pandas import read_csv
from sklearn.model_selection import train_test_split


@click.command()
@click.option('--train_size', type=float, default=None)
@click.option('--test_size', type=float, default=0.2)
@click.option('--random_state', type=int, default=0)
@click.option('--target', type=str, default='funder_name')
def main(train_size, test_size, target, random_state):
    """ Performs test-train split

    Args:
        train_size (float, int, or None):
            If float, should be between 0.0 and 1.0 and represent the
            proportion of the dataset to include in the train split. If
            int, represents the absolute number of train samples. If None,
            the value is automatically set to the complement of the test size.
            Defaults to None.

        test_size (float, int or None, optional):
            If float, should be between 0.0 and 1.0 and represent the proportion
            of the dataset to include in the test split. If int, represents the
            absolute number of test samples. If None, the value is set to the
            complement of the train size. By default, the value is set to 0.25.
            The default will change in version 0.21. It will remain 0.25 only
            if ``train_size`` is unspecified, otherwise it will complement
            the specified ``train_size``. Defaults to 0.2

        random_state (int, RandomState instance or None, optional):
            If int, random_state is the seed used by the random number generator;
            If RandomState instance, random_state is the random number generator;
            If None, the random number generator is the RandomState instance used
            by `np.random`. Defaults to 0.

        target (str, optional):
            The Gateway to Research column name to use as a target

    """
    logger = logging.getLogger(__name__)

    logger.info('load gateway to research data')
    Xy = (read_csv(f"{project_dir}/data/processed/gtr_tokenised.csv", usecols=[target])
          .join(read_csv(f"{project_dir}/data/processed/gtr_embedding.csv", index_col=0))
          )

    msg = (f'Building train-test split ({train_size}, {test_size}) '
           f'with target `{target}` using seed: {random_state}')
    logging.info(msg)

    X_train, X_test, y_train, y_test = train_test_split(
            Xy.drop(target, 1), Xy[target],
            train_size=train_size, test_size=test_size,
            random_state=random_state, shuffle=True
            )

    train_fout = f"{project_dir}/data/processed/gtr_train.csv"
    test_fout = f"{project_dir}/data/processed/gtr_test.csv"

    X_train.join(y_train).to_csv(train_fout)
    msg = f'Saving training set to {train_fout}'
    logging.info(msg)

    X_test.join(y_test).to_csv(test_fout)
    msg = f'Saving test set to {test_fout}'
    logging.info(msg)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # Define project base directory
    project_dir = Path(__file__).resolve().parents[2]

    main()

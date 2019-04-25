import logging
import yaml
import gateway_to_cookies
from pathlib import Path
from pandas import read_csv
from joblib import dump
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

logger = logging.getLogger(__name__)


def main():
    """ Runs RandomForestClassifier on training set"""

    # Train set input
    train_fin = f"{project_dir}/data/processed/gtr_train.csv"
    # Model output
    clf_fout = f"{project_dir}/models/gtr_forest.pkl"

    with open(project_dir / 'model_config.yaml', 'rt') as f:
        config = yaml.safe_load(f.read())['model']
    target = config['target']
    logger.info(f'Loaded train-test split parameters: {config}')

    Xy = read_csv(train_fin, index_col=0)
    logger.info(f"Loaded train data")
    X, y = Xy.drop(target, 1), Xy[target]

    logger.info(f"Training classifier...")
    # NOTE: In reality the model pipeline would be more sophisticated
    # and would likely exist in it's own file
    clf = RandomForestClassifier(**config['hyperparameters'])
    clf.fit(X, y)
    logger.info(f"Train Accuracy: {accuracy_score(y, clf.predict(X))}")

    with open(clf_fout, 'wb') as fd:
        dump(clf, fd)
    logger.info(f"Saved classifier to {clf_fout}")


if __name__ == '__main__':
    # Define project base directory
    project_dir = Path(__file__).resolve().parents[2]

    try:
        main()
    except (Exception, KeyboardInterrupt) as e:
        logging.exception(e, stack_info=True)
        raise e

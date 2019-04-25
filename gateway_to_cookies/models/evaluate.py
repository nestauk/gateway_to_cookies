import logging
import yaml
import gateway_to_cookies
from pathlib import Path
from pandas import read_csv
from joblib import load
from sklearn.metrics import accuracy_score

logger = logging.getLogger(__name__)


def main():
    """Evaluates model metrics on test set.

    Outputs metrics to `models/metrics.txt`
    """

    test_fin = f"{project_dir}/data/processed/gtr_test.csv"
    clf_fin = f"{project_dir}/models/gtr_forest.pkl"
    metrics_fout = f"{project_dir}/models/metrics.txt"

    with open(project_dir / 'model_config.yaml', 'rt') as f:
        config = yaml.safe_load(f.read())['evaluate']
    target = config['target']
    logger.info(f'Loaded train-test split parameters: {config}')

    Xy = read_csv(test_fin, index_col=0)
    logger.info(f"Loaded test data")
    X, y = Xy.drop(target, 1), Xy[target]

    clf = load(clf_fin)
    logger.info(f"Loaded model")

    accuracy = accuracy_score(y, clf.predict(X))
    logger.info(f"Test Accuracy: {accuracy}")

    with open(metrics_fout, 'w') as f:
        f.write(f"{'gtr_clf'} accuracy: {accuracy:4f}\n")
    logger.info(f"Saved classifier to {metrics_fout}")


if __name__ == '__main__':
    # Define project base directory
    project_dir = Path(__file__).resolve().parents[2]

    try:
        main()
    except (Exception, KeyboardInterrupt) as e:
        logging.exception(e, stack_info=True)
        raise e

import click
import joblib
import pandas as pd

from datetime import datetime


from data import make_dataset
from feature import make_features
from models import make_model
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import KFold, train_test_split


@click.group()
def cli():
    pass


@click.command()
@click.option("--input_filename", default="data/raw/train.csv", help="File training data")
@click.option("--model_dump_filename", default="models/dump.json", help="File to dump model")


def train(input_filename: pd.DataFrame, model_dump_filename: str = "models/dump.json", model=None):
    if isinstance(input_filename, str):
        input = make_dataset(input_filename)
    elif not isinstance(input_filename, pd.DataFrame):
        raise TypeError("Input must be either a dataframe or a string")

    df = pd.read_csv(input)
    
    if model is None:
        model = make_model()

    model.fit(df)

    return joblib.dump(model, model_dump_filename)




@click.command()
@click.option("--input_filename", default="data/raw/train.csv", help="File training data")
@click.option("--model_dump_filename", default="models/dump.json", help="File to dump model")
@click.option("--output_filename", default="data/processed/prediction.csv", help="Output file for predictions")


def predict(model_dump_filename,
            input_filename="data/raw/test.csv",
            output_filename: str = "data/processed/prediction.csv"):
    if isinstance(input_filename, str):
        input_filename = make_dataset(input_filename)

    elif not isinstance(input_filename, pd.DataFrame):
        raise TypeError("Input must be either a dataframe or a string")

    model = joblib.load(model_dump_filename)
    dataset = input_filename

    result = model.predict(dataset)

    result.to_csv(output_filename, index=False)




@click.command()
@click.option("--input_filename", default="data/raw/train.csv", help="File training data")


def evaluate(input_filename):
    # Read CSV
    df = make_dataset(input_filename)

    # Make features (tokenization, lowercase, stopwords, stemming...)
    X, y = make_features(df)

    # Object with .fit, .predict methods
    model = make_model()        # TODO : ajouter une str

    # Run k-fold cross validation. Print results
    return evaluate_model(model, X, y)


def evaluate_model(model: str,
                   X,
                   y,
                   nb_split: int = 3):
    # Run k-fold cross validation. Print results
    kf = KFold(n_splits=nb_split)

    for train_df, test_df in kf.split(X):
        model_name = "models/" + datetime.now().strftime("%Y%m%d_%H%M%S") + ".json"
        trained_model = train(input_filename=train_df, model=model, model_dump_filename=model_name)
        predict(input=test_df, model_dump_filename=trained_model)




cli.add_command(train)
cli.add_command(predict)
cli.add_command(evaluate)


if __name__ == "__main__":
    cli()

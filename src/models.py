from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier


def make_model(model_str: str = "random_forest"):
    """
    Return selected model.
    """
    if model_str.lower() == "random_forest":
        return RandomForestClassifier()

    elif model_str.lower() == "naive_bayes":
        return GaussianNB()

    elif model_str.lower() == "decision_tree":
        return DecisionTreeClassifier()
    else:
        raise ValueError(f"Model '{model_str}' is not recognized.")

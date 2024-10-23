from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd


def make_features(df):
    # # TODO :
    # y = df["is_comic"]

    # # Fake X: constant equal to 0
    # X = [[0]] * len(y)

    one_hot = CountVectorizer(lowercase=True)
    count_matrix = one_hot.fit_transform(df["video_name"])
    count_array = count_matrix.toarray()
    X = pd.DataFrame(data=count_array,columns = one_hot.get_feature_names_out())
    y = df["is_comic"]
    return X, y





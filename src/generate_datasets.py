import random


from src.data import make_dataset


def split_dataset(file_path: str,
                  ratio_train: float = 1 / 3,
                  ratio_predict: float = 1 / 3,
                  ratio_evaluate: float = 1 / 3,
                  rand: int = None) -> None:
    """

    """
    if (ratio_train + ratio_predict + ratio_evaluate != 1
            or ratio_train == 0
            or ratio_predict == 0
            or ratio_evaluate == 0):
        ratio_train = 1 / 3
        ratio_predict = 1 / 3
        ratio_evaluate = 1 / 3

    if rand is None:
        rand = random.randint(0, 1_000)
    dataset = make_dataset(file_path)
    shuffled_dataset = dataset.sample(frac=1, random_state=rand)
    featured_dataset = make_dataset(shuffled_dataset)

    len_df = len(featured_dataset)
    train_dataset = featured_dataset[0: int(ratio_train * len_df)]
    evaluate_dataset = featured_dataset[int(ratio_predict * len_df) : int(ratio_predict * len_df) + int(ratio_evaluate * len_df)]
    predict_dataset = featured_dataset[int(ratio_predict * len_df) + int(ratio_evaluate * len_df) : ]

    save_path = file_path.rsplit("/", 1)[0]
    train_dataset.to_csv(save_path + "train.csv", index=False, mode='w')
    predict_dataset.to_csv(save_path + "testp.csv", index=False, mode='w')
    evaluate_dataset.to_csv(save_path + "evaluate.csv", index=False, mode='w')


if __name__ == '__main__':
    file_path = "../data/raw/names_train.csv"
    split_dataset(file_path)

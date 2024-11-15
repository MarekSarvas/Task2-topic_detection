""" Prepares dialogue dataset from DeepPavlov Topics(https://deeppavlov.ai/datasets/topics) 
dataset by selecting 100 topics.
"""
import random
import argparse
from pathlib import Path

import pandas as pd

from categories import SELECTED_CATEGORIES


def parse_args():
    """ Return arguments for dataset generation when the module is run as main.

    Returns:
        arguments (argparse.Namespace): Program arguments
    """
    parser = argparse.ArgumentParser(description="Parse arguments for dataset creation.")

    parser.add_argument(
        "--data",
        type=str,
        default="data/dp_topics_downsampled_data/valid.csv",
        help="Path to checkpoint of trained model."
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="data/callcenter",
        help="Path to the validation data JSON file."
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=100,
        help="Number of samples to select from bigger dataset."
    )
    return parser.parse_args()


def select_dialogs(data_path: str, n: int = 100) -> pd.DataFrame:
    """ Select 100 "dialogues" from the dataset by randomly choosing 
    data from categories in SELECTED_CATEGORIES.

    Args:
        data_path (str): Path to csv with Deeppavlov topics.
        n (int, optional): Number of "dialogues". Defaults to 100.

    Raises:
        Exception: Check if there is enough dialogues to choose from
                    after the topic filtering.

    Returns:
        df_n (pd.DataFrame): Created dataset of topics with 
                            "text", "topic" and "id" columns. 
    """
    # Load data and select chosen topics
    df = pd.read_csv(data_path)
    df = df[df["topic"].isin(SELECTED_CATEGORIES)]

    if df.size < n:
        raise Exception(f"Not enough data to sample {n} dialogues")

    # select 'n' random dialogues
    df_n = df.sample(n=n, random_state=42).reset_index(drop=True)
    df_n["id"] = range(0, len(df_n))

    return df_n


def main(args):
    """ Select dialogues, and save the created dataset as csv file.

    Args:
        args (argparse.Namespace): Program arguments.
    """
    df = select_dialogs(args.data, n=args.n_samples)

    callcenter_datadir_path = Path(args.out_dir)

    if not callcenter_datadir_path.exists():
        callcenter_datadir_path.mkdir(parents=True, exist_ok=True)
        print(f"Directory '{callcenter_datadir_path}' created.")

    callcenter_data_path = callcenter_datadir_path.joinpath(f"data_{args.n_samples}.csv")
    print(f"Saving created dataset into {callcenter_data_path}")

    df.to_csv(callcenter_data_path, index=False)


if __name__ == "__main__":
    random.seed(42)
    args = parse_args()
    main(args)

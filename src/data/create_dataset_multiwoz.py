""" Prepares dialogue dataset from Multiwoz dataset by selecting 100 dialogues.
"""
import random
import json
import argparse
from typing import Dict, List
from collections import Counter
from pathlib import Path

import pandas as pd



def parse_args():
    """ Return arguments for dataset generation when the module is run as main.

    Returns:
        arguments (argparse.Namespace): Program arguments
    """
    parser = argparse.ArgumentParser(description="Parse arguments for dataset creation.")

    parser.add_argument(
        "--multiwoz",
        type=str,
        default="data/MultiWOZ_2.1/data.json",
        help="Path to checkpoint of trained model."
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="data/callcenter_multiwoz",
        help="Path to the validation data JSON file."
    )
    parser.add_argument(
        "--join_turns",
        action="store_true",
        help="If set, every turn in dialogue is concatenated into one string."
    )
    return parser.parse_args()


def extract_turn_label(act_metadata: Dict) -> str:
    """Extracts topic of each turn from the conversation for future purposes.

    Args:
        act_metadata (Dict): One turn of dialogue containing info about the turn topic

    Returns:
        turn_label (str): Turn topic or empty string for general or no topic.
    """
    turn_labels = []
    for key, _ in act_metadata.items():
        turn_labels.append(key.split("-")[0])

    # most_common returns List[Tuple[str, int]] ->
    # take most common item and the name of the label
    if len(turn_labels) > 0:
        turn_label = Counter(turn_labels).most_common()[0][0]
    else:
        turn_label = ""

    # remove "general" label
    if turn_label == "general":
        return ""
    return turn_label


def parse_dialog(dialog: List[Dict], join_turns: bool = False) -> pd.DataFrame:
    """ Creates dataframe with 'text' and 'topic' columns from each dialogue. 
    If 'join_turns' is set to True, the dataframe has only one row. Otherwise 
    number of rows is equal to number of turns in the dialogue.

    Args:
        dialog (List[Dict]): Dialogue  consisting of list of turn,
                            where each turn is Dict of text and other metadata.
        join_turns (bool, optional): If all turns should be concatenated into one string. 
                                    Defaults to False.

    Returns:
        dialog_data_df (pd.DataFrame): Dialogue text and corresponding topic.
    """
    turns_data = []

    for turn in dialog:
        turn_label = extract_turn_label(turn["dialog_act"])
        turns_data.append({"text": turn["text"].strip(), "topic": turn_label})

    dialog_data_df = pd.DataFrame(turns_data)

    # Select the label as the most common topic during the dialogue.
    if join_turns:
        # Most_common returns List[Tuple[str, int]] ->
        # take most common item and the name of the label
        topics = [t for t in dialog_data_df["topic"].to_list() if t != ""]
        topic = Counter(topics).most_common()[0][0]

        # Concatenate text and asign topic.
        dialog_data_df["topic"] = topic
        dialog_data_df = dialog_data_df.groupby('topic').agg({
            'text': " ".join,
        }).reset_index()

    return dialog_data_df


def main(args):
    """ Select dialogues, and save the created dataset as csv file.

    Args:
        args (argparse.Namespace): Program arguments.
    """
    with open(args.multiwoz, encoding="utf-8") as f:
        data = json.load(f)

    # 1. create dataset from multiwoz by selecting random 100 dialogues
    cleaned_data = []
    # get 100 dialog "transcriptions"
    dialogue_ids = random.sample(list(data.keys()), 100)
    for dial_id in dialogue_ids:
        dialogue_data = parse_dialog(data[dial_id]["log"], join_turns=args.join_turns)
        # add id into the data and add into list
        cleaned_data.append(dialogue_data)

    # 2. Create final Dataframe 
    df = pd.concat(cleaned_data)
    df["id"] = [i for i in range(0, len(df))]

    # 3. Save the dataframe to disk
    callcenter_datadir_path = Path(args.out_dir)
    if not callcenter_datadir_path.exists():
        callcenter_datadir_path.mkdir(parents=True, exist_ok=True)
        print(f"Directory '{callcenter_datadir_path}' created.")

    callcenter_data_path = callcenter_datadir_path.joinpath(
        f"data_100{"_joined" if args.join_turns else ""}.csv")
    print(f"Saving created dataset into {callcenter_data_path}")

    df.to_csv(callcenter_data_path, index=False, encoding='utf-8')


if __name__ == "__main__":
    random.seed(42)
    args = parse_args()
    main(args)

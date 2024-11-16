""" Module implementing topic detection pipeline. Consisting of: 
1. embedding clustering
2. key-word/key-phrase extraction

"""
import random
import argparse
from pathlib import Path
from typing import List
from collections import Counter

import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
import nltk
import spacy
from nltk.stem import WordNetLemmatizer
import pytextrank


from keyphrase_extraction import KeyphraseExtractionPipeline


def parse_args():
    parser = argparse.ArgumentParser(description="Parse data and hyper params for topic detection.")

    parser.add_argument(
        "--call_transcripts",
        type=str,
        default="data/callcenter_multiwoz/data_100.csv",
        help="Path to csv containing callcenter transcriptions and corresponding labels."
    )
    parser.add_argument(
        "--embeddings",
        type=str,
        default="data/callcenter_multiwoz/embeddings_20.npy",
        help="Path to the embeddings file."
    )
    parser.add_argument(
        "--eps",
        type=float,
        default=0.0006346702575683594,
        help="Path to the embeddings file."
    )
    parser.add_argument(
        "--min_samples",
        type=int,
        default=40,
        help="Path to the embeddings file."
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="multiwoz_topics",
        help="Path to directory where the info about discovered topics will be saved."
    )

    return parser.parse_args()


def cluster_dbscan(embeddings: np.ndarray, eps: float, min_samples: int) -> DBSCAN:
    """ Discover topic clusters using DBSCAN

    Args:
        embeddings (np.ndarray): Embeddings extracted from text data.
        eps (float): Eps param for DBSCAN clustering
        min_samples (int): Min_samples param for DBSCAN clustering

    Returns:
        DBSCAN: _description_
    """
    dbscan_model = DBSCAN(eps=eps,
                          min_samples=min_samples,
                          metric="cosine")
    clusters = dbscan_model.fit(embeddings)
    # mean = 0.0004064719832967967
    # best = 0.0006346702575683594

    return clusters


def group_clusters(df: pd.DataFrame, labels: np.ndarray) -> pd.DataFrame:
    """ Cluster dialogues together based on the output of clustering method.

    Args:
        df (pd.DataFrame): Call center data.
        labels (np.ndarray): Cluster labels predicted by the clustering algorithm.

    Returns:
        pd.DataFrame: Callcenter data aggregated by the cluster ID.
    """
    df["cluster_id"] = labels
    grouped = df.groupby('cluster_id').agg({
        'text': lambda x: list(x),
        'topic': lambda x: list(x),
        'id': lambda x: list(x)
    }).reset_index()

    return grouped


def normalize_text(text: str) -> List[str]:
    """ Normalize dialogue text by lowercasing, removing stopwords 
    and using nlt lemmatizer and stemmer.

    Args:
        text (str): Turn (or dialogue) text.

    Returns:
        List[str]: Normalized text.
    """
    # hack to remove whitespaces
    text = " ".join(text.split())
    text = text.lower()

    lemmatizer = WordNetLemmatizer()
    text = lemmatizer.lemmatize(text)

    sno = nltk.stem.SnowballStemmer('english')
    text = sno.stem(text)

    stopwords = nltk.corpus.stopwords.words('english')
    text = [w for w in text.split(" ") if w not in stopwords]

    return text


def extract_keywords(ds: pd.Series, topic_file: Path, n_common: int = 10):
    """ Extracts keywords and/or key-phrases from every utterance of the dialogue,
    or from the whole dialogue (depends on the dataset).

    Args:
        ds (pd.Series): Text data for one turn (or dialogue)
        topic_file (Path): Path to file where key-phrases and keywords will be stored.
        n_common (int): How many of the most common keywords to store.
    """
    text = " ".join(ds['text'])
    text = normalize_text(text)

    # 1. Extracts n most common words
    keywords = Counter(text).most_common()[:n_common]
    with open(topic_file, "w", encoding="utf-8") as f:
        f.write("Most common words in this topic:\n")
        tmp_keywords = ", ".join([keyword[0] for keyword in keywords])
        f.write(tmp_keywords+"\n")

        # 2. Extracts 10 top-ranked key-phrases using spacy
        nlp = spacy.load("en_core_web_sm")
        nlp.add_pipe("textrank")
        doc = nlp(" ".join(text))

        f.write("\nKey-phrases extracted with nltk:\n")
        for i, phrase in enumerate(doc._.phrases):
            if i > 10:
                break
            #print(phrase.text, phrase.rank)
            f.write(f"{phrase.text}\n")

        f.write("\nKey-words extracted with pre-trained model:\n")
        # 3. Extracts keywords using pre-trained transformer model
        model_name = "ml6team/keyphrase-extraction-distilbert-inspec"
        extractor = KeyphraseExtractionPipeline(model=model_name, device="cuda")
        keywords = extractor(" ".join(text))
        tmp_keywords = ", ".join([keyword for keyword in keywords])
        f.write(tmp_keywords+"\n")


def main(args):
    metadata = pd.read_csv(args.call_transcripts, encoding="utf-8")
    embeddings = np.load(args.embeddings)

    clusters = cluster_dbscan(embeddings, args.eps, args.min_samples)

    # group text of dialogues based on the predicted cluster label
    grouped_df = group_clusters(metadata, clusters.labels_)
    grouped_df["cluster_size"] = grouped_df["id"].apply(len)
    print(grouped_df[["cluster_id", "cluster_size"]])

    # remove outliers if there are many to prevent from taking it
    # as the biggest cluster
    grouped_df = grouped_df[grouped_df["cluster_id"] != -1]
    # get biggest clusters
    common_topics = grouped_df.nlargest(5, "cluster_size")["cluster_id"].to_list()

    topic_dir = Path(args.out_dir)
    topic_dir.mkdir(parents=True, exist_ok=True)

    print(f"Extracting topics into {args.out_dir} dir")
    for i, c_id in enumerate(common_topics):
        topic_file = topic_dir / f"topic_{i}.txt"
        #print(f"Cluster: {c_id}")
        extract_keywords(grouped_df.iloc[c_id], topic_file)
        label_file = topic_dir / f"topic_{i}_labels.txt"
        with open(label_file, "w", encoding="utf-8") as f:
            labels = [str(topic) for topic in grouped_df.iloc[c_id]["topic"]]
            labels = ", ".join(labels)
            f.write(labels + "\n")
        #print("="*30)


if __name__ == "__main__":
    random.seed(42)
    nltk.download('stopwords')
    nltk.download('punkt_tab')
    nltk.download('wordnet')

    args = parse_args()
    main(args)

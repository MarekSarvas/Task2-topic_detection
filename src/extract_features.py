""" Module implementing feature extraction from text data using Huggingface
transformers library and pre-trained sentence embedding model.

"""
import random
import argparse
from pathlib import Path

import tqdm
import numpy as np
import torch
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer
import umap.umap_ as umap

from data.dataloaders import TextDataset


def parse_args():
    """ Parses and returns program arguments and hyper-parameters for 
    feature extraction from dialogues.

    Returns:
        arguments (argparse.Namespace): Program arguments
    """
    parser = argparse.ArgumentParser(
            description="Parse data and hyper params for embedding extraction."
    )

    parser.add_argument(
        "--call_transcripts",
        type=str,
        default="data/callcenter/data_100.csv",
        help="Path to csv containing callcenter transcriptions and corresponding labels."
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="data/callcenter",
        help="Path to the dir where extracted embeddings will be stored."
    )
    parser.add_argument(
        "--model",
        type=str,
        default="sentence-transformers/all-mpnet-base-v2",
        help="Name of the model to use from Huggingface."
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for embedding extraction."
    )
    parser.add_argument(
        "--reduce_to",
        type=int,
        default=32,
        help="Size of the reduced embeddings."
    )
    return parser.parse_args()


def load_model(model_name: str) -> SentenceTransformer:
    """

    Args:
        model_name (str): Name of the pre-trained huggingface model

    Returns:
        model (SentenceTransformer): Pre-trained huggingface model
    """
    # tokenizer = AutoTokenizer.from_pretrained(model_name)
    # model = AutoModel.from_pretrained(model_name)
    model = SentenceTransformer(model_name)
    model = model.eval()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    return model


def dim_reduction(embeddings_raw: np.ndarray, reduce_to: int) -> np.ndarray:
    """_summary_

    Args:
        embeddings_raw (np.ndarray): Text embeddings extracted by the huggingface model.
        reduce_to (int): Dimension of the new reduced embeddings. 

    Returns:
        np.ndarray: New embeddings with the size of (N, reduce_to)
    """
    umap_model = umap.UMAP(n_neighbors=10,
                           n_components=reduce_to,
                           metric="cosine",
                           min_dist=0,
                           spread=1,
                           random_state=42)

    new_embeddings = umap_model.fit_transform(embeddings_raw)
    return new_embeddings



def main(args):
    """Extracts embeddings from the text data, and perform dimensionality
    reduction. Both embeddings are stored in the 'args.out_dir' as numpy
    ndarrays embeddings_raw.npy and embeddings_<reduce_to>.npy, respectively.


    Args:
        args (_type_): _description_
    """
    # 1. initialize model and dataset
    model = load_model(args.model)
    data = TextDataset(csv_path=args.call_transcripts)
    text_dataloader = DataLoader(data, batch_size=args.batch_size, shuffle=False)

    embedding_file_path = Path(args.out_dir).joinpath("embeddings_raw.npy")
    # 2. extract embeddings from dialogue text using the pre-trained model
    embeddings_arr = []
    for (_, text) in tqdm.tqdm(text_dataloader, total=len(text_dataloader)):
        text = list(text)
        embeddings = model.encode(text)
        embeddings_arr.append(embeddings)

    # 3. save the embeddings as a numpy ndarray with size (dataset_size, 768)
    embeddings_arr = np.concatenate(embeddings_arr, axis=0)
    with open(embedding_file_path, "wb") as f:
        np.save(f, embeddings_arr)

    # 4. perform dimensionality reduction
    if args.reduce_to > 0:
        reduced_embeddings = dim_reduction(embeddings_arr, args.reduce_to)
        print(f"Reduced embeddings to size: {reduced_embeddings.shape}")
    else:
        reduced_embeddings = embeddings_arr
        print(f"Keeping the embeddings size of: {embeddings_arr.shape}")

    # 5. save the reduced embeddings as a numpy ndarray with size (dataset_size, args.reduce_to)
    reduced_embedding_file_path = Path(args.out_dir).joinpath(f"embeddings_{args.reduce_to}.npy")
    with open(reduced_embedding_file_path, "wb") as f:
        np.save(f, reduced_embeddings)


if __name__ == "__main__":
    random.seed(42)
    args = parse_args()
    main(args)

# Dataset creation
1. Create **data/** directory
2. Download multiwoz data from `https://github.com/budzianowski/multiwoz/tree/master` or DP topics from `https://deeppavlov.ai/datasets/topics` unzip into **data/** dir.
3. Create conversation dataset from the unzip data by runing **src/data/create_dataset.py** or **src/data/create_multiwoz_dataset.py**, e.g.:
```bash
python3 src/data/create_multiwoz_dataset.py \
    --multiwoz data/MultiWOZ_2.1/data.json \
    --out_dir data/callcenter_multiwoz
```

# Topic Detection
1. Extract the embeddings from the previously created dataset. The full 768dim embeddings will be stored as numpy array in **data/callcenter_multiwoz/embeddings_raw.npy** and reduced embeddings will be sotred as **data/callcenter_multiwoz/embeddings_<reduce_to>.npy**.
```bash
python3 src/extract_features.py \
    --call_transcripts data/callcenter_multiwoz/data_100.csv \
    --out_dir data/callcenter_multiwoz \
    --batch_size 32 \
    --reduce_to 20 
```

2. Test the best embedding size and compute eps parameter for the dbscan by generating various embedding sizes and run
```bash
python3 src/dbscan.py --embedding_sizes 2 20 32 64
```


2. Cluster embeddings and discover topics and extract key-words/phrases that will be stored in **multiwoz_topics/** dir by default:
```bash
python3 src/topic_detection.py
```

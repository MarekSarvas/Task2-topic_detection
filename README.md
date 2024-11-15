# Dataset creation
1. Download multiwoz data from `https://github.com/budzianowski/multiwoz/tree/master` or DP topics from `https://deeppavlov.ai/datasets/topics` unzip into data/ dir.
2. Create conversation dataset from the unzip data by runing **src/data/create_dataset.py** or **src/data/create_multiwoz_dataset.py**, e.g.:
    ```python
    python src/data/create_multiwoz_dataset.py \
        --multiwoz data/MultiWOZ_2.1/data.json \
        --out_dir data/callcenter_multiwoz
    ```


import os
from dataclasses import dataclass
from io import BytesIO
from typing import Optional

import torch
import glob
import pandas as pd
from PIL import Image
from accelerate.logging import get_logger
from datasets import load_from_disk, load_dataset, Dataset
from hydra.utils import instantiate
from omegaconf import II

from trainer.datasetss.base_dataset import BaseDataset, BaseDatasetConfig
from trainer.datasetss.clip_hf_dataset import ProcessorConfig, CLIPHFDataset

logger = get_logger(__name__)

@dataclass
class CLIPCustomDatasetConfig(BaseDatasetConfig):
    _target_: str = "trainer.datasetss.clip_custom_dataset.CLIPCustomDataset"
    dataset_name: str = "exaone"
    dataset_config_name: str = "null"
    dataset_root: str = "/home/data/exaone_gen_imgs"
    annotation_dir: str = "annotations"

    from_disk: bool = False
    train_split_key: str = "train_custom"
    valid_split_key: str = "validation_unique_custom"
    test_split_key: str = "test_unique_custom"
    train_split_name: str = 'train'
    valid_split_name: str = 'valid'
    test_split_name: str ='test'
    cache_dir: Optional[str] = None

    caption_column_name: str = "caption"
    input_ids_column_name: str = "input_ids"
    image_0_column_name: str = "jpg_0"
    image_1_column_name: str = "jpg_1"
    label_0_column_name: str = "label_0"
    label_1_column_name: str = "label_1"
    are_different_column_name: str = "are_different"
    has_label_column_name: str = "has_label"

    pixels_0_column_name: str = "pixel_values_0"
    pixels_1_column_name: str = "pixel_values_1"

    image_0_uid_column_name: str = 'image_0_uid'
    image_1_uid_column_name: str = 'image_1_uid'

    num_examples_per_prompt_column_name: str = "num_example_per_prompt"

    keep_only_different: bool = False
    keep_only_with_label: bool = False
    keep_only_with_label_in_non_train: bool = True

    processor: ProcessorConfig = ProcessorConfig()

    limit_examples_per_prompt: int = -1

    only_on_best: bool = False

class CLIPCustomDataset(CLIPHFDataset):

    def __init__(self, cfg: CLIPCustomDatasetConfig, split: str = "train"):
        self.cfg = cfg
        self.split = split
        
        assert not self.cfg.only_on_best, "only_on_best is not supported yet"
        
        logger.info(f"Loading {self.split} dataset")

        annotation_dir = os.path.join(cfg.dataset_root, cfg.annotation_dir)
        self.dataset = self.load_local_dataset(annotation_dir, self.split)
        # self.dataset = self.load_local_dataset(self.dataset_root, self.split)
        logger.info(f"Loaded {len(self.dataset)} examples from {self.split} dataset")

        if self.cfg.keep_only_different:
            self.dataset = self.dataset.filter(lambda x: x[self.cfg.are_different_column_name])

        # TODO: if not self.cfg.keep_only_with_label, make label 0.5 for the samples with are_different=False  
        if self.cfg.keep_only_with_label:
            logger.info(f"Keeping only examples with label")
            self.dataset = self.dataset.filter(lambda x: x[self.cfg.has_label_column_name])
            logger.info(f"Kept {len(self.dataset)} examples from {self.split} dataset")
        elif self.cfg.keep_only_with_label_in_non_train and self.split != self.cfg.train_split_name:
            logger.info(f"Keeping only examples with label in {self.split} split")
            self.dataset = self.dataset.filter(lambda x: x[self.cfg.has_label_column_name])
            logger.info(f"Kept {len(self.dataset)} examples from {self.split} dataset")

        if self.cfg.limit_examples_per_prompt > 0:
            logger.info(f"Limiting examples per prompt to {self.cfg.limit_examples_per_prompt}")
            df = self.dataset.to_pandas()
            # df = df.drop('__index_level_0__', axis=1)
            logger.info(f"Loaded {len(df)} examples from {self.split} dataset")
            df = df.groupby(self.cfg.caption_column_name).head(self.cfg.limit_examples_per_prompt)
            logger.info(f"Kept {len(df)} examples from {self.split} dataset")
            self.dataset = Dataset.from_pandas(df)

        if self.cfg.only_on_best and self.split == self.cfg.train_split_name:
            logger.info(f"Keeping only best examples for training")
            train_dataset = self.dataset.remove_columns([self.cfg.image_0_column_name, self.cfg.image_1_column_name])
            df = train_dataset.to_pandas()
            df = df[df[self.cfg.has_label_column_name] == 1]
            image_0_wins_df = df[df[self.cfg.label_0_column_name] == 1]
            image_1_wins_df = df[df[self.cfg.label_0_column_name] == 0]
            bad_image_0_to_good_image_1 = dict(zip(image_1_wins_df.image_0_uid, image_1_wins_df.image_1_uid))
            bad_image_1_to_good_image_0 = dict(zip(image_0_wins_df.image_1_uid, image_0_wins_df.image_0_uid))
            bad_images_uids2good_images_uids = bad_image_0_to_good_image_1 | bad_image_1_to_good_image_0
            image_0_uid2image_col_name = dict(zip(df.image_0_uid, [self.cfg.image_0_column_name] * len(df.image_0_uid)))
            image_1_uid2image_col_name = dict(zip(df.image_1_uid, [self.cfg.image_1_column_name] * len(df.image_1_uid)))
            uid2image_col_name = image_0_uid2image_col_name | image_1_uid2image_col_name

            bad_uids = set()
            for bad_image, good_image in bad_images_uids2good_images_uids.items():
                cur_good = {bad_image}
                while good_image in bad_images_uids2good_images_uids:
                    if good_image in cur_good:
                        bad_uids.add(bad_image)
                        break
                    cur_good.add(good_image)
                    good_image = bad_images_uids2good_images_uids[good_image]
                bad_images_uids2good_images_uids[bad_image] = good_image

            df = df[~(df.image_0_uid.isin(bad_uids) | df.image_1_uid.isin(bad_uids))]
            keep_ids = df.index.tolist()
            self.dataset = self.dataset.select(keep_ids)
            new_ids = list(range(len(df)))
            uid2index = dict(zip(df.image_0_uid, new_ids)) | dict(zip(df.image_1_uid, new_ids))
            logger.info(f"Kept only {len(self.dataset)} best examples for training")
            self.bad_images_uids2good_images_uids = bad_images_uids2good_images_uids
            self.uid2index = uid2index
            self.uid2image_col_name = uid2image_col_name

        logger.info(f"Loaded {len(self.dataset)} examples from {self.split} dataset")

        processor = instantiate(cfg.processor)
        self.tokenizer = processor.tokenizer
        self.image_processor = processor.image_processor

    def load_local_dataset(self, path: str, split: str):
        files = glob.glob(os.path.join(path, f"{split}*.parquet"))
        data = [pd.read_parquet(f,engine='pyarrow') for f in files] # do not use engine=fastparquet
        merged_data = pd.concat(data, ignore_index=True)
        return Dataset.from_pandas(merged_data)

    def process_image(self, image):
        if isinstance(image, dict):
            image = image["bytes"]
        if isinstance(image, bytes):
            image = Image.open(BytesIO(image))
        if isinstance(image, str):
            image = Image.open(image)
        image = image.convert("RGB")
        pixel_values = self.image_processor(image, return_tensors="pt")["pixel_values"]
        return pixel_values

    def __getitem__(self, idx):
        example = self.dataset[idx]

        if self.cfg.only_on_best and self.split == self.cfg.train_split_name:
            if example[self.cfg.label_0_column_name]:
                bad_image_uid = example["image_1_uid"]
                good_image_column_name = self.cfg.image_0_column_name
            else:
                bad_image_uid = example["image_0_uid"]
                good_image_column_name = self.cfg.image_1_column_name
            good_image_uid = self.bad_images_uids2good_images_uids[bad_image_uid]
            good_image_index = self.uid2index[good_image_uid]
            example[good_image_column_name] = self.dataset[good_image_index][self.uid2image_col_name[good_image_uid]]

        # TODO: remove type casting after changing data type in parquet file
        if not example[self.cfg.has_label_column_name] and ((not self.cfg.keep_only_with_label_in_non_train and self.split != self.cfg.train_split_name) or (not self.cfg.keep_only_with_label and self.split == self.cfg.train_split_name)):
            label_0 = torch.tensor(0.5)[None]
            label_1 = torch.tensor(0.5)[None]
        else:
            label_0 = torch.tensor(example[self.cfg.label_0_column_name])[None]
            label_1 = torch.tensor(example[self.cfg.label_1_column_name])[None]

        # if self.split == self.cfg.train_split_name:
        #     print(label_0, label_1)

        input_ids = self.tokenize(example)

        pixel_0_values = self.process_image(os.path.join(self.cfg.dataset_root, example[self.cfg.image_0_uid_column_name] + '.png'))
        pixel_1_values = self.process_image(os.path.join(self.cfg.dataset_root, example[self.cfg.image_1_uid_column_name] + '.png'))

        item = {
            self.cfg.input_ids_column_name: input_ids,
            self.cfg.pixels_0_column_name: pixel_0_values,
            self.cfg.pixels_1_column_name: pixel_1_values,
            self.cfg.label_0_column_name: label_0,
            self.cfg.label_1_column_name: label_1,
            self.cfg.num_examples_per_prompt_column_name: torch.tensor(example[self.cfg.num_examples_per_prompt_column_name])[None],
        }
        return item
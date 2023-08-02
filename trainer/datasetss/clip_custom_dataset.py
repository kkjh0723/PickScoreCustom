from dataclasses import dataclass
from io import BytesIO
from typing import Optional

import torch
from PIL import Image
from accelerate.logging import get_logger
from datasets import load_from_disk, load_dataset, Dataset
from hydra.utils import instantiate
from omegaconf import II

from trainer.datasetss.base_dataset import BaseDataset, BaseDatasetConfig
from trainer.datasetss.clip_hf_dataset import ProcessorConfig

logger = get_logger(__name__)

@dataclass
class CLIPCustomDatasetConfig(BaseDatasetConfig):
    _target_: str = "trainer.datasetss.clip_hf_dataset.CLIPHFDataset"
    dataset_name: str = "yuvalkirstain/pickapic_v1"
    dataset_config_name: str = "null"

    from_disk: bool = False
    train_split_key: str = "train_custom"
    valid_split_key: str = "validation_unique_custom"
    test_split_key: str = "test_unique_custom"
    train_split_name: str = 'train'
    valid_split_name: str = 'validation_unique'
    test_split_name: str ='test_unique'
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

    num_examples_per_prompt_column_name: str = "num_example_per_prompt"

    keep_only_different: bool = False
    keep_only_with_label: bool = False
    keep_only_with_label_in_non_train: bool = True

    processor: ProcessorConfig = ProcessorConfig()

    limit_examples_per_prompt: int = -1

    only_on_best: bool = False
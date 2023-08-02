from hydra.core.config_store import ConfigStore

from trainer.datasetss.clip_hf_dataset import CLIPHFDatasetConfig
from trainer.datasetss.clip_custom_dataset import CLIPCustomDatasetConfig

cs = ConfigStore.instance()
cs.store(group="dataset", name="clip", node=CLIPHFDatasetConfig)
cs.store(group="dataset_custom", name="clip", node=CLIPCustomDatasetConfig)

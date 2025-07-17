from pydantic import BaseModel, FilePath
import yaml

class DatasetConfig(BaseModel):
    input_dir: str
    menu_json: str
    colormap_csv: str
    gt_dir: str
    results_dir: str

class ModelConfig(BaseModel):
    pe: dict
    segmentation: dict

class SimilarityConfig(BaseModel):
    type: str = "cosine"
    threshold: float = 0.6
    num_patches: int = 5
    patch_size: int = 224
    knn_k: int = 5


class EvaluationConfig(BaseModel):
    metrics: list[str]

class APIConfig(BaseModel):
    host: str = "127.0.0.1"
    port: int = 8000

class EmbeddingConfig(BaseModel):
    directory: str
    model: str 
    use_mean: bool

class ExperimentConfig(BaseModel):
    dataset: DatasetConfig
    models: ModelConfig
    similarity: SimilarityConfig
    evaluation: EvaluationConfig
    api: APIConfig
    embeddings: EmbeddingConfig

    @classmethod
    def load(cls, path: FilePath):
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(**data)

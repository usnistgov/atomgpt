from typing import Optional
from pydantic_settings import BaseSettings


class TrainingPropConfig(BaseSettings):
    """Training config defaults and validation."""

    benchmark_file: Optional[str] = None
    # "AI-SinglePropertyPrediction-exfoliation_energy-dft_3d-test-mae"
    id_prop_path: Optional[str] = None
    prefix: str = "xyz"
    model_name: str = "gpt2"
    leaderboard_dir: str = (
        "/wrk/knc6/AFFBench/jarvis_leaderboard/jarvis_leaderboard"
    )
    batch_size: int = 8
    max_length: int = 512
    num_epochs: int = 500
    latent_dim: int = 1024
    learning_rate: float = 1e-3
    test_each_run: bool = True
    include_struct: bool = False
    pretrained_path: str = ""
    seed_val: int = 42
    n_train: Optional[int] = None
    n_val: Optional[int] = None
    n_test: Optional[int] = None
    train_ratio: Optional[float] = None
    val_ratio: float = 0.1
    test_ratio: float = 0.1
    keep_data_order: bool = False
    output_dir: str = "temp"

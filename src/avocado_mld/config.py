from dataclasses import dataclass


@dataclass(frozen=True)
class TrainingConfig:
    metadata_csv: str = "data/metadata.csv"
    image_root: str = "."
    output_dir: str = "outputs"
    image_size: int = 224
    batch_size: int = 16
    num_workers: int = 0
    epochs: int = 5
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    latent_margin: float = 0.05
    random_state: int = 42
    lambda_ord: float = 1.0
    lambda_reg: float = 1.0
    lambda_temp: float = 0.5
    lambda_view: float = 0.2
    device: str = "auto"

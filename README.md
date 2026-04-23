# Avocado MLD Project

This course project studies avocado ripening prediction from image data. The repository implements a monotonic latent degradation (MLD) pipeline that learns both ripening stage and remaining shelf life from temporal avocado observations collected under different storage conditions.

## Project Scope

- Train image-based models for ripening stage classification and remaining-days regression.
- Encourage monotonic latent behavior over time and consistency across multi-view images.
- Evaluate model outputs with classification, ranking, and regression metrics.
- Save experiment artifacts such as metrics, prediction tables, training histories, and figures.

## Repository Structure

- `src/avocado_mld/`: core training pipeline, dataset utilities, models, losses, metrics, and analysis code.
- `tests/`: unit tests for metadata processing, losses, metrics, model behavior, and training outputs.
- `data/`: metadata files and dataset-related assets.
- `outputs/`: saved experiment runs, baseline results, and generated figures.
- `notebooks/`: notebook-based exploration and research work.

## Dataset Note

The training scripts expect a metadata CSV in `data/` that follows `data/metadata_template.csv`. Image paths can be absolute or resolved relative to the configured image root. The compressed dataset archive in this repository is tracked with Git LFS.

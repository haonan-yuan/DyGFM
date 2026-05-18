"""DyGFM project paths (all resolved from this repository root)."""
import os

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

STATIC_DIR = os.path.join(PROJECT_ROOT, "static")
DYNAMIC_DIR = os.path.join(PROJECT_ROOT, "dynamic")
FINE_TUNE_DIR = os.path.join(PROJECT_ROOT, "fine_tune")

STATIC_DATA_DIR = os.path.join(STATIC_DIR, "data")
STATIC_SAVE_MODEL_DIR = os.path.join(STATIC_DIR, "save_model")

DYNAMIC_DATA_DIR = os.path.join(DYNAMIC_DIR, "data")
DYNAMIC_SAVE_MODEL_DIR = os.path.join(DYNAMIC_DIR, "save_model")
DYNAMIC_CHECKPOINTS_DIR = os.path.join(DYNAMIC_DIR, "checkpoints")
DYNAMIC_EDGE_FEATURE_DIR = os.path.join(DYNAMIC_DATA_DIR, "edge_feature")
DYNAMIC_NORMAL_TIME_DIR = os.path.join(DYNAMIC_DATA_DIR, "normal_time_pt")
DYNAMIC_FEATURES_DIR = os.path.join(DYNAMIC_DATA_DIR, "features")

PROCESSED_DIR = os.path.join(PROJECT_ROOT, "processed")
DOWNSTREAM_DATA_DIR = os.path.join(PROJECT_ROOT, "downstream_data")
GRAPH_STRUCTURE_DIR = os.path.join(PROJECT_ROOT, "graph_structure")

FINE_TUNE_SAVE_MODEL_DIR = os.path.join(FINE_TUNE_DIR, "save_model")
FINE_TUNE_CHECKPOINTS_DIR = os.path.join(FINE_TUNE_DIR, "checkpoints")
SENTENC_BRANCH_DIR = os.path.join(FINE_TUNE_DIR, "sentenc_branch")
TIME_BRANCH_DIR = os.path.join(FINE_TUNE_DIR, "time_branch")

DATASETS = ("genre", "mooc", "reddit", "wikipedia")


def rel(path: str) -> str:
    """Path relative to PROJECT_ROOT (for display in logs)."""
    return os.path.relpath(path, PROJECT_ROOT)


def latest_checkpoint(save_dir: str, prefix: str) -> str:
    if not os.path.isdir(save_dir):
        raise FileNotFoundError(f"Checkpoint directory not found: {save_dir}")
    candidates = [
        os.path.join(save_dir, name)
        for name in os.listdir(save_dir)
        if name.startswith(f"{prefix}_") and name.endswith(".pt")
    ]
    if not candidates:
        raise FileNotFoundError(
            f"No checkpoint matching '{prefix}_*.pt' in {save_dir}"
        )
    return max(candidates, key=os.path.getmtime)

# results/__init__.py
from pathlib import Path

CODE_DIR = Path(__file__).parent.parent

ENTITY = "ljgoodall2001-rhodes-university"
PROJECT_BASE = "WLASL"
LABEL_SUFFIX = "instances_fixed_frange_bboxes_len.json"
NUM_INSTANCES_SUFFIX = "num_instances.json"
WORST_INSTANCES_SUFFIX = "f1-score_MViTv2_B_32x3_asl2000_004.json"
CLASSES_PATH = CODE_DIR / "info/wlasl_class_list.json"
WLASL_ROOT = CODE_DIR / "../data/WLASL"
LABELS_PATH = WLASL_ROOT / "preprocessed/labels"
RAW_DIR = "WLASL2000"
SPLIT_DIR = "splits"
RUNS_PATH = CODE_DIR / "runs"
CONFIGS_PATH = CODE_DIR / "configfiles"
ZFILL = 3
CONFIG_FILETYPE = ".toml"
SEED = 42
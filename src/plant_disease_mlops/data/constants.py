"""Pinned Hugging Face Hub revisions for PlantVillage ingestion.

The live ``mohanty/PlantVillage`` dataset card still documents image splits, but the
default Hub config currently exposes a text-only schema. We load binary archives and
split lists from known-good revisions instead.
"""

HF_REPO_ID = "mohanty/PlantVillage"

# Commit with ``data.zip`` (~2.1GB) containing ``raw/color/<class>/...`` layout.
REVISION_DATA_ZIP = "cb8c40a4b0f758c00ec3bd3e689c776790a929c5"
ZIP_FILENAME = "data.zip"

# Commit with ``splits/color_{train,test}.txt`` paths relative to the zip root.
REVISION_SPLITS = "ded437f547bfabb00b388b99043a237e3a21ce54"

SPLIT_TRAIN = "splits/color_train.txt"
SPLIT_TEST = "splits/color_test.txt"

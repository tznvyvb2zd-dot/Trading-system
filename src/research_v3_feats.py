"""Model lab v3, job A: does the new feature set (raw or ranked) help the current model types?"""
from research_v3_common import run
run({"base_lgbm": ("lgbm", "base"), "base_lr": ("lr", "base"),
     "new_lgbm": ("lgbm", "new"), "new_lr": ("lr", "new"),
     "ranked_lgbm": ("lgbm", "ranked"), "ranked_lr": ("lr", "ranked")}, "feats")

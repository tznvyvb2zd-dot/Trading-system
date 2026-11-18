"""Model lab v3, job B: better model types on the full ranked feature set."""
from research_v3_common import run
run({"reg_lgbm": ("lgbm_reg", "ranked"), "rank_lgbm": ("lgbm_rank", "ranked"),
     "bag_lgbm": ("lgbm_bag", "ranked"), "mlp": ("mlp", "ranked")}, "models", save_shared=False)

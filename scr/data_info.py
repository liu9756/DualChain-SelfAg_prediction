from data_check import check_featurized_dir

stats = check_featurized_dir(
    "/users/PAS2177/liu9756/DualChain-SelfAg_prediction/data/featurized",
    rare_thresholds=(1, 5, 10, 20, 30),
    topk_for_coverage=(5, 10, 20, 50, 100),
    return_stats=True,
)

stats["labels"]["rare_class_stats"][20]

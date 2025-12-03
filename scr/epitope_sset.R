library(dplyr)
library(tidyr)
library(stringr)

pred <- read.csv("/users/PAS2177/liu9756/DualChain-SelfAg_prediction/results/finetune_results_min5/TCR_HLA_pairs_predictions_min5.csv",
                 stringsAsFactors = FALSE)


colnames(pred)

epitope_cols <- grep("^top[0-9]+_epitope$", colnames(pred), value = TRUE)

epitope_cols

sample_epitope_set_top1 <- pred %>%
  group_by(sample) %>%
  summarise(
    n_pairs = n(),  
    n_unique_epitope_top1 = n_distinct(top1_epitope),
    epitope_set_top1 = paste(sort(unique(top1_epitope)), collapse = "; ")
  )

sample_epitope_set_top1

write.csv(sample_epitope_set_top1,
          "/users/PAS2177/liu9756/DualChain-SelfAg_prediction/results/finetune_results/min5_sample_epitope_set_top1.csv",
          row.names = FALSE)
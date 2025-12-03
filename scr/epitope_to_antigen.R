library(dplyr)
library(tidyr)
library(stringr)

sample_top1 <- read.csv(
  "/users/PAS2177/liu9756/DualChain-SelfAg_prediction/results/finetune_results/min5_sample_epitope_set_top1.csv",
  stringsAsFactors = FALSE
)

mapping_list <- read.csv(
  "/users/PAS2177/liu9756/DualChain-SelfAg_prediction/data/my_data/epitope_table_export_1763761141.csv",
  stringsAsFactors = FALSE
)

mapping_clean <- mapping_list %>%
  rename(
    IEDB_ID   = Epitopes...IEDB.ID,
    Epitope   = Epitopes...Epitope,
    Antigen   = Epitopes...Antigen,
    Organism  = Epitopes...Organism,
    References = Epitopes.....References,
    Assays     = Epitopes.....Assays
  ) %>%
  mutate(Epitope = as.character(Epitope))

sample_long <- sample_top1 %>%
  select(sample, n_pairs, n_unique_epitope_top1, epitope_set_top1) %>%
  mutate(epitope_set_top1 = ifelse(is.na(epitope_set_top1), "", epitope_set_top1)) %>%
  separate_rows(epitope_set_top1, sep = ";\\s*") %>%
  rename(Epitope = epitope_set_top1) %>%
  filter(Epitope != "")


sample_epitope_annot <- sample_long %>%
  left_join(mapping_clean, by = "Epitope")


sample_antigen_unique <- sample_epitope_annot %>%
  filter(!is.na(Antigen)) %>%      
  distinct(sample, Antigen) %>%    
  arrange(sample, Antigen)

library(dplyr)
library(tidyr)
library(stringr)


out_dir_stats <- "/users/PAS2177/liu9756/DualChain-SelfAg_prediction/results_prediction_min5/antigen_stats"
dir.create(out_dir_stats, recursive = TRUE, showWarnings = FALSE)


antigen_stats <- sample_antigen_unique %>%
  group_by(Antigen) %>%
  summarise(
    n_samples = n_distinct(sample),                   
    sample_list = paste(sort(unique(sample)), collapse = ";"),
    .groups = "drop"
  ) %>%
  arrange(desc(n_samples), Antigen)


head(antigen_stats, 20)
write.csv(
  antigen_stats,
  file.path(out_dir_stats, "min5_antigen_across_samples_top1.csv"),
  row.names = FALSE
)


sample_antigen_with_ns <- sample_antigen_unique %>%
  left_join(antigen_stats, by = "Antigen")

sample_unique_antigens <- sample_antigen_with_ns %>%
  filter(n_samples == 1) %>%     
  group_by(sample) %>%
  summarise(
    n_unique_antigen = n(),                                
    unique_antigen_list = paste(sort(unique(Antigen)), 
                                collapse = ";"),
    .groups = "drop"
  )


head(sample_unique_antigens)
write.csv(
  sample_unique_antigens,
  file.path(out_dir_stats, "sample_unique_antigens_top1.csv"),
  row.names = FALSE
)

sample_shared_summary <- sample_antigen_with_ns %>%
  filter(n_samples > 1) %>%       
  group_by(sample) %>%
  summarise(
    n_shared_antigen = n(),                
    mean_over_samples = mean(n_samples),   
    max_over_samples  = max(n_samples),    
    .groups = "drop"
  )


head(sample_shared_summary)
library(tidyr)  
sample_antigen_total <- sample_antigen_unique %>%
  group_by(sample) %>%
  summarise(
    n_total_antigen = n(),  
    .groups = "drop"
  )

sample_antigen_summary <- sample_antigen_total %>%
  left_join(sample_unique_antigens,  by = "sample") %>%
  left_join(sample_shared_summary,   by = "sample") %>%
  mutate(
    n_unique_antigen = replace_na(n_unique_antigen, 0L),
    n_shared_antigen = replace_na(n_shared_antigen, 0L)
  )


sample_antigen_summary

write.csv(
  sample_antigen_summary,
  file.path(out_dir_stats, "min5_sample_antigen_summary_top1.csv"),
  row.names = FALSE
)

top20_antigen <- antigen_stats %>%
  arrange(desc(n_samples)) %>%
  head(20)

top20_antigen

write.csv(
  top20_antigen,
  file.path(out_dir_stats, "min5_top20_antigen_by_samples_top1.csv"),
  row.names = FALSE
)

out_dir <- "/users/PAS2177/liu9756/DualChain-SelfAg_prediction/results_prediction_min5/min5_antigen_mapping_by_sample"
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

samples <- unique(sample_antigen_unique$sample)

for (s in samples) {
  df_s <- sample_antigen_unique %>%
    filter(sample == s)
  
  out_csv <- file.path(out_dir, paste0(s, "_antigen_top1.csv"))
  write.csv(df_s, out_csv, row.names = FALSE)
  message("Saved: ", out_csv)
}

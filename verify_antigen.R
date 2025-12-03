library(Seurat)
library(dplyr)
library(tidyr)
library(scRepertoire)
library(ggplot2)
library(SingleCellExperiment)
library(HGNChelper)


data <- readRDS("/users/PAS2177/liu9756/APK_Data/Filtered_Optimaized_CD4T_Small_TCR_label.rds")
features <- c("AKR1A1","AADACL4",)
FeaturePlot(data,features = "AADAC",label = TRUE)
library(ggplot2)
library(dplyr)
library(funModeling)

split_data <- function(data, ratio=0.8){
  idx <- sort(sample(nrow(data),nrow(data)*ratio))
  data_train <- data[idx, ]
  data_test <- data[-idx, ]
  list(train=data_train, test=data_test)
}


# jain -------------------------------------------------------------------


jain <- read.table("data/jain.txt", header = FALSE, sep="\t", dec = ".")
jain$V3 <- as.factor(jain$V3)

ggplot(data=jain, aes(x=V1, y=V2, color=V3)) + geom_point()

summary(jain)
describe(jain)
status(jain)
data_integrity(jain)
plot_num(jain)
correlation_table(jain, "V3")

set.seed(123)
jain_split <- split_data(jain)
jain_split_train <- jain_split$train
jain_split_test <- jain_split$test
save(jain_split_train, file = "data/jain_train.Rda")
save(jain_split_test, file = "data/jain_test.Rda")

# spambase ----------------------------------------------------------------

spambase <- read.table("data/spambase.data", header = FALSE, sep = ",", dec = ".")
spambase_names <- read.table("data/spambase_names.txt", header = FALSE, sep = ":",comment.char = "")
colnames(spambase) <- spambase_names[,1]
spambase$TARGET <- as.factor(spambase$TARGET)
head(spambase)


describe(spambase)
status(spambase)
data_integrity(spambase)
plot_num(spambase)
correlation_table(spambase, "TARGET")

set.seed(123)
spambase_split <- split_data(spambase)
spambase_split_train <- spambase_split$train
spambase_split_test <- spambase_split$test
save(spambase_split_train, file = "data/spambase_train.Rda")
save(spambase_split_test, file = "data/spambase_test.Rda")

# FOREX -------------------------------------------------------------------

forex <- read.table("data/FOREX.csv", header = TRUE, sep = ",", dec = ".")
forex$Class <- as.factor(ifelse(forex$Class, 1, 0))
forex <- forex %>% select(-Timestamp)
head(forex)

describe(forex)
status(forex)
data_integrity(forex)
plot_num(forex)
correlation_table(forex, "Class")



# skin segmentation -------------------------------------------------------

skin_seg <- read.table("data/skin-segmentation.csv", header = TRUE, sep = ",", dec = ".")
skin_seg$Class <- as.factor(skin_seg$Class)
head(skin_seg)

describe(skin_seg)
status(skin_seg)
data_integrity(skin_seg)
plot_num(skin_seg)
correlation_table(skin_seg, "Class")

set.seed(123)
skin_seg_split <- split_data(skin_seg)
skin_seg_split_train <- skin_seg_split$train
skin_seg_split_test <- skin_seg_split$test
save(skin_seg_split_train, file = "data/skin_seg_train.Rda")
save(skin_seg_split_test, file = "data/skin_seg_test.Rda")

# occupancy ---------------------------------------------------------------
occupancy1 <- read.table("data/datatest.txt", header = TRUE, sep = ",", dec = ".")
occupancy2 <- read.table("data/datatest2.txt", header = TRUE, sep = ",", dec = ".")
occupancy3 <- read.table("data/datatraining.txt", header = TRUE, sep = ",", dec = ".")
occupancy <- rbind(occupancy1, occupancy2, occupancy3)
occupancy <- occupancy %>% select(-date)
occupancy$Occupancy <- as.factor(occupancy$Occupancy)
head(occupancy)


describe(occupancy)
status(occupancy)
data_integrity(occupancy)
plot_num(occupancy)
correlation_table(occupancy, "Occupancy")
cor(occupancy[,-6])

set.seed(123)
occupancy_split <- split_data(occupancy)
occupancy_split_train <- occupancy_split$train
occupancy_split_test <- occupancy_split$test
save(occupancy_split_train, file = "data/occupancy_train.Rda")
save(occupancy_split_test, file = "data/occupancy_test.Rda")

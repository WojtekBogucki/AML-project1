# preparing datasets and saving training and test sets
library(ggplot2)
library(dplyr)
library(funModeling)
library(corrplot)

split_data <- function(data, ratio=0.8){
  idx <- sort(sample(nrow(data),nrow(data)*ratio))
  data_train <- data[idx, ]
  data_test <- data[-idx, ]
  list(train=data_train, test=data_test)
}


# jain -------------------------------------------------------------------


jain <- read.table("data/jain.txt", header = FALSE, sep="\t", dec = ".")
jain$V3 <- as.factor(jain$V3)
levels(jain$V3) <- c(0,1)
jain$V3 <- as.numeric(as.vector(jain$V3))
n <- ncol(jain)
jain[, -n] <- scale(jain[, -n])

ggplot(data=jain, aes(x=V1, y=V2, color=V3)) + geom_point()

summary(jain)
describe(jain)
status(jain)
data_integrity(jain)
plot_num(jain)
correlation_table(jain, "V3")
corrplot(cor(jain), method = "number")

set.seed(123)
jain_split <- split_data(jain)
jain_train <- jain_split$train
jain_test <- jain_split$test
save(jain_train, file = "data/jain_train.Rda")
save(jain_test, file = "data/jain_test.Rda")

# spambase ----------------------------------------------------------------

spambase <- read.table("data/spambase.data", header = FALSE, sep = ",", dec = ".")
spambase_names <- read.table("data/spambase_names.txt", header = FALSE, sep = ":",comment.char = "")
colnames(spambase) <- spambase_names[,1]
n <- ncol(spambase)
spambase[, -n] <- scale(spambase[, -n])
head(spambase)


describe(spambase)
status(spambase)
data_integrity(spambase)
plot_num(spambase)
correlation_table(spambase, "TARGET")
corrplot(cor(spambase), method = "color")

set.seed(123)
spambase_split <- split_data(spambase)
spambase_train <- spambase_split$train
spambase_test <- spambase_split$test
save(spambase_train, file = "data/spambase_train.Rda")
save(spambase_test, file = "data/spambase_test.Rda")

# mammography -------------------------------------------------------------------

mammography <- read.table("data/mammography.csv", header = TRUE, sep = ",", dec = ".")
mammography$class <- ifelse(mammography$class==1, 1, 0)
n <- ncol(mammography)
mammography[, -n] <- scale(mammography[, -n])
head(mammography)

describe(mammography)
status(mammography)
data_integrity(mammography)
plot_num(mammography)
correlation_table(mammography, "class")
corrplot(cor(mammography), method = "number")

set.seed(123)
mammography_split <- split_data(mammography)
mammography_train <- mammography_split$train
mammography_test <- mammography_split$test
save(mammography_train, file = "data/mammography_train.Rda")
save(mammography_test, file = "data/mammography_test.Rda")


# skin segmentation -------------------------------------------------------

skin_seg <- read.table("data/skin-segmentation.csv", header = TRUE, sep = ",", dec = ".")
skin_seg$Class <- as.factor(skin_seg$Class)
levels(skin_seg$Class) <- c(0,1)
skin_seg$Class <- as.numeric(as.vector(skin_seg$Class))
n <- ncol(skin_seg)
skin_seg[, -n] <- scale(skin_seg[, -n])
head(skin_seg)

describe(skin_seg)
status(skin_seg)
data_integrity(skin_seg)
plot_num(skin_seg)
correlation_table(skin_seg, "Class")
corrplot(cor(skin_seg), method = "number")

set.seed(123)
skin_seg_split <- split_data(skin_seg)
skin_seg_train <- skin_seg_split$train
skin_seg_test <- skin_seg_split$test
save(skin_seg_train, file = "data/skin_seg_train.Rda")
save(skin_seg_test, file = "data/skin_seg_test.Rda")

# occupancy ---------------------------------------------------------------
occupancy1 <- read.table("data/datatest.txt", header = TRUE, sep = ",", dec = ".")
occupancy2 <- read.table("data/datatest2.txt", header = TRUE, sep = ",", dec = ".")
occupancy3 <- read.table("data/datatraining.txt", header = TRUE, sep = ",", dec = ".")
occupancy <- rbind(occupancy1, occupancy2, occupancy3)
occupancy <- occupancy %>% select(-date)
n <- ncol(occupancy)
occupancy[, -n] <- scale(occupancy[, -n])
head(occupancy)


describe(occupancy)
status(occupancy)
data_integrity(occupancy)
plot_num(occupancy)
correlation_table(occupancy, "Occupancy")
corrplot(cor(occupancy), method = "number")

set.seed(123)
occupancy_split <- split_data(occupancy)
occupancy_train <- occupancy_split$train
occupancy_test <- occupancy_split$test
save(occupancy_train, file = "data/occupancy_train.Rda")
save(occupancy_test, file = "data/occupancy_test.Rda")

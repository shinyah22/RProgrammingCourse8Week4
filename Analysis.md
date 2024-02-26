---
title: "Predictive Modeling Project"
author: "Shinya Hashimoto"
date: "2024-02-23"
output: 
  html_document:
    keep_md: true
---

### Overview
This project involves constructing a machine learning algorithm to predict the quality of activity from an activity monitor using data collected from accelerometers on wearable devices. The primary goal is to predict the manner (`classe` variable) in which exercises were performed.

## Repository Contents

- This document in both `.Rmd` (source) and `.html` (compiled) formats.
- Additional scripts and datasets used in the analysis.

### Data Loading and Preprocessing


```r
library(caret)
library(ggplot2)
library(rpart.plot)
library(corrplot)
set.seed(12345)

# Data download URLs
trainingUrl <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
testingUrl <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"

# Reading data
training <- read.csv(url(trainingUrl), na.strings=c("NA","#DIV/0!", ""))
testing <- read.csv(url(testingUrl), na.strings=c("NA","#DIV/0!", ""))
```

### Data Cleaning and Preprocessing


```r
# Remove columns with NA values
training <- training[, colSums(is.na(training)) == 0]
testing <- testing[, colSums(is.na(testing)) == 0]

# Remove the first 7 columns
training <- training[, -c(1:7)]
testing <- testing[, -c(1:7)]
```

### Model Building and Evaluation


```r
subSamples <- createDataPartition(y=training$classe, p=0.75, list=FALSE)
subTraining <- training[subSamples,]
subTesting <- training[-subSamples,]

# Setting training control
control <- trainControl(method="cv", number=3, verboseIter=FALSE)

# Building and evaluating models
models <- c("rpart", "rf", "gbm", "svmLinear")
names(models) <- c("Decision Tree", "Random Forest", "Gradient Boosted Trees", "SVM")

results <- lapply(models, function(method) {
  # Build model
  model <- train(classe ~ ., data=subTraining, method=method, trControl=control, tuneLength=1)
  # Predict on test data
  prediction <- predict(model, subTesting)
  # Evaluate predictions
  cm <- confusionMatrix(prediction, factor(subTesting$classe))
  # Return confusion matrix and model
  return(list(cm = cm, model = model))
})
```

```
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.1238
##      2        1.5236             nan     0.1000    0.0867
##      3        1.4654             nan     0.1000    0.0647
##      4        1.4219             nan     0.1000    0.0554
##      5        1.3863             nan     0.1000    0.0433
##      6        1.3575             nan     0.1000    0.0451
##      7        1.3284             nan     0.1000    0.0351
##      8        1.3044             nan     0.1000    0.0379
##      9        1.2808             nan     0.1000    0.0299
##     10        1.2612             nan     0.1000    0.0297
##     20        1.1062             nan     0.1000    0.0172
##     40        0.9336             nan     0.1000    0.0103
##     50        0.8753             nan     0.1000    0.0067
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.1245
##      2        1.5238             nan     0.1000    0.0846
##      3        1.4668             nan     0.1000    0.0666
##      4        1.4224             nan     0.1000    0.0534
##      5        1.3872             nan     0.1000    0.0499
##      6        1.3549             nan     0.1000    0.0420
##      7        1.3268             nan     0.1000    0.0356
##      8        1.3031             nan     0.1000    0.0381
##      9        1.2796             nan     0.1000    0.0341
##     10        1.2568             nan     0.1000    0.0297
##     20        1.1049             nan     0.1000    0.0186
##     40        0.9278             nan     0.1000    0.0121
##     50        0.8692             nan     0.1000    0.0058
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.1319
##      2        1.5226             nan     0.1000    0.0883
##      3        1.4640             nan     0.1000    0.0681
##      4        1.4191             nan     0.1000    0.0557
##      5        1.3815             nan     0.1000    0.0431
##      6        1.3529             nan     0.1000    0.0433
##      7        1.3239             nan     0.1000    0.0430
##      8        1.2968             nan     0.1000    0.0363
##      9        1.2737             nan     0.1000    0.0308
##     10        1.2539             nan     0.1000    0.0303
##     20        1.0977             nan     0.1000    0.0160
##     40        0.9278             nan     0.1000    0.0100
##     50        0.8680             nan     0.1000    0.0060
## 
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094             nan     0.1000    0.1301
##      2        1.5239             nan     0.1000    0.0885
##      3        1.4669             nan     0.1000    0.0684
##      4        1.4222             nan     0.1000    0.0525
##      5        1.3864             nan     0.1000    0.0499
##      6        1.3535             nan     0.1000    0.0393
##      7        1.3279             nan     0.1000    0.0391
##      8        1.3020             nan     0.1000    0.0323
##      9        1.2816             nan     0.1000    0.0358
##     10        1.2589             nan     0.1000    0.0309
##     20        1.1059             nan     0.1000    0.0184
##     40        0.9337             nan     0.1000    0.0084
##     50        0.8760             nan     0.1000    0.0081
```

```r
# Display results (accuracy only)
results_df <- sapply(results, function(x) x$cm$overall['Accuracy'])
data.frame(Model=names(models), Accuracy=unlist(results_df), row.names=NULL)
```

```
##                    Model  Accuracy
## 1          Decision Tree 0.2844617
## 2          Random Forest 0.9955139
## 3 Gradient Boosted Trees 0.7512235
## 4                    SVM 0.7873165
```

## Predictions on Test Set
Conduct test data estimation with the best performing model.


```r
# Predictions on test set
# Using Random Forest model for predictions
# Retrieving model based on 'rf' key
best_model <- results[["Random Forest"]]$model

# Execute prediction on test dataset
pred <- predict(best_model, subTesting)

# Display prediction results
print(pred)
```

```
##    [1] A A A A A A A A A A A A A A A A A A A A A A A A A A A A A A A A A A A A A
##   [38] A A A A A A A A A A A A A A A A A A A A A A A A A A A A A A A A A A A A A
##   [75] A A A A A A A A A A A A A A A A A A A A A A A A A A A A A A A A A A A A A
##  [112] A A A A A A A A A A A A A A A A A A A A A A A A A A A A A A A A A A A A A
##  [149] A A A A A A A A A A A A A A A A A A A A A A A A A A A A A A A A A A A A A
##  [186] A A A A A A A A A A A A A A A A A A A A A A A A A A A A A A A A A A A A A
##  [223] A A A A A A A A A A A A A A A A A A A A A A A A A A A A A A A A A A A A A
##  [260] A A A A A A A A A A A A A A A A A A A A A A A A A A A A A A A A A A A A A
##  [297] A A A A A A A A A A A A A A A A A A A A A A A A A A A A A A A A A A A A A
##  [334] A A A A A A A A A A A A A A A A A A A A A A A A A A A A A A A A A A A A A
##  [371] A A A A A A A A A A A A A A A A A A A A A A A A A A A A A A A A A A A A A
##  [408] A A A A A A A A A A A A A A A A A A A A A A A A A A A A A A A A A A A A A
##  [445] A A A A A A A A A A A A A A A A A A A A A A A A A A A A A A A A A A A A A
##  [482] A A A A A A A A A A A A A A A A A A A A A A A A A A A A A A A A A A A A A
##  [519] A A A A A A A A A A A A A A A A A A A A A A A A A A A A A A A A A A A A A
##  [556] A A A A A A A A A A A A A A A A A A A A A A A A A A A A A A A A A A A A A
##  [593] A A A A A A A A A A A A A A A A A A A A A A A A A A A A A A A A A A A A A
##  [630] A A A A A A A A A A A A A A A A A A A A A A A A A A A A A A A A A A A A A
##  [667] A A A A A A A A A A A A A A A A A A A A A A A A A A A A A A A A A A A A A
##  [704] A A A A A A A A A A A A A A A A A A A A A A A A A A A A A A A A A A A A A
##  [741] A A A A A A A A A A A A A A A A A A A A A A A A A A A A A A A A A A A A A
##  [778] A A A A A A A A A A A A A A A A A A A A A A A A A A A A A A A A A A A A A
##  [815] A A A A A A A A A A A A A A A A A A A A A A A A A A A A A A A A A A A A A
##  [852] A A A A A A A A A A A A A A A A A A A A A A A A A A A A A A A A A A A A A
##  [889] A A A A A A A A A A A A A A A A A A A A A A A A A A A A A A A A A A A A A
##  [926] A A A A A A A A A A A A A A A A A A A A A A A A A A A A A A A A A A A A A
##  [963] A A A A A A A A A A A A A A A A A A A A A A A A A A A A A A A A A A A A A
## [1000] A A A A A A A A A A A A A A A A A A A A A A A A A A A A A A A A A A A A A
## [1037] A A A A A A A A A A A A A A A A A A A A A A A A A A A A A A A A A A A A A
## [1074] A A A A A A A A A A A A A A A A A A A A A A A A A A A A A A A A A A A A A
## [1111] A A A A A A A A A A A A A A A A A A A A A A A A A A A A A A A A A A A A A
## [1148] A A A A A A A A A A A A A A A A A A A A A A A A A A A A A A A A A A A A A
## [1185] A A A A A A A A A A A A A A A A A A A A A A A A A A A A A A A A A A A A A
## [1222] A A A A A A A A A A A A A A A A A A A A A A A A A A A A A A A A A A A A A
## [1259] A A A A A A A A A A A A A A A A A A A A A A A A A A A A A A A A A A A A A
## [1296] A A A A A A A A A A A A A A A A A A A A A A A A A A A A A A A A A A A A A
## [1333] A A A A A A A A A A A A A A A A A A A A A A A A A A A A A A A A A A A A A
## [1370] A A A A A A A A A A A A A A A A A A A A A A A A A A B B B B B B B B B B B
## [1407] B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B
## [1444] B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B
## [1481] B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B
## [1518] B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B
## [1555] B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B
## [1592] B B B B B B B B B B B B B B B B B B B B B B A B B B B B B B B B B B B B B
## [1629] B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B
## [1666] B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B
## [1703] B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B
## [1740] B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B
## [1777] B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B
## [1814] B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B
## [1851] B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B
## [1888] B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B
## [1925] B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B
## [1962] B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B
## [1999] B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B
## [2036] B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B
## [2073] B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B
## [2110] B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B
## [2147] B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B
## [2184] B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B
## [2221] B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B
## [2258] B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B
## [2295] B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B
## [2332] B B B B B B B B B B B B B C C C C C B C C C C C C C C C C C C C C C C C C
## [2369] C C C C C C C C C C C C C C C C C C C C C C C C C C C C C C C C C C C C C
## [2406] C C C C C C C C C C C C C C C C C C C C C C C C C C C C C C C C C C C C C
## [2443] C C C C C C C C C C C C C C C C C C C C C C C C C C C C C C C C C C C C C
## [2480] C C C C C C C C C C C C C C C C C C C C C C C C C C C C C C C C C C C C C
## [2517] C C C C C C C C C C C C C C C C C C C C C C C C C C C C C C C C C C C C C
## [2554] C C C C C C C C C C C C C C C C C C C C C C C C C C C C C C C C C C C C C
## [2591] C C C C C C C C C C C C C C C C C C C C C C C C C C C C C C C C C C C C C
## [2628] C C C C C C C C C C C C C C C C C C C C C C C C C C C C C C C C C C C C C
## [2665] C C C C C C C C C C C C C C C C C C C C C C C C C C C C C C C C C C C C C
## [2702] C C C C C C C C C C C C C C C C C C C C C C C C C C C C C C C C C C C C C
## [2739] C C C C C C C C C C C C C C C C C C C C C C C C C C C B C C C C C C C B C
## [2776] C C C C C C C C C C C C C C C C C C C C C C C C C C C C C C C C C C C C C
## [2813] C C C C C C C C C C C C C C C C C C C C C C C C C C C C C C C C C C C C C
## [2850] C C C C C C C C C C C C C C C C C C C C C C C C C C C C C C C C C C C C C
## [2887] C C C C C C C C C C C C C C C C C C C C C C C C C C C C C C C C C C C C C
## [2924] C C C C C C C C C C C C C D C C C C C C C C C C C C C C C C C C C C C C C
## [2961] C C C C C C C C C C C C C B C C C C C C C C C C C C C C C C C C C C C C C
## [2998] C C C C C C C C C C C C C C C C C C C C C C C C C C C C C C C C C C C C C
## [3035] C C C C C C C C C C C C C C C C C C C C C C C C C C C C C C C C C C C C C
## [3072] C C C C C C C C C C C C C C C C C C C C C C C C C C C C C C C C C C C C C
## [3109] C C C C C C C C C C C C C C C C C C C C C C C C C C C C C C C C C C C C C
## [3146] C C C C C C C C C C C C C C C C C C C C C C C C C C C C C C C C C C C C C
## [3183] C C C C C C C C C C C C C C C C C D D D D D D D D D D D D D D D D D D D D
## [3220] D D D D D D D D D D D D D D D D D D D D D D D D D C D D D D D D D D D D D
## [3257] D D D D D D D D D D D D D D D D D D D D D D D D D D D D D D D D D D D D D
## [3294] D D D D D D D D D D D D D D D D D D D D D D D D D D D D D D D D D D D D D
## [3331] D D D D D D D D D D D D D D D D D D D D D D D D D D D D D D D D D D D D D
## [3368] D D D D D D D D D D D D D D D D D D D D D D D D D D D D D D D D D D D D D
## [3405] D D D D D D D D D D D D D D D D D D C D D D D D D D D D D D D D D D D D D
## [3442] D D D D D D D D D D D D D D D D D D D D D D D D D D D D D D D D D D D D D
## [3479] D D D D D D D D D D D D D D D D D D D D D D D D D D D D D D D D D D D D D
## [3516] D D D D D D D D D D D D D D D D D D D D E C C D D C C C D D D D D D D D D
## [3553] D D D D D D D D D D D D D D D D D D D D D D D D D D D D D D D D D D D D D
## [3590] D D D D D D D D D D D D D D D D D C D D D D C D D D D D D D D D D D D D D
## [3627] D D D D D D D D D D D D D D D D D D D D D D D D D D D D D D D D D D D D D
## [3664] D D D D D D D D D D D D D D D D D D D D D D D D D D D D D D D D D D D D D
## [3701] D D D D E E E D D D D D D D D D D D D D D D D D D D D D D D D D D D D D D
## [3738] D D D D D D D D D D D D D D D D D D D D D D D D D D D D D D D D D D D D D
## [3775] D D D D D D D D D D D D D D D D D D D D D D D D D D D D D D D D D D D D D
## [3812] D D D D D D D D D D D C D C D D D D D D D D D D D D D D D D D D D D D D D
## [3849] D D D D D D D D D D D D D D D D D D D D D D D D D D D D D D D D D D D D D
## [3886] D D D D D D D D D D D D D D D D D D D D D D D D D D D D D D D D D D D D D
## [3923] D D D D D D D D D D D D D D D D D D D D D D D D D D D D D D D D D D D D D
## [3960] D D D D D D D D D D D D D D D D D D D D D D D D D D D D D D D D D D D D D
## [3997] D D D D D D D E E E E E E E E E E E E E E E E E E E E E E E E E E E E E E
## [4034] E E E E E E E E E E E E E E E E E E E E E E E E E E E E E E E E E E E E E
## [4071] E E E E E E E E E E E E E E E E E E E E E E E E E E E E E E E E E E E E E
## [4108] E E E E E E E E E E E E E E E E E E E E E E E E E E E E E E E E E E E E E
## [4145] E E E E E E E E E E E E E E E E E E E E E E E E E E E E E E E E E E E E E
## [4182] E E E E E E E E E E E E E E E E E E E E E E E E E E E E D E E E E E E E E
## [4219] E E E E E E E E E E E E E E E E E E E E E E E E E E E E E E E E E E E E E
## [4256] E E E E E E E E E E E E E E E E E E E E E E E E E E E E E E E E E E E E E
## [4293] E E E E E E E E E E E E E E E E E E E E E E E E E E E E E E E E E E E E E
## [4330] E E E E E E E E E E E E E E E E E E E E E E E E E E E E E E E E E E E E E
## [4367] E E E E E E E E E E E E E E E E E E E E E E E E E E E E E E E E E E E E E
## [4404] E E E E E E E E E E E E E E E E E E E E E E E E E E E E E E E E E E E E E
## [4441] E E E E E E E E E E E E E E E E E E E E E E E E E E E E E E E E E E E E E
## [4478] E E E E E E E E E E E E E E E E E E E E E E E E E E E E E E E E E E E E E
## [4515] E E E E E E E E E E E E E E E E E E E E E E E E E E E E E E E E E E E E E
## [4552] E E E E E E E E E E E E E E E E E E E E E E E E E E E E E E E E E E E E E
## [4589] E E E E E E E E E E E E E E E E E E E E E E E E E E E E E E E E E E E E E
## [4626] E E E E E E E E E E E E E E E E E E E E E E E E E E E E E E E E E E E E E
## [4663] E E E E E E E E E E E E E E E E E E E E E E E E E E E E E E E E E E E E E
## [4700] E E E E E E E E E E E E E E E E E E E E E E E E E E E E E E E E E E E E E
## [4737] E E E E E E E E E E E E E E E E E E E E E E E E E E E E E E E E E E E E E
## [4774] E E E E E E E E E E E E E E E E E E E E E E E E E E E E E E E E E E E E E
## [4811] E E E E E E E E E E E E E E E E E E E E E E E E E E E E E E E E E E E E E
## [4848] E E E E E E E E E E E E E E E E E E E E E E E E E E E E E E E E E E E E E
## [4885] E E E E E E E E E E E E E E E E E E E E
## Levels: A B C D E
```

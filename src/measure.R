measure <- function(true_classes, predicted_classes){
  total_population <- length(true_classes)
  correct_predictions <- true_classes == predicted_classes
  incorrect_predictions <- !correct_predictions
  positives<- true_classes == 1
  negatives<- true_classes == 0
  positive_count<- sum(positives)
  negative_count<- sum(negatives)
  predicted_positive_count<- sum(predicted_classes == 1)
  predicted_negative_count<- sum(predicted_classes == 0)
  y_mean <- positive_count / total_population
  
  true_positive <- sum(correct_predictions & positives)
  true_negative <- sum(correct_predictions & negatives)
  false_positive <- sum(incorrect_predictions & negatives)
  false_negative <- sum(incorrect_predictions & positives)
  
  accuracy <- (true_positive + true_negative) / total_population
  precision <- true_positive / predicted_positive_count
  recall <- true_positive / positive_count
  miss_rate <- false_negative / positive_count
  specificity <- true_negative / negative_count
  prevalence <- positive_count / total_population
  false_omission_rate <- false_negative / predicted_negative_count
  fall_out <- false_positive / negative_count
  positive_likelihood_ratio <- recall /  fall_out
  negative_likelihood_ratio <- miss_rate / specificity
  false_discovery_rate <- false_positive / predicted_positive_count
  negative_predictive_value <- true_negative / predicted_negative_count
  diagnostic_odds_ratio <- positive_likelihood_ratio / negative_likelihood_ratio
  f1_score <- 2 * precision * recall / (precision + recall)
  
  r2_score <- 1 - sum((true_classes - predicted_classes)**2) / 
    sum((true_classes - y_mean)**2)

  
  structure(.Data = list(
    true_positive = true_positive,
    true_negative = true_negative,
    false_positive = false_positive,
    false_negative = false_negative,
    accuracy = accuracy,
    precision = precision,
    recall = recall,
    miss_rate = miss_rate,
    specificity = specificity,
    prevalence = prevalence,
    false_omission_rate <- false_omission_rate,
    fall_out = fall_out,
    positive_likelihood_ratio = positive_likelihood_ratio,
    negative_likelihood_ratio = negative_likelihood_ratio,
    false_discovery_rate = false_discovery_rate,
    negative_predictive_value = negative_predictive_value,
    diagnostic_odds_ratio = diagnostic_odds_ratio,
    f1_score = f1_score,
    r2_score = r2_score
  ),
    class = "measure")
}

print.measure <- function(x){
  cat("Model statistics: ", "\n")
  cat("Accuracy:  ", x$accuracy, "\n")
  cat("Precision: ", x$precision, "\n")
  cat("Recall:    ", x$recall, "\n")
  cat("F1 score:  ", x$f1_score, "\n")
  cat("R2 score:  ", x$r2_score, "\n")
}

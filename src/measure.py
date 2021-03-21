import numpy as np


class Measure:

    def __init__(self, true_classes, predicted_classes):

        total_population = len(true_classes)
        correct_predictions = np.equal(true_classes, predicted_classes)
        incorrect_predictions = np.logical_not(correct_predictions)
        positives = np.equal(true_classes, 1)
        negatives = np.equal(true_classes, 0)
        positive_count = np.sum(positives)
        negative_count = np.sum(negatives)
        predicted_positive_count = np.sum(np.equal(predicted_classes, 1))
        predicted_negative_count = np.sum(np.equal(predicted_classes, 1))

        self.true_positive = np.sum(np.logical_and(correct_predictions, positives))
        self.true_negative = np.sum(np.logical_and(correct_predictions, negatives))
        self.false_positive = np.sum(np.logical_and(incorrect_predictions, negatives))
        self.false_negative = np.sum(np.logical_and(incorrect_predictions, positives))

        self.accuracy = (self.true_positive + self.true_negative) / total_population if total_population else None
        self.precision = self.true_positive / predicted_positive_count if predicted_positive_count else None
        self.recall = self.true_positive / positive_count if positive_count else None
        self.miss_rate = self.false_negative / positive_count if positive_count else None
        self.specificity = self.true_negative / negative_count if negative_count else None
        self.prevalence = positive_count / total_population if total_population else None
        self.false_omission_rate = self.false_negative / predicted_negative_count if predicted_negative_count else None
        self.fall_out = self.false_positive / negative_count if negative_count else None
        self.positive_likelihood_ratio = self.recall /  self.fall_out if self.fall_out else None
        self.negative_likelihood_ratio = self.miss_rate / self.specificity if self.specificity else None
        self.false_discovery_rate = self.false_positive / predicted_positive_count if predicted_positive_count else None
        self.negative_predictive_value = self.true_negative / predicted_negative_count if predicted_negative_count else None
        self.diagnostic_odds_ratio = self.positive_likelihood_ratio / self.negative_likelihood_ratio if self.negative_likelihood_ratio else None
        self.f1_score = 2 * self.precision * self.recall / (self.precision + self.recall) if (self.precision + self.recall) else None

    def __repr__(self):

        results = 'Model quality measures: \n'
        results = results + 'Accuracy:  ' + str(self.accuracy) + '\n'
        results = results + 'Precision: ' + str(self.precision) + '\n'
        results = results + 'Recall:    ' + str(self.recall) + '\n'
        results = results + 'F1 score:  ' + str(self.f1_score) + '\n'

        return results

if __name__ == "__main__":
    a = [1,1,0,0]
    b = [1,1,1,0]
    measure = Measure(a,b)
    print(measure)
    print(measure.diagnostic_odds_ratio)
__author__ = 'Guy'

import data
import classifier
import utils
from search.LearningState import LearnState, get_legal_operators
from search.FirstChoiceLocalSearch import FirstChoiceLocalSearch

class KFoldCrossValidation:
    def __init__(self, classifier, data_set, data_labels):
        self.__k = 10
        self.classifier = classifier
        self.partitioned_data = []
        self.partitioned_label = []
        for i in xrange(self.__k):
            self.partitioned_data.append(data_set[i::self.__k])
            self.partitioned_label.append(data_labels[i::self.__k])



    def crossValidate(self):
        before_search_accuracy = []
        after_search_accuracy = []
        for i in xrange(self.__k):
            test_data = self.partitioned_data.pop(0)
            test_labels = self.partitioned_label.pop(0)
            validation_data = self.partitioned_data.pop(0)
            validation_labels = self.partitioned_label.pop(0)
            training_data = _merge_list(self.partitioned_data)
            training_labels = _merge_list(self.partitioned_label)
            self.classifier.train(training_data, training_labels)
            before_search_accuracy.append(checkAccuracy(self.classifier.classify(test_data), test_labels))
            start_state = LearnState(get_legal_operators(training_data), training_data, training_labels)
            local_search = FirstChoiceLocalSearch(start_state)
            optimal_state, ops = local_search.search(validation_data, validation_labels,  self.classifier, [])
            self.classifier.train(optimal_state.training_set, optimal_state.training_set_labels)
            test_set = test_data
            test_set_labels = test_labels
            for op in ops:
                test_set = op[0](test_set, training = False)
                test_set_labels = op[1](test_set_labels, training = False)
            after_search_accuracy.append(checkAccuracy(self.classifier.classify(test_set), test_set_labels))

            self.partitioned_data.append(test_data)
            self.partitioned_label.append(test_labels)
            self.partitioned_data.insert(0, validation_data)
            self.partitioned_label.insert(0, validation_labels)
            print "done", i + 1
        return before_search_accuracy, after_search_accuracy



def checkAccuracy(classifier_labels, real_labels):
     return float(len([(c , e) for (c , e) in zip(classifier_labels, real_labels) if c == e])) / len(real_labels)


def _merge_list(lists):
    ls = []
    for l in lists:
        ls = ls + l
    return ls


data_lst = data.load_hw3_data_2()
kfcv = KFoldCrossValidation(classifier.KNearestNeighbours(1), data_lst[0], data_lst[1])
before, after = kfcv.crossValidate()
print before, float(sum(before)) / len(before)
print after, float(sum(after)) / len(after)
print utils.student_paired_t_test(before, after)

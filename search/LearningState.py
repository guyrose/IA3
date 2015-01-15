__author__ = 'Guy'

from abstract_search import SearchState
import numpy as np
from copy import deepcopy
from random import shuffle


class LearnState(SearchState):
    def __init__(self, legal_operators, training_set, training_set_labels):
        super(self).__init__(legal_operators)
        self.training_set = training_set
        self.training_set_labels = training_set_labels

    def evaluate(self, evaluation_set, evaluation_set_labels, operator, classifier):
        classifier.train(self.training_set, self.training_set_labels)
        return float(len([(c , e) for (c , e) in zip(self.classifier.classify(evaluation_set), evaluation_set_labels) if c == e])) / len(evaluation_set)

    def get_next_states(self):
        '''Implementation of the transitions for a state.
        This implementation requires your operators to receive a state and return a state (including operators).
        '''
        return [(state, op) for (state, op) in
        zip([operator(self) for operator in self._legal_operators], self._legal_operators)]


def feature_operator_create(i):
    def feature_operator_set(sample_set):
        new_set = deepcopy(sample_set)
        for x in new_set:
            del new_set[i]
        return new_set

    def feature_operator_label(sample_label):
        return deepcopy(sample_label)

    return feature_operator_set, feature_operator_label


def examples_operator_create(to_remove):
    def examples_operator_set(sample_set):
        new_set = deepcopy(sample_set)
        if len(sample_set) < 5:
            return new_set
        for x in to_remove:
            del new_set[x]
        return new_set

    def examples_operator_label(sample_label):
        new_label = deepcopy(sample_label)
        if len(sample_label) < 5:
            return new_label
        for x in to_remove:
            del new_label[x]
        return new_label

    return examples_operator_set, examples_operator_label


def get_legal_operators(training_set):
    #Create a feature operator. As many as the number of features (the size of one of the lists)
    to_remove = range(len(training_set))
    ops = [feature_operator_create(i) for i in range(len(training_set[0]))]

    for i in xrange(len(training_set[0])):
        shuffle(to_remove)
        ops.append(examples_operator_create(to_remove[:5])
    return ops



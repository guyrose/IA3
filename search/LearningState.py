__author__ = 'Guy'

from abstract_search import SearchState
import numpy as np
from copy import deepcopy
from random import shuffle


class LearnState(SearchState):
    def __init__(self, legal_operators, training_set, training_set_labels):
        super(LearnState, self).__init__(legal_operators)
        self.training_set = training_set
        self.training_set_labels = training_set_labels

    def evaluate(self, evaluation_set, evaluation_set_labels, classifier):
        classifier.train(self.training_set, self.training_set_labels)
        return float(len([(c , e) for (c , e) in zip(classifier.classify(evaluation_set), evaluation_set_labels) if c == e])) / len(evaluation_set)

    def get_next_states(self):
        '''Implementation of the transitions for a state.
        This implementation requires your operators to receive a state and return a state (including operators).
        '''
        return [(state, op) for (state, op) in
        zip([LearnState(get_legal_operators(operator[0](self.training_set)), operator[0](self.training_set), operator[1](self.training_set_labels)) for operator in self._legal_operators], self._legal_operators)]


def feature_operator_create(i):
    def feature_operator_set(sample_set, training=True):
        new_set = deepcopy(sample_set)
        for x in new_set:
            del x[i]
        return new_set

    def feature_operator_label(sample_label, training=True):
        return deepcopy(sample_label)

    return feature_operator_set, feature_operator_label


def examples_operator_create(to_remove):
    def examples_operator_set(sample_set, training=True):
        if not training:
            return sample_set
        new_set = deepcopy(sample_set)
        if len(sample_set) < 5:
            return new_set
        to_remove.sort(reverse=True)
        for x in to_remove:
            del new_set[x]
        return new_set


    def examples_operator_label(sample_label, training=True):
        if not training:
            return sample_label
        new_label = deepcopy(sample_label)
        if len(sample_label) < 5:
            return new_label
        to_remove.sort(reverse=True)
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
        ops.append(examples_operator_create(to_remove[:5]))
    return ops



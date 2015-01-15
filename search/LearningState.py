__author__ = 'Guy'

from abstract_search import SearchState
import numpy as np
from random import shuffle


class LearnState(SearchState):
    def __init__(self, classifier,  training_set, training_set_labels,  legal_operators):
        self.classifier = classifier
        self._legal_operators = legal_operators
        self.training_set = training_set
        self.training_set_labels = training_set_labels

    def evaluate(self, evaluation_set, evaluation_set_labels, *args, **kwargs):
        self.classifier.train(self.training_set.tolist(), self.training_set_labels.tolist()[0])
        return float(len([(c , e) for (c , e) in zip(self.classifier.classify(evaluation_set), evaluation_set_labels) if c == e])) / len(evaluation_set)

    def get_next_states(self):
        '''Implementation of the transitions for a state.
        This implementation requires your operators to receive a state and return a state (including operators).
        '''
        return [(state, op) for (state, op) in
        zip([operator(self) for operator in self._legal_operators], self._legal_operators)]


def feature_operator_create(i):
    def feature_operator(state, training = True):
        return LearnState(state.classifer, np.delete(state.training_set, i, 1), state.training_set_labels, get_legal_operators(state.training_set))
    return feature_operator


def examples_operator_create():
    def examples_operator(state, training = True):
        if not training:
            return state
        if state.training_set.shape[0] < 5:
            return state
        to_remove = shuffle(range(state.training_set.shape[0]))[:5]
        return LearnState(state.classifer, np.delete(state.training_set, to_remove, 0), np.delete(state.training_set_labels, to_remove, 0), get_legal_operators(state.training_set))


def get_legal_operators(training_set):
    ops = [feature_operator_create(i) for i in xrange(training_set.shape[1])]
    ops += [examples_operator_create() for i in xrange(training_set.shape[1])]
    return ops



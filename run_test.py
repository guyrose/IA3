__author__ = 'Guy'


import data
import classifier
from search.LearningState import get_legal_operators, LearnState
from search.FirstChoiceLocalSearch import FirstChoiceLocalSearch
import numpy as np

data_ls = data.load_hw3_data_1()
local_search = FirstChoiceLocalSearch(LearnState(classifier.KNearestNeighbours(3), np.matrix(data_ls[0][0]), np.matrix(data_ls[0][1]), get_legal_operators(np.matrix(data_ls[0][0]))))
optimal_state, ops = local_search.search(np.matrix(data_ls[1][0]), np.matrix(data_ls[1][1]))
knn = classifier.KNearestNeighbours(3)
knn.train(optimal_state.training_set, optimal_state.training_set_labels)
test_set = data_ls[2][0]
test_set_labels = data_ls[2][1]
for op in ops:
    test_set = op(test_set, train = False)
    test_set_labels = op(test_set_labels, train = False)

print float(len([(c , e) for (c , e) in zip(knn.classify(test_set.toList()), test_set_labels.toList()[0]) if c == e])) / len(test_set)

__author__ = 'Guy'
from abstract_search import LocalSearch
from random import shuffle


class FirstChoiceLocalSearch(LocalSearch):

    def search(self, evaluation_set, evaluation_set_labels, classifier, **kwargs):
       
        current = self._current_state
        current_evaluation = current.evaluate(evaluation_set, evaluation_set_labels, *args, **kwargs)
        while True:
            next_states = current.get_next_states()
            shuffle(next_states)
            for (next_state, ops) in next_states:
                new_evaluation = op[0](evaluation_set)
                new_evaluation_labels = op[1](evaluation_set_labels)
                next_evaluation = next_state.evaluate(new_evaluation, new_evaluation_labels, classifier)
                if next_evaluation > current_evaluation:
                    current_evaluation = next_evaluation
                    current = next_state
                    self.operators.append(op)
                    break
            return current, self.operators
        return current, self.operators

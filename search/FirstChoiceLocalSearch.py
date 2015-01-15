__author__ = 'Guy'
from abstract_search import LocalSearch
from random import shuffle


class FirstChoiceLocalSearch(LocalSearch):


    def search(self, evaluation_set, evaluation_set_labels, *args, **kwargs):
        current = self._current_state
        current_evaluation = current.evaluate(evaluation_set, evaluation_set_labels, *args, **kwargs)
        while True:
            for (next_state, op) in shuffle(current.get_next_states()):
                next_evaluation = next_state.evaluate(op(evaluation_set, train = False), evaluation_set_labels)
                if  next_evaluation > current_evaluation:
                    current_evaluation = next_evaluation
                    current = next_state
                    self.operators.append(op)
                    break
            return current, self.operators
        return current, self.operators






    

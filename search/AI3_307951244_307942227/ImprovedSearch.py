__author__ = 'Guy'
from random import shuffle

from BasicSearch import FirstChoiceLocalSearch


class ImprovedFirstChoiceLocalSearch(FirstChoiceLocalSearch):
    def search(self, evaluation_set, evaluation_set_labels, classifier, *args, **kwargs):
        current = self._current_state
        current_evaluation = current.evaluate(evaluation_set, evaluation_set_labels, classifier)
        next_evaluation_set = evaluation_set
        next_evaluation_labels = evaluation_set_labels
        ## Find local maximum
        while True:
            next_states = current.get_next_states()
            shuffle(next_states)
            improved = False
            for (next_state, ops) in next_states:
                new_evaluation_set = ops[0](evaluation_set, training=False)
                new_evaluation_labels = ops[1](evaluation_set_labels, training=False)
                next_evaluation_value = next_state.evaluate(new_evaluation_set, new_evaluation_labels, classifier)
                if next_evaluation_value > current_evaluation:
                    current_evaluation = next_evaluation_value
                    best_next_state = next_state
                    next_evaluation_set = new_evaluation_set
                    next_evaluation_labels = new_evaluation_labels
                    next_operator = ops
                    improved = True
            if not improved:
                return current, self.operators
            current = best_next_state
            self.operators.append(next_operator)
            evaluation_set = next_evaluation_set
            evaluation_set_labels = next_evaluation_labels

        return next_state, self.operators

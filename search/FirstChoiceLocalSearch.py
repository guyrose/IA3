__author__ = 'Guy'
from abstract_search import LocalSearch
from random import shuffle


class FirstChoiceLocalSearch(LocalSearch):

    def search(self, evaluation_set, evaluation_set_labels, classifier, *args, **kwargs):
        current = self._current_state
        current_evaluation = current.evaluate(evaluation_set, evaluation_set_labels, classifier)
        print "start", str(current_evaluation)
        while True:
            next_states = current.get_next_states()
            shuffle(next_states)
            improved = False
            for (next_state, ops) in next_states:
                new_evaluation = ops[0](evaluation_set, training=False)
                new_evaluation_labels = ops[1](evaluation_set_labels, training=False)
                next_evaluation = next_state.evaluate(new_evaluation, new_evaluation_labels, classifier)
                if next_evaluation > current_evaluation:
                    current_evaluation = next_evaluation
                    current = next_state
                    evaluation_set = new_evaluation
                    evaluation_set_labels = new_evaluation_labels
                    self.operators.append(ops)
                    print "found", str(current_evaluation)
                    improved = True
                    break
            if not improved:
                return current, self.operators
        return current, self.operators

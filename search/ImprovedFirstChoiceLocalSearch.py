__author__ = 'Guy'
from abstract_search import LocalSearch
from random import shuffle, choice
import FirstChoiceLocalSearch
from LearningState import get_legal_operators, LearnState


class ImprovedFirstChoiceLocalSearch(FirstChoiceLocalSearch):
    def search(self, evaluation_set, evaluation_set_labels, classifier, *args, **kwargs):
        max_state = self._current_state
        max_ops = []
        max_evaluation = self._current_state.evaluate(evaluation_set, evaluation_set_labels, classifier)
        for i in xrange(5):
            current = self._current_state
            next_state = current
            operators = []
            current_evaluation = current.evaluate(evaluation_set, evaluation_set_labels, classifier)
            ## choose operators at random for "random reset"
            for j in xrange(10):
                legal_operators = get_legal_operators(current.training_set)
                op = choice(legal_operators)
                operators.append(op)
                current = LearnState(get_legal_operators(op[0](current.training_set)), op[0](current.training_set), op[1](current.training_set_labels))
            ## Find local maximum
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
                        next_state = next_state
                        evaluation_set = new_evaluation
                        evaluation_set_labels = new_evaluation_labels
                        next_operator = ops
                        improved = True
                if not improved:
                    if current_evaluation > max_evaluation:
                        max_evaluation = current_evaluation
                        max_state = current
                        max_ops = operators
                    break
                current = next_state
                operators.append(ops)
        return max_state, max_ops

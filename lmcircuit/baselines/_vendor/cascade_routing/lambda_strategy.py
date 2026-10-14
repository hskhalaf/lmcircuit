# Modified by lmcircuit (Apache-2.0 section 4b): kept Strategy and ConstantStrategy only (RepetitiveConstantStrategy and HyperoptStrategy are unused); standard-library logging instead of loguru; no hyperopt import. See ../__init__.py for provenance.
import numpy as np
import logging

logger = logging.getLogger(__name__)


class Strategy:
    def __init__(self):
        """
        Initializes an instance of the Strategy class. These strategies optimize the lambda parameters.
        """

        pass

    def compute_lambdas(self, current_optimal_lambdas, execute_function, 
                        cost_target, args_execute_function):
        """
        Compute the lambdas for model selection.

        Parameters:
        - current_optimal_lambdas (list): The current optimal lambdas in the selection algorithm.
        - execute_function (function): The function to execute for each lambda which returns both cost and quality.
        - cost_target (float): The target cost for model selection.
        - args_execute_function (tuple): The arguments to pass to the execute_function.

        Raises:
        - NotImplementedError: This method is not implemented yet.
        """
        raise NotImplementedError
    
class ConstantStrategy(Strategy):
    def __init__(self, max_lambda, n_iterations=20):
        """
        Initialize a ConstantStrategy object.
        Assumes that all values of lambda are the same and performs a binary search to find the optimal lambda.

        Parameters:
        - max_lambda (float): The maximum value for lambda.
        - n_iterations (int, optional): The number of iterations. Default is 20.
        """
        super().__init__()
        self.max_lambda = max_lambda
        self.n_iterations = n_iterations

    def compute_lambdas(self, current_optimal_lambdas, execute_function, cost_target, args_execute_function):
        logger.debug("Computing lambdas with constant strategy")
        lambda_min = (0, None, None)
        lambda_max = (self.max_lambda, None, None)

        max_lambda = self.max_lambda
        while max_lambda < 1e10:
            lambdas = [max_lambda for _ in range(len(current_optimal_lambdas))]
            output_dict = execute_function(lambdas, *args_execute_function)
            cost = output_dict['cost']
            if cost > cost_target:
                logger.info(f'Max lambda {max_lambda} is too low, increasing max lambda')
                lambda_min = (max_lambda, cost, output_dict['quality'])
                max_lambda *= 2
                lambda_max = (max_lambda, None, None)
            else:
                lambda_max = (max_lambda, cost, output_dict['quality'])
                break

        # Do binary search to find the optimal lambda
        for _ in range(self.n_iterations):
            lambda_mid = (lambda_min[0] + lambda_max[0]) / 2
            logger.debug(f"Lambda mid: {lambda_mid}")
            lambdas = [lambda_mid for _ in range(len(current_optimal_lambdas))]
            output_dict = execute_function(lambdas, *args_execute_function)
            cost = output_dict['cost']

            if cost < cost_target:
                lambda_max = (lambda_mid, cost, output_dict['quality'])
            else:
                lambda_min = (lambda_mid, cost, output_dict['quality'])
            logger.debug(f"Cost: {cost}")
            logger.debug(f"Quality: {output_dict['quality']}")

        if lambda_max[2] is None:
            lambdas = [lambda_max[0] for _ in range(len(current_optimal_lambdas))]
            output_dict = execute_function(lambdas, *args_execute_function)
            lambda_max = (lambda_max[0], output_dict['cost'], output_dict['quality'])

        logger.info(f"Final Lambda: {lambda_max[0]}")
        logger.info(f"Final Cost: {lambda_max[1]}")
        logger.info(f"Final Quality: {lambda_max[2]}")
        
        return [lambda_max[0] for _ in range(len(current_optimal_lambdas))], lambda_max[1], lambda_max[2]

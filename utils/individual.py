import numpy as np
import node as Node
from utils import draw
import protectedOperators as po
import random

CONSTANT = [
    np.pi,           # π ≈ 3.14159
    np.e,            # e ≈ 2.71828
    (1 + np.sqrt(5)) / 2,  # Golden ratio φ ≈ 1.61803
    np.sqrt(2),      # √2 ≈ 1.41421
    np.sqrt(3),      # √3 ≈ 1.73205
    np.log(2),       # ln(2) ≈ 0.69314
    np.log(10),      # ln(10) ≈ 2.30259
    1.0,             # Unity
    2.0,             # Simple integer
    0.5,             # Half
    -1.0,            # Negative unity
    0.0              # Zero
]

NODE_TYPE = random.choices(
    ['binary', 'unary', 'variable', 'constant'],
    weights=[0.35, 0.35, 0.15, 0.15]
)[0]

class Individual:
    def __init__(self, maxDepth, xTrain, yTrain, individualAttempts):
        self.root = None
        self.maxDepth = maxDepth
        self.xTrain = xTrain
        self.yTrain = yTrain
        self.individualAttempts = individualAttempts
        self.fitness = None
        self.numVariables = xTrain.shape[0] if xTrain is not None else 0
        self.variables = [f'x{i}' for i in range(self.numVariables)]

    def generate(self):
        """
        Generate a random individual structure for the individual.
        """
        self.root = self.generateRandomIndividual()
        self.computeFitness()

    def generateRandomIndividual(self, currentDepth=0, maxDepth=None):
        """
        Recursively generate a random individual structure.
        """
        if maxDepth is None:
            maxDepth = self.maxDepth

        if currentDepth >= maxDepth or (self.root is not None and random.random() < 0.1):
            # Create a leaf node with a constant or variable
            if random.random() < 0.5 and self.variables:
                return Node.Node(random.choice(self.variables), 'variable')
            else:
                return Node.Node(random.choice(CONSTANT), 'constant')
        # Randomly choose an operator
        else:
            if NODE_TYPE == 'binary':
                op = random.choice(po.OPERATORS_BINARY)
                left = self.generateRandomIndividual(currentDepth + 1, maxDepth)
                right = self.generateRandomIndividual(currentDepth + 1, maxDepth)
                return Node.Node(op, successors=[left, right], name='operatorBinary')

            elif NODE_TYPE == 'unary':
                op = random.choice(po.OPERATORS_UNARY)
                child = self.generateRandomIndividual(currentDepth + 1, maxDepth)
                return Node.Node(op, successors=[child], name='operatorUnary')

            elif NODE_TYPE == 'variable':
                return Node.Node(self.variables[random.randint(0, self.numVariables - 1)], name='variable')

            else:  # constant
                return Node.Node(CONSTANT[random.randint(0, len(CONSTANT) - 1)], name='constant')

    def computeFitness(self, get_pred=False):
        """
        Compute fitness by evaluating the expression represented by self.root.
        Uses self.xTrain as input variables and self.yTrain as target.
        """

        if self.root is None:
            return None

        # Get expression string from the root node (e.g. "add(x0, multiply(x1, 3.14))")
        expression = str(self.root)  # uses __str__ -> long_name

        # Prepare evaluation environment with variables x0, x1, ... mapped to columns in xTrain
        eval_globals = {"np": np, "nan": np.nan, "inf": np.inf}

        # Add all operators by name into eval_globals
        for name, func in po.OPERATORS.items():
            eval_globals[name] = func

        # Add variables (x0, x1, ...) mapped to self.xTrain rows or columns depending on data shape
        # Assuming self.xTrain shape is (num_vars, num_samples), map x0, x1, ... accordingly:
        for i, var in enumerate(self.variables):
            eval_globals[var] = self.xTrain[i]

        try:
            # Create lambda function that takes x (ignored here, but could be used)
            formula = eval(f"lambda x: {expression}", eval_globals)
            y_pred = formula(self.xTrain)

            # Check for NaNs or Infs in prediction
            if np.isnan(y_pred).any() or np.isinf(y_pred).any():
                self.fitness = float('inf')
            else:
                # Mean Squared Error as fitness
                self.fitness = float(np.mean((self.yTrain - y_pred) ** 2))

            if get_pred:
                return y_pred

        except Exception as e:
            # On error, assign bad fitness and optionally warn or handle gracefully
            self.fitness = float('inf')
            # You can also print or log e here if debugging
            if get_pred:
                return None
            
    def clone(self):
        """
        Clone of the individual
        """
        newIndividual = Individual(self.maxDepth, self.xTrain, self.yTrain, self.individualAttempts)
        newIndividual.root = self.root.clone() if self.root else None
        newIndividual.fitness = self.fitness
        return newIndividual
    
    def getAllNodes(self):
        """
        Get all nodes in the individual tree.
        """
        return self.root.subtree if self.root else set()



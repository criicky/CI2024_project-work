import numpy as np
import utils.node as Node
from utils.draw import draw
import utils.protectedOperators as po
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

            nodeType = random.choices(
                ['binary', 'unary', 'variable', 'constant'],
                weights=[0.35, 0.35, 0.15, 0.15]
            )[0]

            if nodeType == 'binary':
                op = random.choice(list(po.OPERATORS_BINARY.values()))
                left = self.generateRandomIndividual(currentDepth + 1, maxDepth)
                right = self.generateRandomIndividual(currentDepth + 1, maxDepth)
                return Node.Node(op, successors=[left, right], name='operatorBinary')

            elif nodeType == 'unary':
                op = random.choice(list(po.OPERATORS_UNARY.values()))
                child = self.generateRandomIndividual(currentDepth + 1, maxDepth)
                return Node.Node(op, successors=[child], name='operatorUnary')

            elif nodeType == 'variable':
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
        Get all nodes in the individual as (node, parent, index, depth) tuples.
        """
        if self.root is None:
            return []

        nodes = []

        def traverse(node, parent=None, index=None, depth=0):
            nodes.append((node, parent, index, depth))
            for idx, child in enumerate(node.successors):
                traverse(child, node, idx, depth + 1)

        traverse(self.root)
        return nodes
    
    def getIndividualHeight(self, node):
        if node is None:
            return 0
        left_height = self.getIndividualHeight(node.successors[0]) if node.successors else 0
        right_height = self.getIndividualHeight(node.successors[1]) if node.successors else 0
        return 1 + max(left_height, right_height)
    
    def mutate(self):
        """
        Mutation for the Individual
        Using different mutations techniques
        """
        if self.root is None:
            return
        
        if random.random() < 0.6:
            self.replaceChild()
        else:
            mutations = [
                self.mutateIndividual,
                self.swapChilds,
                self.promoteChild,
                self.replaceWithConstant
            ]
            random.choice(mutations)()

    def replaceChild(self):
        """
        Select a random node among all the sub nodes and replace it with a new
        randomly generated subtree.
        It goes on until fitness is valid
        """
        nodes = self.getAllNodes()  # Each tuple: (node, parent, is_left, depth)
        attempts = self.individualAttempts

        while attempts > 0:
            node_to_mutate, parent, isLeft, nodeDepth = random.choice(nodes)

            remainingDepth = self.maxDepth - nodeDepth
            newIndividual = self.generateRandomIndividual(
                currentDepth = 0,
                maxDepth = remainingDepth
            )

            # Store the original reference for potential revert
            if parent is None:
                oldIndividual = self.root
                self.root = newIndividual
            elif isLeft:
                oldIndividual = parent.successors[0]
                parent.successors[0] = newIndividual
            else:
                oldIndividual = parent.successors[1]
                parent.successors[1] = newIndividual

            # Check fitness
            self.computeFitness()
            if self.fitness != np.inf:
                return  # Successful mutation, exit early

            # Revert if invalid
            if parent is None:
                self.root = oldIndividual
            elif isLeft:
                parent.successors[0] = oldIndividual
            else:
                parent.successors[1] = oldIndividual

            attempts -= 1

    def mutateIndividual(self):
        """
        Randomly change the operator/function of a node without altering structure.
        """
        nodes = self.getAllNodes()
        if not nodes:
            return
        
        node, _, _, _ = random.choice(nodes)
        
        # Only mutate function nodes (not constants or variables)
        if node.is_leaf:
            return  # skip leaf mutation here (or handle constants separately)
        
        if node.arity == 2:
            new_func = random.choice(po.OPERATORS_BINARY)
        elif node.arity == 1:
            new_func = random.choice(po.OPERATORS_UNARY)
        else:
            return  # No mutation for unusual arity

        # Update function and name
        node._func = lambda *args, **kwargs: new_func(*args)
        node._str = getattr(new_func, "__name__", str(new_func))

        self.computeFitness()

    def swapChilds(self):
        """
        Swaps the children of a randomly chosen binary operator.
        """
        binary_nodes = [
            n for n in self.getAllNodes()
            if n[0].arity == 2  # Only binary operators
        ]
        if not binary_nodes:
            return
        
        node, _, _, _ = random.choice(binary_nodes)
        # Swap first and second successors
        succ = node.successors
        node.successors = [succ[1], succ[0]]
        
        self.computeFitness()

    def promoteChild(self):
        """
        Replace a randomly selected child with one of its sub-childs.
        """
        nodes = self.getAllNodes()
        if not nodes:
            return
        
        attempts = self.individualAttempts
        while attempts > 0:
            node, parent, idx, _ = random.choice(nodes)
            
            # Find internal nodes (non-leaf with at least one child)
            if not node.is_leaf and node.successors:
                # Choose one of the node's children to "hoist"
                chosen_child = random.choice(node.successors)
                
                # Replace the original node with the chosen child
                if parent is None:
                    self.root = chosen_child
                else:
                    parent.successors[idx] = chosen_child
                break
            
            attempts -= 1

        self.computeFitness()

    def replaceWithConstant(self):
        """
        Replaces a randomly selected operator child with a constant equal to its evaluated mean value.
        """
        
        nodes = self.getAllNodes()  # Expected format: (node, parent, index, depth)
        if not nodes:
            return

        attempts = self.individualAttempts
        valid_collapse_found = False

        while attempts > 0 and not valid_collapse_found:
            node, parent, idx, _ = random.choice(nodes)

            # Only collapse operator nodes (arity >= 1)
            if node.arity == 0:
                attempts -= 1
                continue

            try:
                # Use __str__ to get the full expression string
                expression = str(node)
                eval_globals = {"np": np, "nan": np.nan, "inf": np.inf}

                formula = eval(f"lambda x: {expression}", eval_globals)
                collapsed_value = float(np.mean(formula(self.x_train)))
            except Exception:
                collapsed_value = 0.0

            # Create new constant node
            new_node = Node.Node(collapsed_value)

            # Replace node in parent or at root
            if parent is None:
                self.root = new_node
            else:
                successors_copy = parent.successors
                successors_copy[idx] = new_node
                parent.successors = successors_copy

            self.computeFitness()

            if self.fitness != np.inf:
                valid_collapse_found = True
            else:
                # Revert change if invalid
                if parent is None:
                    self.root = node
                else:
                    successors_copy = parent.successors
                    successors_copy[idx] = node
                    parent.successors = successors_copy

            attempts -= 1

    def crossover(self, recombinationChild):
        """
        Perform recombination with another child by swapping randomly chosen sub-childs.
        The swap is accepted only if it does not exceed max depth in either offspring
        and both result in valid fitness values.

        Parameters:
            recombinationChild: Another child instance.

        Returns:
            tuple: Two new offspring trees.
        """
        offspring1 = self.clone()
        offspring2 = recombinationChild.clone()

        nodes1 = offspring1.getAllNodes()  # Expected: (node, parent, index, depth)
        nodes2 = offspring2.getAllNodes()

        attempts = self.individualAttempts
        isSwappable = False

        while attempts > 0 and not isSwappable:
            node1, parent1, idx1, depth1 = random.choice(nodes1)
            node2, parent2, idx2, depth2 = random.choice(nodes2)

            height1 = offspring1.getIndividualHeight(node1)
            height2 = offspring2.getIndividualHeight(node2)

            # Check depth constraints for both offspring after swap
            if (depth1 + height2 <= self.maxDepth) and (depth2 + height1 <= self.maxDepth):
                # Clone the chosen subtrees
                subtree1 = node1.clone()
                subtree2 = node2.clone()

                # Swap in offspring1
                if parent1 is None:
                    offspring1.root = subtree2
                else:
                    succs = parent1.successors
                    succs[idx1] = subtree2
                    parent1.successors = succs

                # Swap in offspring2
                if parent2 is None:
                    offspring2.root = subtree1
                else:
                    succs = parent2.successors
                    succs[idx2] = subtree1
                    parent2.successors = succs

                # Compute fitness for both offspring
                offspring1.computeFitness()
                offspring2.computeFitness()

                if offspring1.fitness != np.inf and offspring2.fitness != np.inf:
                    isSwappable = True
                else:
                    # Revert if invalid
                    if parent1 is None:
                        offspring1.root = node1
                    else:
                        succs = parent1.successors
                        succs[idx1] = node1
                        parent1.successors = succs

                    if parent2 is None:
                        offspring2.root = node2
                    else:
                        succs = parent2.successors
                        succs[idx2] = node2
                        parent2.successors = succs

            attempts -= 1

        if not isSwappable:
            return self.clone(), recombinationChild.clone()

        return offspring1, offspring2
    
    def __str__(self):
        return self.root.__str__() if self.root is not None else ""
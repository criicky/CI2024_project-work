from platform import node
import numpy as np
import utils.node as Node
import utils.protectedOperators as po
import random
import matplotlib.pyplot as plt
import networkx as nx
import re

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

        if currentDepth >= maxDepth:
            # Create a leaf node with a constant or variable
            if random.random() < 0.5 and self.variables:
                var_name = random.choice(self.variables)
                return Node.Node(var_name, successors=[], name=var_name)
            else:
                const_value = random.choice(CONSTANT)
                return Node.Node(const_value, successors=[], name=str(const_value))
        else:
            if random.random() < 0.7:
                if random.random() < 0.5:
                    # Binary operator
                    op_func = random.choice(list(po.OPERATORS_BINARY.values()))
                    op_name = [k for k, v in po.OPERATORS_BINARY.items() if v == op_func][0]
                    left = self.generateRandomIndividual(currentDepth + 1, maxDepth)
                    right = self.generateRandomIndividual(currentDepth + 1, maxDepth)
                    return Node.Node(op_func, successors=[left, right], name=op_name)
                else:
                    # Unary operator
                    op_func = random.choice(list(po.OPERATORS_UNARY.values()))
                    op_name = [k for k, v in po.OPERATORS_UNARY.items() if v == op_func][0]
                    child = self.generateRandomIndividual(currentDepth + 1, maxDepth)
                    return Node.Node(op_func, successors=[child], name=op_name)
            else:
                # Leaf node
                if random.random() < 0.5 and self.variables:
                    var_name = random.choice(self.variables)
                    return Node.Node(var_name, successors=[], name=var_name)
                else:
                    const_value = random.choice(CONSTANT)
                    return Node.Node(const_value, successors=[], name=str(const_value))

    def computeFitness(self, get_pred=False):
        """
        Compute fitness by evaluating the expression represented by self.root.
        Uses self.xTrain as input variables and self.yTrain as target.
        """
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
                raise ValueError("y_pred contains NaN or Inf")

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
        newIndividual.root = self.root.clone() if self.root is not None else None
        newIndividual.fitness = self.fitness
        return newIndividual

    def getAllNodes(self, node=None, parent=None, isLeft=None, depth=0):
        """
        Get all nodes in the individual as (node, parent, isLeft, depth) tuples.
        """
        if node is None:
            node = self.root
        nodes = [(node, parent, isLeft, depth)]

        # Recurse for each actual successor
        for i, child in enumerate(node.successors):
            if child is not None:
                nodes.extend(self.getAllNodes(child, node, i == 0, depth + 1))

        return nodes

    def getIndividualHeight(self, node):
        if node is None or not node.successors:
            return 1  # leaf node counts as height 1

        left_height = self.getIndividualHeight(node.successors[0]) if len(node.successors) > 0 else 0
        right_height = self.getIndividualHeight(node.successors[1]) if len(node.successors) > 1 else 0

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
        isValid = False

        while attempts > 0 and not isValid:
            oldIndividual, parent, isLeft, nodeDepth = random.choice(nodes)

            remainingDepth = self.maxDepth - nodeDepth
            newIndividual = self.generateRandomIndividual(
                currentDepth = 0,
                maxDepth = remainingDepth
            )

            # Store the original reference for potential revert
            if parent is None:
                self.root = newIndividual
            elif isLeft:
                succ = parent.successors
                succ[0] = newIndividual
                parent.successors = succ
            else:
                succ = parent.successors
                succ[1] = newIndividual
                parent.successors = succ

            # Check fitness
            self.computeFitness()
            if self.fitness != np.inf:
                isValid = True
            else:
                if parent is None:
                    self.root = oldIndividual
                elif isLeft:
                    succ = parent.successors
                    succ[0] = oldIndividual
                    parent.successors = succ
                else:
                    succ = parent.successors
                    succ[1] = oldIndividual
                    parent.successors = succ


            attempts -= 1

    def mutateIndividual(self):
        """
        Randomly change the operator/function of a node without altering structure.
        """
        nodes = self.getAllNodes()
        if not nodes:
            return
        
        node, _, _, _ = random.choice(nodes)
        
        if node.arity == 2:
            new_func = random.choice(po.OPERATORS_BINARY)
        elif node.arity == 1:
            new_func = random.choice(po.OPERATORS_UNARY)
        else:
            return  # No mutation for unusual arity
        
        node.value = new_func

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
        Replace a randomly selected node with one of its children (if any).
        """
        nodes = self.getAllNodes()
        attempts = self.individualAttempts
        if not nodes:
            return

        node, _, _, _ = random.choice(nodes)
        while attempts > 0 and node.successors is not None:
            attempts -= 1
            node, _, _, _ = random.choice(nodes)

        if attempts <= 0:
            return
        
        self.root = node  # Promote the selected node to root
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
            node, parent, isLeft, _ = random.choice(nodes)

            # Only collapse operator nodes (arity >= 1)
            if node.arity == 0:
                attempts -= 1
                continue

            try:
                # Use __str__ to get the full expression string
                expression = str(node)
                eval_globals = {"np": np, "nan": np.nan, "inf": np.inf}

                formula = eval(f"lambda x: {expression}", eval_globals)
                collapsed_value = float(np.mean(formula(self.xTrain)))
            except Exception:
                collapsed_value = 0.0

            # Create new constant node
            new_node = Node.Node(collapsed_value, successors=[], name='constant')

            # Replace node in parent or at root
            if parent is None:
                self.root = new_node
            elif isLeft:
                succ = parent.successors
                succ[0] = new_node
                parent.successors = succ
            else:
                succ = parent.successors
                succ[1] = new_node
                parent.successors = succ

            self.computeFitness()

            if self.fitness != np.inf:
                valid_collapse_found = True
            else:
                # Revert change if invalid
                if parent is None:
                    self.root = node
                elif isLeft:
                    succ = parent.successors
                    succ[0] = node
                    parent.successors = succ
                else:
                    succ = parent.successors
                    succ[1] = node
                    parent.successors = succ

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
            node1, parent1, isLeft1, depth1 = random.choice(nodes1)
            node2, parent2, isLeft2, depth2 = random.choice(nodes2)

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
                elif isLeft1:
                    succ = parent1.successors
                    succ[0] = subtree2
                    parent1.successors = succ
                else:
                    succ = parent1.successors
                    succ[1] = subtree2
                    parent1.successors = succ

                # Swap in offspring2
                if parent2 is None:
                    offspring2.root = subtree1
                elif isLeft2:
                    succ = parent2.successors
                    succ[0] = subtree1
                    parent2.successors = succ
                else:
                    succ = parent2.successors
                    succ[1] = subtree1
                    parent2.successors = succ

                # Compute fitness for both offspring
                offspring1.computeFitness()
                offspring2.computeFitness()

                if offspring1.fitness != np.inf and offspring2.fitness != np.inf:
                    isSwappable = True
                    break  # Valid swap, exit loop
                else:
                    # Revert if invalid
                    if parent1 is None:
                        offspring1.root = node1
                    elif isLeft1:
                        succ = parent1.successors
                        succ[0] = node1
                        parent1.successors = succ
                    else:
                        succ = parent1.successors
                        succ[1] = node1
                        parent1.successors = succ

                    if parent2 is None:
                        offspring2.root = node2
                    elif isLeft2:
                        succ = parent2.successors
                        succ[0] = node2
                        parent2.successors = succ
                    else:
                        succ = parent2.successors
                        succ[1] = node2
                        parent2.successors = succ

            attempts -= 1

        if not isSwappable:
            return self.clone(), recombinationChild.clone()

        return offspring1, offspring2
    
    def size(self):
        """
        Compute the size of the individual (number of nodes).
        """
        def countNodes(node):
            if node is None:
                return 0
            if not hasattr(node, 'successors') or not node.successors:
                return 1
            leftCount = countNodes(node.successors[0]) if len(node.successors) > 0 else 0
            rightCount = countNodes(node.successors[1]) if len(node.successors) > 1 else 0
            return 1 + leftCount + rightCount

        return countNodes(self.root)

    def __str__(self):
        return self.root.__str__() if self.root is not None else ""
    
    op_map = {
        "neg": "−", "abs": "|x|", "sqrt": "√", "exp": "exp", "log": "ln",
        "log2": "log₂", "log10": "log₁₀", "sin": "sin", "cos": "cos",
        "tan": "tan", "asin": "arcsin", "acos": "arccos", "atan": "arctan",
        "sinh": "sinh", "cosh": "cosh", "tanh": "tanh", "sqr": "²",
        "cbrt": "∛", "rec": "1/x", "add": "+", "sub": "−", "mul": "×",
        "div": "÷", "pow": "^", "max": "max", "min": "min", "mod": "mod"
    }

    @staticmethod
    def _parse_expr(expr, counter=[0]):
        expr = expr.strip()
        m = re.match(r'([a-zA-Z0-9_]+)\((.*)\)$', expr)
        if m:
            func, args = m.group(1), m.group(2)
            node_id = f"n{counter[0]}"; counter[0] += 1
            node = (node_id, Individual.op_map.get(func, func))
            # split arguments at commas with parenthesis depth
            parts, depth, buf = [], 0, ""
            for c in args:
                if c == "," and depth == 0:
                    if buf.strip(): parts.append(buf.strip())
                    buf = ""
                else:
                    if c == "(": depth += 1
                    elif c == ")": depth -= 1
                    buf += c
            if buf.strip(): parts.append(buf.strip())
            children = [Individual._parse_expr(p, counter) for p in parts]
            return (node, children)
        else:
            node_id = f"n{counter[0]}"; counter[0] += 1
            label = expr
            if re.match(r'^(\d+(\.\d+)?|\.\d+)$', label):
                label = f"{float(label):.2f}"
            return ((node_id, label), [])

    @staticmethod
    def _build_graph(tree, G=None):
        if G is None:
            G = nx.DiGraph()
        (node, children) = tree
        G.add_node(node[0], label=node[1])
        for child in children:
            (cnode, _) = child
            G.add_edge(node[0], cnode[0])
            Individual._build_graph(child, G)
        return G

    @staticmethod
    def _hierarchy_pos(G, root, width=1., vert_gap=0.2, vert_loc=0):
        leaves = [n for n in nx.dfs_preorder_nodes(G) if G.out_degree(n) == 0]
        n_leaves = len(leaves)
        x_leaf = {leaf: i/(n_leaves-1) if n_leaves > 1 else 0.5
                  for i, leaf in enumerate(leaves)}
        pos = {}
        def set_pos(node, vert_loc):
            children = list(G.successors(node))
            if not children:
                pos[node] = (x_leaf[node], vert_loc)
            else:
                for c in children: set_pos(c, vert_loc - vert_gap)
                x = sum(pos[c][0] for c in children) / len(children)
                pos[node] = (x, vert_loc)
        set_pos(root, vert_loc)
        return pos

    def plot(self):
        """Parse self into tree and draw with networkx."""
        expr = str(self)
        expr = re.sub(r"\s+", "", expr)  # clean spaces
        tree = Individual._parse_expr(expr, [0])
        G = Individual._build_graph(tree)
        root = tree[0][0]
        pos = Individual._hierarchy_pos(G, root)
        plt.figure(figsize=(40, 24))
        for n, data in G.nodes(data=True):
            label = data["label"]
            if re.match(r'^(\d+(\.\d+)?|\.\d+)$|^x\d+$', label):
                plt.scatter(*pos[n], s=500, c="blue", alpha=0.3,
                            marker="s", edgecolors="k")
            else:
                plt.scatter(*pos[n], s=500, c="red", alpha=0.3,
                            marker="o", edgecolors="k")
            plt.text(pos[n][0], pos[n][1], label,
                     ha="center", va="center", fontsize=10)
        nx.draw_networkx_edges(G, pos, arrows=True,
                               edge_color="black", arrowsize=10)
        plt.axis("off")
        plt.show()
from typing import List, Dict
import math
import time
import heapq
from mcts import mcts as MCTS
from mcts import treeNode as TreeNode

from dr_claude import datamodels
from dr_claude.planning import states

CONDITION_PREDICTION_DELAY_FACTOR = 0.65


class MultiChildMixin:
    def search(self: MCTS, initialState: states.StateBase, top_k: int) -> List:
        self.root = TreeNode(initialState, None)

        if self.limitType == "time":
            timeLimit = time.time() + self.timeLimit / 1000
            while time.time() < timeLimit:
                self.executeRound()
        else:
            for _ in range(self.searchLimit):
                self.executeRound()

        bestChild = self.getBestChild(self.root, 0, top_k)
        return self.getAction(self.root, bestChild)

    def getBestChild(
        self: MCTS, node: TreeNode, explorationValue: float, top_k: int = 1
    ) -> List[TreeNode]:
        node_values = []
        for i, child in enumerate(node.children.values()):
            nodeValue = (
                child.totalReward / child.numVisits
                + explorationValue
                * math.sqrt(2 * math.log(node.numVisits) / child.numVisits)
            )
            heapq.heappush(node_values, (nodeValue, i, child))
            # Keep only the top_k node values
            if len(node_values) > top_k:
                heapq.heappop(node_values)

        # Return the children associated with the top_k node values
        top_k_nodes = [heapq.heappop(node_values)[2] for _ in range(len(node_values))]
        # The nodes are popped in ascending order, so reverse the list
        top_k_nodes.reverse()
        return top_k_nodes

    def getAction(self, root: TreeNode, bestChild: List[TreeNode]):
        nodes: List[TreeNode] = []
        for action, node in root.children.items():
            if node in bestChild:
                nodes.append(action)
        return nodes

    def selectNode(self: MCTS, node):
        while not node.isTerminal:
            if node.isFullyExpanded:
                node = self.getBestChild(node, self.explorationConstant, top_k=1)[0]
            else:
                return self.expand(node)[0]
        return node

    def expand(self, node: TreeNode):
        """
        The default method will simply expand one action at a time and return them. However,
        to make this more efficient, we should expand all the actions at once and set a good prior
        for them, then use UCT to choose the best child.
        """
        actions = node.state.getPossibleActions()
        symptom_probs: Dict[
            datamodels.Symptom, float
        ] = node.state.dynamics.getSymptomProbabilityDict(
            node.state.pertinent_pos, node.state.pertinent_neg
        )
        condition_probs: Dict[
            datamodels.Condition, float
        ] = node.state.dynamics.getConditionProbabilityDict(
            node.state.pertinent_pos, node.state.pertinent_neg
        )
        for action in actions:
            if action not in node.children:
                child_node = TreeNode(node.state.takeAction(action), node)
                child_node.numVisits = 1
                if child_node.isTerminal:
                    child_node.totalReward += (
                        condition_probs[action] * CONDITION_PREDICTION_DELAY_FACTOR
                    )
                else:
                    child_node.totalReward += symptom_probs[action]
                node.children[action] = child_node
        if len(actions) == len(node.children):
            node.isFullyExpanded = True
        if node.numVisits == 0:
            node.numVisits += len(node.children)
        return self.getBestChild(node, self.explorationConstant, top_k=1)


class DrClaudeMCTS(MultiChildMixin, MCTS):
    ...

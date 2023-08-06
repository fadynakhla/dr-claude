from typing import Any
import math
import time
import heapq
from mcts import mcts as MCTS
from mcts import treeNode as TreeNode

from dr_claude.planning import states


class MultiChildMixin(MCTS):
    def search(self, initialState: states.ActionState, top_k: int):
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

    def getBestChild(self: MCTS, node: TreeNode, explorationValue: float, top_k: int):
        node_values = []
        for i, child in enumerate(node.children.values()):
            nodeValue = (
                child.totalReward / child.numVisits
                + explorationValue
                * math.sqrt(2 * math.log(node.numVisits) / child.numVisits)
            )
            # Use negative value because heapq is a min heap, i to break ties
            heapq.heappush(node_values, (-nodeValue, i, child))
            # Keep only the top_k node values
            if len(node_values) > top_k:
                heapq.heappop(node_values)

        # Return the children associated with the top_k node values
        top_k_nodes = [heapq.heappop(node_values)[2] for _ in range(len(node_values))]
        # The nodes are popped in ascending order, so reverse the list
        top_k_nodes.reverse()
        return top_k_nodes

    def getAction(self, root, bestChild):
        nodes = []
        for action, node in root.children.items():
            if node in bestChild:
                nodes.append(action)
        return nodes


class MultiChoiceMCTS(MultiChildMixin, mcts):
    ...

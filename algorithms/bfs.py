from collections import deque
from typing import List, Optional


class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


# Same Tree - https://leetcode.com/problems/same-tree/description/?envType=problem-list-v2&envId=breadth-first-search
# Time Complexity: O(min(N, M)) where N and M are the number of nodes in the two trees, worst case both trees are identical
# Space Complexity: O(min(H1, H2)) where H1 and H2 are the heights of the two trees, worst case both trees are skewed
def is_same_tree(self, p: Optional[TreeNode], q: Optional[TreeNode]) -> bool:
    if p is None and q is None:
        return True

    elif p is None or q is None:
        return False

    elif p.val == q.val:
        return self.is_same_tree(p.left, q.left) and self.is_same_tree_recursive(
            p.right, q.right
        )

    return False


# Binary Tree Level Order Traversal - https://leetcode.com/problems/binary-tree-level-order-traversal/description/?envType=problem-list-v2&envId=breadth-first-search
# Time Complexity: O(N) where N is the number of nodes in the tree
# Space Complexity: O(W) where W is the maximum width of the tree
def levelOrder(self, root: Optional[TreeNode]) -> List[List[int]]:
    if not root:
        return []

    result = []
    queue = deque([root])

    while len(queue) > 0:
        level = []
        for i in range(len(queue)):
            node = queue.popleft()
            level.append(node.val)
            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)
        result.append(level)

    return result


# Reverse odd levels of binary tree - https://leetcode.com/problems/reverse-odd-levels-of-binary-tree/description/?envType=problem-list-v2&envId=breadth-first-search
# # Time Complexity: O(N) where N is the number of nodes in the tree
# Space Complexity: O(W) where W is the maximum width of the tree
def reverseOddLevels(root: Optional[TreeNode]) -> Optional[TreeNode]:
    if not root:
        return None

    level = 0
    queue = deque([root])

    while len(queue) > 0:
        if level % 2 != 0:
            lp = 0
            rp = len(queue) - 1
            while lp < rp:
                queue[lp].val, queue[rp].val = queue[rp].val, queue[lp].val
                lp += 1
                rp -= 1

        for i in range(len(queue)):
            node = queue.popleft()
            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)
        level += 1

    return root


# Deepest Leaves Sum - https://leetcode.com/problems/deepest-leaves-sum/description/?envType=problem-list-v2&envId=breadth-first-search
# Time Complexity: O(N) where N is the number of nodes in the tree
# Space Complexity: O(W) where W is the maximum width of the tree
def deepestLeavesSum(root: Optional[TreeNode]) -> int:
    if root is None:
        return 0

    level = 0
    queue = deque([root])
    result_by_row = {}

    while queue:
        row_sum = 0
        for i in range(len(queue)):
            node = queue.popleft()
            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)
            row_sum += node.val
        result_by_row[level] = row_sum
        level += 1

    return result_by_row[level - 1]

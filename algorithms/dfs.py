from typing import List, Optional


class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


# Range Sum of BST - https://leetcode.com/problems/range-sum-of-bst/description/?envType=problem-list-v2&envId=depth-first-search
# Time Complexity: O(N)
# Space Complexity: O(H)
def rangeSumBST(self, root: Optional[TreeNode], low: int, high: int) -> int:
    if root is None:
        return 0
    elif root.val < low:
        return self.rangeSumBST(root.right, low, high)
    elif root.val > high:
        return self.rangeSumBST(root.left, low, high)
    return (
        root.val
        + self.rangeSumBST(root.left, low, high)
        + self.rangeSumBST(root.right, low, high)
    )


# Binary Search Tree to Greater Sum Tree - https://leetcode.com/problems/binary-search-tree-to-greater-sum-tree/description/?envType=problem-list-v2&envId=depth-first-search
# Time Complexity: O(N)
# Space Complexity: O(H)
class SolutionBstToGst:
    def bstToGst(self, root: Optional[TreeNode]) -> Optional[TreeNode]:
        self.val = 0

        def reverse_in_order_traversal(node: Optional[TreeNode]):
            if not node:
                return

            reverse_in_order_traversal(node.right)
            self.val += node.val
            node.val = self.val
            reverse_in_order_traversal(node.left)

        reverse_in_order_traversal(root)

        return root


# Number of islands - https://leetcode.com/problems/number-of-islands/description/
# Time Complexity: O(M*N) where M is number of rows and N is number of columns
# Space Complexity: O(M*N) in the worst case when the grid is filled with lands
class SolutionNumIslands:
    def numIslands(self, grid: List[List[str]]) -> int:
        if not grid:
            return 0

        count = 0
        for i in range(len(grid)):
            for j in range(len(grid[0])):
                if grid[i][j] == "1":
                    self.dfs(grid, i, j)
                    count += 1
        return count

    def dfs(self, grid, i, j):
        if i < 0 or j < 0 or i >= len(grid) or j >= len(grid[0]) or grid[i][j] != "1":
            return
        grid[i][j] = "#"
        self.dfs(grid, i + 1, j)
        self.dfs(grid, i - 1, j)
        self.dfs(grid, i, j + 1)
        self.dfs(grid, i, j - 1)

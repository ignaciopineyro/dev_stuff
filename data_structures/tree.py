class TreeNode:
    def __init__(self, value):
        self.value = value
        self.children = []

    def add_child(self, child_node):
        self.children.append(child_node)

    def __repr__(self):
        return f"TreeNode({self.value})"


class Tree:
    def __init__(self, root_value):
        self.root = TreeNode(root_value)

    def traverse(self, node=None, depth=0):
        if node is None:
            node = self.root
        print("  " * depth + str(node.value))
        for child in node.children:
            self.traverse(child, depth + 1)


tree = Tree("root")
child1 = TreeNode("child1")
child2 = TreeNode("child2")
tree.root.add_child(child1)
tree.root.add_child(child2)
child1.add_child(TreeNode("child1.1"))
child2.add_child(TreeNode("child2.1"))
tree.traverse()

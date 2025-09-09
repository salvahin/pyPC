class Node:
    def __init__(self, value):
        self.value = value
        self.left = None
        self.right = None

class BinarySearchTree:
    def __init__(self):
        self.root = None
    
    def insert(self, value):
        if self.root is None:
            self.root = Node(value)
        else:
            self._insert_recursive(self.root, value)
    
    def _insert_recursive(self, node, value):
        if value < node.value:
            if node.left is None:
                node.left = Node(value)
            else:
                self._insert_recursive(node.left, value)
        elif value > node.value:
            if node.right is None:
                node.right = Node(value)
            else:
                self._insert_recursive(node.right, value)
    
    def search(self, value):
        return self._search_recursive(self.root, value)
    
    def _search_recursive(self, node, value):
        if node is None:
            return False
        
        if value == node.value:
            return True
        elif value < node.value:
            return self._search_recursive(node.left, value)
        else:
            return self._search_recursive(node.right, value)
    
    def find_min(self):
        if self.root is None:
            return None
        
        current = self.root
        while current.left is not None:
            current = current.left
        return current.value
    
    def find_max(self):
        if self.root is None:
            return None
        
        current = self.root
        while current.right is not None:
            current = current.right
        return current.value
    
    def height(self):
        return self._height_recursive(self.root)
    
    def _height_recursive(self, node):
        if node is None:
            return 0
        
        left_height = self._height_recursive(node.left)
        right_height = self._height_recursive(node.right)
        
        return max(left_height, right_height) + 1
    
    def is_balanced(self):
        return self._is_balanced_recursive(self.root)[0]
    
    def _is_balanced_recursive(self, node):
        if node is None:
            return True, 0
        
        left_balanced, left_height = self._is_balanced_recursive(node.left)
        if not left_balanced:
            return False, 0
        
        right_balanced, right_height = self._is_balanced_recursive(node.right)
        if not right_balanced:
            return False, 0
        
        is_balanced = abs(left_height - right_height) <= 1
        height = max(left_height, right_height) + 1
        
        return is_balanced, height

def test_bst(a, b, c, d):
    bst = BinarySearchTree()
    values = [a, b, c, d]
    
    for val in values:
        if val > 0:
            bst.insert(val)
    
    if bst.root is None:
        return 0
    
    min_val = bst.find_min()
    max_val = bst.find_max()
    
    if min_val is None or max_val is None:
        return 1
    
    if bst.search(a):
        if bst.search(b):
            if bst.height() > 2:
                if bst.is_balanced():
                    return 10
                else:
                    return 8
            else:
                return 5
        else:
            return 3
    else:
        return 2

if __name__ == '__main__':
    result = test_bst(5, 3, 7, 1)
    print(f"Result: {result}")
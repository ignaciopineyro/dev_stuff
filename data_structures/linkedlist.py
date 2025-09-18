class LinkedListNode:
    def __init__(self, data):
        self.data = data
        self.next = None

class LinkedList:
    def __init__(self):
        self.head = None

    def append(self, data) -> None:
        new_node = LinkedListNode(data)

        if not self.head:
            self.head = new_node
            return
        
        curr = self.head
        while curr.next:
            curr = curr.next

        curr.next = new_node

    def display(self) -> None:
        elems = []
        curr = self.head
        while curr:
            elems.append(curr.data)
            curr = curr.next
        print(elems)

ll = LinkedList()
ll.append(1)
ll.append(2)
ll.append(3)
ll.display()
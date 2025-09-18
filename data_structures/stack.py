# Stack using list
# - Implemented using list
# - LIFO

stack = []
stack.append(1)        # [1]
stack.append(2)        # [1, 2]
top = stack[-1]        # Peek at the top element
stack.pop()            # [1]

print(stack)
print(top)
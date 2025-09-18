from collections import deque

# QUEUE
# - Implemented using deque which supports queue and stack operations
# - FIFO

queue = deque([1, 2, 3])
queue.append(4) # Enqueue
queue.popleft() # Dequeue

print(queue)
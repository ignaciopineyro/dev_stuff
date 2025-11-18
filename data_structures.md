# DATA STRUCTURES

Data structures are ways to organize, store, and manage data in a computer so that it can be accessed and modified efficiently. They are essential for writing efficient algorithms and for solving complex computational problems.

---

## **Primitive Data Structures**

Basic types provided by most programming languages.
Examples: Integer, Float, Character, Boolean.

---

## **Linear Data Structures**

Linear data structures are collections of elements arranged sequentially, where each element has a unique predecessor and successor (except the first and last).They're called linear because the data items are stored or logically arranged in a straight line — you can traverse them in a single run from start to end. Linear data structures are simple to process sequentially and easy to implement but are inflexible, have shifting elements and are sequentialy dependent (you must pass through elements in order (unless indexed)).

- Single-level traversal: One element follows another.
- Fixed or dynamic size: Some have predetermined length, others grow/shrink.
- Order matters: The position of elements is important.
- Access patterns: Sequential or direct (depending on type).

| Structure     | Description                                           | Typical Operations                         |
|---------------|-------------------------------------------------------|---------------------------------------------|
| **Array**     | Fixed-size, indexed collection of elements of the same type. | Access by index, update, iterate.           |
| **Linked List** | Nodes linked by pointers/references.                 | Insert, delete, traverse.                   |
| **Stack**     | LIFO (Last In, First Out) structure.                   | `push`, `pop`, `peek`.                      |
| **Queue**     | FIFO (First In, First Out) structure.                  | `enqueue`, `dequeue`, `peek`.               |
| **Deque**     | Double-ended queue; insert/remove at both ends.        | `append`, `appendleft`, `pop`, `popleft`.   |


---

## **Non-Linear Data Structures**

Non-linear data structures store data in a way where elements are not arranged sequentially. Instead, they are organized in a hierarchical or networked manner, allowing for multiple paths of traversal. This allows complex models relationships naturally, enables fast searching in certain structures (e.g., BST, heap) and allows flexible connections (e.g., graphs can represent many-to-many relationships). On the other hand, they are more complex to implement and maintain, traversal algorithms (DFS, BFS) require more logic than linear structures and can consume more memory due to pointer/references overhead.

- Multiple relationships: Each element can connect to multiple others.
- No strict predecessor/successor: Unlike linear structures.
- Efficient representation of complex relationships (hierarchies, graphs).
- Flexible traversal: Multiple possible paths instead of a single line.

| Structure                  | Description                                                       | Typical Use Cases                               |
|----------------------------|-------------------------------------------------------------------|-------------------------------------------------|
| **Tree**                   | Hierarchical structure with a root node and child nodes.          | File systems, organization charts, XML/HTML DOM. |
| **Binary Tree**            | Tree where each node has at most two children.                    | Searching, sorting, expression parsing.        |
| **Binary Search Tree (BST)** | Binary tree with ordered nodes (left < parent < right).          | Fast lookups, ordered data storage.            |
| **Heap**                   | Complete binary tree with heap property (min or max at root).     | Priority queues, scheduling.                   |
| **Graph**                  | Collection of nodes (vertices) connected by edges.                | Social networks, maps, recommendation systems. |


---

## **Hash-Based Structures**

Hash-based structures are data structures that use a hash function to map keys to positions (indexes or buckets) in memory, allowing for fast access, insertion, and deletion. They're widely used when quick lookups are needed. They accept many key types (strings, numbers, tuples, etc.).And are great for large datasets with frequent searches. On the other hand, collisions reduce efficiency.

- Key-value mapping: Store data as pairs (key → value).
- Hash function: Converts a key into an index/bucket number.
- Collision handling: Required when different keys map to the same index.
- Average O(1) time complexity for search, insert, and delete.

| Structure         | Description                                                      | Typical Use Cases                         |
|-------------------|------------------------------------------------------------------|-------------------------------------------|
| **Hash Table**    | Array + hash function; resolves collisions by chaining or open addressing. | Database indexing, caches.               |
| **Hash Map**      | Key-value store without guaranteed order (e.g., Python `dict`).  | Config settings, frequency counting.      |
| **Hash Set**      | Stores only unique keys (no values).                             | Removing duplicates, membership testing.  |
| **Bloom Filter**  | Probabilistic structure for membership tests (may have false positives). | Web caching, spell checkers.             |

# Python

## CPython

CPython is the standard and most common implementation of the Python programming language. It's implemented in the C programming language and is the official reference implementation.

It can be defined as **both an interpreter and a compiler** since it reads your Python source code (.py files) and compiles it into bytecode, a simplified low-level representation of your code. Then executes that bytecode using the Python Virtual Machine (PVM), which is part of CPython.

Cpython uses **reference counting** for memory management (each object keeps track of how many references point to it). It can integrate easily with C extensions (one reason libraries like NumPy are extremely fast). It also has the greatest ecosystem compatibility, since most libraries are built with CPython in mind.

## Garbage Collector

Python’s garbage collector is responsible for automatically managing memory so developers don’t have to manually allocate and free it (like in C). Its core mechanism is reference counting, where every object tracks how many references point to it. When that count reaches zero, the object’s memory can be immediately reclaimed. This makes memory management predictable and straightforward.

However, reference counting alone can’t handle reference cycles—situations where two or more objects reference each other even though they’re no longer in use. To address this, Python includes a cyclic garbage collector that periodically scans for these groups of unreachable objects and frees them.

The Global Interpreter Lock (GIL) plays a role in keeping this process safe. Because reference counts are modified constantly as objects are created, passed around, or deleted, they must be updated atomically. The GIL ensures that only one thread executes Python bytecode at a time, which prevents race conditions that could corrupt reference counts.

## GIL

The GIL (Global Interpreter Lock) is a mechanism that exists specifically in CPython. It is essentially a global lock that **prevents multiple threads** from executing Python bytecode at the same time inside the interpreter. CPython’s memory management relies on reference counting (the reference count of an object increases by one whenever a new reference to it is created, as in assigning an object to a new variable, for example). If several threads modified reference counts simultaneously, it could easily cause corruption or crashes. The GIL ensures only one thread runs Python bytecode at a time in order to prevent other threads to diminish the count on a variable that could be still be in use by other thread.

**For I/O-bound tasks** (networking, file access, waiting for responses), threads work well. This is because the GIL is released during I/O operations. But **for CPU-bound tasks** (intensive computing) where threads do not run in parallel (slower performance) we have to find a solution. 

Some possible solutions are:
- **multiprocessing module**: Creates separate processes, each with its own interpreter and its own GIL. The pros of this methods are real multi-core usage and that it's easy to implement. On the other hand it implies higher overhead than threading and inter-process communication is more expensive.
- **AsyncIO**: Asyncio does not remove the GIL, but avoids threads entirely.It works extremely well when the bottleneck is waiting, not computation.
- **Use libraries written in C/C++ that release the GIL**: NumPy, SciPy,pandas (certain parts), TensorFlow/PyTorch.

## Concurrency

Concurrency refers to the ability of a system to handle multiple tasks seemingly simultaneously by rapidly switching between them. In web applications, concurrency is crucial for managing multiple client requests without blocking the entire application. Unlike parallelism, concurrency doesn't require multiple CPU cores - it's achieved through clever scheduling and task interleaving.

**Time-slicing**: The CPU allocates small time slices to different tasks, creating the illusion of simultaneous execution. Modern operating systems use preemptive multitasking where the scheduler forcibly switches between tasks.

**Non-blocking I/O**: Instead of waiting for I/O operations (database queries, file reads, network requests) to complete, concurrent systems can switch to other tasks. This is particularly effective in web apps where I/O operations are frequent bottlenecks.

**Coroutines and Green Threads**: Lightweight threading mechanisms that allow cooperative multitasking. Libraries like `asyncio` in Python implement coroutines that yield control voluntarily at defined suspension points.

**Event-driven Architecture**: Systems respond to events (HTTP requests, database callbacks, timer expirations) rather than following a linear execution flow. This enables handling thousands of concurrent connections with minimal resource overhead.

### Implementation Strategies:

- **Reactor Pattern**: Single-threaded event loop that dispatches events to appropriate handlers
- **Thread Pools**: Pre-allocated threads that handle incoming requests from a queue
- **Cooperative Threading**: Tasks voluntarily yield control at specific points
- **Message Passing**: Tasks communicate through queues rather than shared memory

Concurrency is essential for web applications to achieve high throughput and responsiveness, especially when dealing with I/O-bound operations that dominate web request processing.

## Parallelism

Parallelism involves the simultaneous execution of multiple tasks across multiple processing units (CPU cores, threads, or distributed machines). Unlike concurrency, which creates the appearance of simultaneity, parallelism achieves true simultaneous execution through hardware-level parallel processing capabilities.

### Types of Parallelism:

**Data Parallelism**: The same operation is performed on different data sets simultaneously. Common in scenarios like batch processing user uploads, image resizing, or mathematical computations across large datasets.

**Task Parallelism**: Different tasks or functions execute simultaneously on separate processing units. Web applications might parallelize database queries, cache updates, and business logic processing.

**Pipeline Parallelism**: Tasks are divided into stages, with each stage processing different parts of the workload simultaneously. HTTP request processing can be pipelined through parsing, authentication, business logic, and response generation stages.

### Hardware Considerations:

**Multi-core Processors**: Modern CPUs provide multiple cores that can execute instructions simultaneously. Web servers leverage this through thread pools or process pools that distribute work across cores.

**NUMA (Non-Uniform Memory Access)**: In multi-processor systems, memory access times vary depending on CPU-memory proximity. Optimizing data locality becomes crucial for performance.

**Cache Coherency**: When multiple cores access shared memory, cache synchronization overhead can reduce parallel efficiency. Lock-free data structures and careful memory layout optimization are essential.

### Parallel Programming Models:

**Shared Memory**: Multiple threads access common memory space, requiring synchronization primitives (mutexes, semaphores, atomic operations) to prevent race conditions and ensure data consistency.

**Message Passing**: Processes communicate through explicit message exchange, eliminating shared state issues but introducing communication overhead. Common in distributed web architectures and microservices.

**Actor Model**: Computation units (actors) encapsulate state and communicate only through message passing, providing natural parallelization boundaries.

### Challenges in Web Applications:

**GIL (Global Interpreter Lock)**: In CPython, the GIL prevents true thread-level parallelism for CPU-bound tasks, necessitating process-based parallelism or alternative implementations.

**Load Balancing**: Distributing work evenly across parallel workers to avoid bottlenecks and maximize resource utilization.

**Race Conditions**: Multiple parallel tasks accessing shared resources can lead to data corruption or inconsistent states without proper synchronization.

**Scalability Bottlenecks**: Amdahl's Law dictates that the speedup from parallelization is limited by the sequential portion of the program. Database connections, shared caches, and serialization points can limit parallel scaling.

Effective parallelism in web applications requires careful architecture design, considering both the computational workload characteristics and the underlying hardware capabilities to achieve optimal performance scaling.

## Multiprocesing

## Multithreading

## Event loop / async

## AsyncIO

## Context switch

## async, await


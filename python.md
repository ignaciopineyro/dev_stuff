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

## Parallelism

## Multiprocesing

## Multithreading

## Event loop / async

## AsyncIO

## Context switch

## async, await


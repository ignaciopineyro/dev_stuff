# GOOD PRACTICES
---

## **SOLID principles**

*ArjanCodes: https://www.youtube.com/watch?v=pTB30aXS77U*

SOLID is an acronym for five design principles intended to make object-oriented designs more understandable, flexible, and maintainable.

1. **Single responsibility principle (SRP)**: There should never be more than one reason for a class to change. In other words, every class should have only one responsibility (high cohesion).
    - Maintainability: When classes have a single, well-defined responsibility, they're easier to understand and modify.
    - Testability: It's easier to write unit tests for classes with a single focus.
    - Flexibility: Changes to one responsibility don't affect unrelated parts of the system

2. **Open-closed principle**: Software entities should be open for extension, but closed for modification, which means you should be able to extend a class behavior, without modifying it.
    - Extensibility: New features can be added without modifying existing code.
    - Stability: Reduces the risk of introducing bugs when making changes.
    - Flexibility: Adapts to changing requirements more easily.

3. **Liskov substitution principle (LSP)**: Derived or child classes must be substitutable for their base or parent classes.
    - Polymorphism: Enables the use of polymorphic behavior, making code more flexible and reusable.
    - Reliability: Ensures that subclasses adhere to the contract defined by the superclass.
    - Predictability: Guarantees that replacing a superclass object with a subclass object won't break the program.

4. **Interface segregation principle (ISP)**: Clients should not be forced to depend upon interfaces that they do not use. It's preferable to have many client interfaces rather than one general interface and each interface should have a specific responsibility.
    - Decoupling: Reduces dependencies between classes, making the code more modular and maintainable.
    - Flexibility: Allows for more targeted implementations of interfaces.
    - Avoids unnecessary dependencies: Clients don't have to depend on methods they don't use.

5. **Dependency inversion principle (DIP)**: High-level modules should not depend on low-level modules. In simpler terms, the DIP suggests that classes should rely on abstractions (e.g., interfaces or abstract classes) rather than concrete implementations.
    - Loose coupling: Reduces dependencies between modules, making the code more flexible and easier to test.
    - Flexibility: Enables changes to implementations without affecting clients.
    - Maintainability: Makes code easier to understand and modify.

---

## **Composition over inheritance principle**

COI is the principle that classes should favor polymorphic behavior and code reuse by their composition (by containing instances of other classes that implement the desired functionality) over inheritance from a base or parent class which is associated with tight coupling and makes a system rigid and harder to modify.

---

## **DRY principle (Don’t Repeat Yourself)**

The DRY principle emphasizes that each piece of knowledge or logic must have a single, unambiguous representation in the system. This principle promotes maintainability and helps reduce errors.

---

## **KISS principle (Keep it simple, stupid)**

The KISS principle suggests that the best solutions are often the simplest ones, and developers should strive to avoid unnecessary complexity.

---

## **YAGNI principle (You Aren’t Gonna Need It)**

YAGNI encourages developers not to add functionality until it is necessary. This principle helps to keep the codebase manageable and prevents over-engineering.

---

## **GRASP principles (General Responsibility Assignment Software Patterns)**

GRASP provides guidelines for assigning responsibilities to classes and objects in object-oriented design. Is a set of nine fundamental principles in object design and responsibility assignment:
- **Controller**: Assign system events to a controller class that represents a use case or session.
This principle prevents UI components from handling business logic.
- **Creator**: A class should create instances of another class if contains objects of that class, uses the created object or has the necessary information to initialize the object.
- **Indirection**: Use an intermediary to decouple components and improve flexibility. This is often seen in mediators, adapters, or proxy patterns.
- **Information expert**: Used to determine where to delegate responsibilities such as methods, computed fields, etc, in order to improve cohesion and reduce dependency.
- **Low coupling**: Reduce dependencies between classes to increase maintainability and flexibility. Classes should depend on abstractions rather than concrete implementations.
- **High cohesion**: A class should have a single, clear purpose to improve readability and reusability. A class with low cohesion does too many unrelated tasks.
- **Polymorphism**: Use polymorphism to handle different cases instead of conditional logic. Instead of checking types with if statements, rely on method overriding.
- **Protected variations**: Design the system to protect against changes that could break functionality. Use design patterns like Dependency Injection, Interfaces, and Encapsulation.
- **Pure fabrication**: Create helper classes when no natural class should take a responsibility. This avoids placing unrelated responsibilities in existing domain objects.

---

## **LoD principle (Law of Demeter or Principle of Least Knowledge)**

The LoD states that an object should only communicate with its immediate neighbors and should have limited knowledge about other objects in the system. It is particularly beneficial in complex systems where tight coupling can lead to increased dependencies and decreased modularity.

# DESIGN PATTERNS
---

## **Creational Patterns (how objects are created)**

### **Singleton**

Singleton is a creational design pattern, which ensures that only one object of its kind exists and provides a single point of access to it for any other code. Singleton has almost the same pros and cons as global variables. Although they’re super-handy, they break the modularity of your code. You can’t just use a class that depends on a Singleton in some other context, without carrying over the Singleton to the other context. Most of the time, this limitation comes up during the creation of unit tests.

```
class SingletonMeta(type):
    '''
    The Singleton class can be implemented in different ways in Python. Some
    possible methods include: base class, decorator, metaclass. We will use the
    metaclass because it is best suited for this purpose.
    '''

    _instances = {}

    def __call__(cls, *args, **kwargs):
        '''
        Possible changes to the value of the `__init__` argument do not affect
        the returned instance.
        '''
        if cls not in cls._instances:
            instance = super().__call__(*args, **kwargs)
            cls._instances[cls] = instance
        return cls._instances[cls]


class Singleton(metaclass=SingletonMeta):
    def some_business_logic(self):
        '''
        Finally, any singleton should define some business logic, which can be
        executed on its instance.
        '''

        # ...


if __name__ == "__main__":
    # The client code.

    s1 = Singleton()
    s2 = Singleton()

    if id(s1) == id(s2):
        print("Singleton works, both variables contain the same instance.")
    else:
        print("Singleton failed, variables contain different instances.")
```

---

### **Factory Method**

Factory method is a creational design pattern which solves the problem of creating product objects without specifying their concrete classes.
It defines a method, which should be used for creating objects instead of using a direct constructor call (new operator). Subclasses can override this method to change the class of objects that will be created. It’s very useful when you need to provide a high level of flexibility for your code.

```
from __future__ import annotations
from abc import ABC, abstractmethod


class Creator(ABC):
    '''
    The Creator class declares the factory method that is supposed to return an
    object of a Product class. The Creator's subclasses usually provide the
    implementation of this method.
    '''

    @abstractmethod
    def factory_method(self):
        '''
        Note that the Creator may also provide some default implementation of
        the factory method.
        '''
        pass

    def some_operation(self) -> str:
        '''
        Also note that, despite its name, the Creator's primary responsibility
        is not creating products. Usually, it contains some core business logic
        that relies on Product objects, returned by the factory method.
        Subclasses can indirectly change that business logic by overriding the
        factory method and returning a different type of product from it.
        '''

        # Call the factory method to create a Product object.
        product = self.factory_method()

        # Now, use the product.
        result = f"Creator: The same creator's code has just worked with {product.operation()}"

        return result


'''
Concrete Creators override the factory method in order to change the resulting
product's type.
'''


class ConcreteCreator1(Creator):
    '''
    Note that the signature of the method still uses the abstract product type,
    even though the concrete product is actually returned from the method. This
    way the Creator can stay independent of concrete product classes.
    '''

    def factory_method(self) -> Product:
        return ConcreteProduct1()


class ConcreteCreator2(Creator):
    def factory_method(self) -> Product:
        return ConcreteProduct2()


class Product(ABC):
    '''
    The Product interface declares the operations that all concrete products
    must implement.
    '''

    @abstractmethod
    def operation(self) -> str:
        pass


'''
Concrete Products provide various implementations of the Product interface.
'''


class ConcreteProduct1(Product):
    def operation(self) -> str:
        return "{Result of the ConcreteProduct1}"


class ConcreteProduct2(Product):
    def operation(self) -> str:
        return "{Result of the ConcreteProduct2}"


def client_code(creator: Creator) -> None:
    '''
    The client code works with an instance of a concrete creator, albeit through
    its base interface. As long as the client keeps working with the creator via
    the base interface, you can pass it any creator's subclass.
    '''

    print(f"Client: I'm not aware of the creator's class, but it still works.\n"
          f"{creator.some_operation()}", end="")


if __name__ == "__main__":
    print("App: Launched with the ConcreteCreator1.")
    client_code(ConcreteCreator1())
    print("\n")

    print("App: Launched with the ConcreteCreator2.")
    client_code(ConcreteCreator2())
```

---

### **Abstract Factory**

Abstract Factory is a creational design pattern, which solves the problem of creating entire product families without specifying their concrete classes.
It defines an interface for creating all distinct products but leaves the actual product creation to concrete factory classes. Each factory type corresponds to a certain product variety. The client code calls the creation methods of a factory object instead of creating products directly with a constructor call (new operator). Since a factory corresponds to a single product variant, all its products will be compatible.Client code works with factories and products only through their abstract interfaces. This lets the client code work with any product variants, created by the factory object. You just create a new concrete factory class and pass it to the client code.

The main difference between a Factory Method and an Abstract Factory is that the factory method is a method, and an abstract factory is an object. Also, the Factory Method delegates the creation of one product while Abstract Factory creates a set of related products together.

```
from __future__ import annotations
from abc import ABC, abstractmethod


class AbstractFactory(ABC):
    '''
    The Abstract Factory interface declares a set of methods that return
    different abstract products. These products are called a family and are
    related by a high-level theme or concept. Products of one family are usually
    able to collaborate among themselves. A family of products may have several
    variants, but the products of one variant are incompatible with products of
    another.
    '''
    @abstractmethod
    def create_product_a(self) -> AbstractProductA:
        pass

    @abstractmethod
    def create_product_b(self) -> AbstractProductB:
        pass


class ConcreteFactory1(AbstractFactory):
    '''
    Concrete Factories produce a family of products that belong to a single
    variant. The factory guarantees that resulting products are compatible. Note
    that signatures of the Concrete Factory's methods return an abstract
    product, while inside the method a concrete product is instantiated.
    '''

    def create_product_a(self) -> AbstractProductA:
        return ConcreteProductA1()

    def create_product_b(self) -> AbstractProductB:
        return ConcreteProductB1()


class ConcreteFactory2(AbstractFactory):
    '''
    Each Concrete Factory has a corresponding product variant.
    '''

    def create_product_a(self) -> AbstractProductA:
        return ConcreteProductA2()

    def create_product_b(self) -> AbstractProductB:
        return ConcreteProductB2()


class AbstractProductA(ABC):
    '''
    Each distinct product of a product family should have a base interface. All
    variants of the product must implement this interface.
    '''

    @abstractmethod
    def useful_function_a(self) -> str:
        pass


'''
Concrete Products are created by corresponding Concrete Factories.
'''


class ConcreteProductA1(AbstractProductA):
    def useful_function_a(self) -> str:
        return "The result of the product A1."


class ConcreteProductA2(AbstractProductA):
    def useful_function_a(self) -> str:
        return "The result of the product A2."


class AbstractProductB(ABC):
    '''
    Here's the the base interface of another product. All products can interact
    with each other, but proper interaction is possible only between products of
    the same concrete variant.
    '''
    @abstractmethod
    def useful_function_b(self) -> None:
        '''
        Product B is able to do its own thing...
        '''
        pass

    @abstractmethod
    def another_useful_function_b(self, collaborator: AbstractProductA) -> None:
        '''
        ...but it also can collaborate with the ProductA.

        The Abstract Factory makes sure that all products it creates are of the
        same variant and thus, compatible.
        '''
        pass


'''
Concrete Products are created by corresponding Concrete Factories.
'''


class ConcreteProductB1(AbstractProductB):
    def useful_function_b(self) -> str:
        return "The result of the product B1."

    '''
    The variant, Product B1, is only able to work correctly with the variant,
    Product A1. Nevertheless, it accepts any instance of AbstractProductA as an
    argument.
    '''

    def another_useful_function_b(self, collaborator: AbstractProductA) -> str:
        result = collaborator.useful_function_a()
        return f"The result of the B1 collaborating with the ({result})"


class ConcreteProductB2(AbstractProductB):
    def useful_function_b(self) -> str:
        return "The result of the product B2."

    def another_useful_function_b(self, collaborator: AbstractProductA):
        '''
        The variant, Product B2, is only able to work correctly with the
        variant, Product A2. Nevertheless, it accepts any instance of
        AbstractProductA as an argument.
        '''
        result = collaborator.useful_function_a()
        return f"The result of the B2 collaborating with the ({result})"


def client_code(factory: AbstractFactory) -> None:
    '''
    The client code works with factories and products only through abstract
    types: AbstractFactory and AbstractProduct. This lets you pass any factory
    or product subclass to the client code without breaking it.
    '''
    product_a = factory.create_product_a()
    product_b = factory.create_product_b()

    print(f"{product_b.useful_function_b()}")
    print(f"{product_b.another_useful_function_b(product_a)}", end="")


if __name__ == "__main__":
    '''
    The client code can work with any concrete factory class.
    '''
    print("Client: Testing client code with the first factory type:")
    client_code(ConcreteFactory1())

    print("\n")

    print("Client: Testing the same client code with the second factory type:")
    client_code(ConcreteFactory2())
```


---

### **Factory Method vs Abstract Factory**

- Factory Method
    - Intent: Defines an interface for creating an object, but lets subclasses decide which class to instantiate.
    - Focus: A single product.
    - Usage: Used when a class cannot anticipate the type of object it needs to create, or when the responsibility of instantiation should be delegated to subclasses.
    - Example: A Dialog class defines a createButton() method. Subclasses like WindowsDialog or WebDialog implement it to return WindowsButton or HTMLButton.

- Abstract Factory
    - Intent: Provides an interface for creating families of related or dependent objects without specifying their concrete classes.
    - Focus: A group of related products.
    - Usage: Used when the system needs to be independent of how its products are created and when it should work with multiple families of products.
    - Example: A GUIFactory can create a Button and a Checkbox. Depending on the factory (WindowsFactory or MacFactory), it produces Windows-style or Mac-style components.

### **Builder**



Builder is a creational design pattern, which allows constructing complex objects step by step. Unlike other creational patterns, Builder doesn’t require products to have a common interface. That makes it possible to produce different products using the same construction process. It’s especially useful when you need to create an object with lots of possible configuration options.

```
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any


class Builder(ABC):
    '''
    The Builder interface specifies methods for creating the different parts of
    the Product objects.
    '''

    @property
    @abstractmethod
    def product(self) -> None:
        pass

    @abstractmethod
    def produce_part_a(self) -> None:
        pass

    @abstractmethod
    def produce_part_b(self) -> None:
        pass

    @abstractmethod
    def produce_part_c(self) -> None:
        pass


class ConcreteBuilder1(Builder):
    '''
    The Concrete Builder classes follow the Builder interface and provide
    specific implementations of the building steps. Your program may have
    several variations of Builders, implemented differently.
    '''

    def __init__(self) -> None:
        '''
        A fresh builder instance should contain a blank product object, which is
        used in further assembly.
        '''
        self.reset()

    def reset(self) -> None:
        self._product = Product1()

    @property
    def product(self) -> Product1:
        '''
        Concrete Builders are supposed to provide their own methods for
        retrieving results. That's because various types of builders may create
        entirely different products that don't follow the same interface.
        Therefore, such methods cannot be declared in the base Builder interface
        (at least in a statically typed programming language).

        Usually, after returning the end result to the client, a builder
        instance is expected to be ready to start producing another product.
        That's why it's a usual practice to call the reset method at the end of
        the `getProduct` method body. However, this behavior is not mandatory,
        and you can make your builders wait for an explicit reset call from the
        client code before disposing of the previous result.
        '''
        product = self._product
        self.reset()
        return product

    def produce_part_a(self) -> None:
        self._product.add("PartA1")

    def produce_part_b(self) -> None:
        self._product.add("PartB1")

    def produce_part_c(self) -> None:
        self._product.add("PartC1")


class Product1():
    '''
    It makes sense to use the Builder pattern only when your products are quite
    complex and require extensive configuration.

    Unlike in other creational patterns, different concrete builders can produce
    unrelated products. In other words, results of various builders may not
    always follow the same interface.
    '''

    def __init__(self) -> None:
        self.parts = []

    def add(self, part: Any) -> None:
        self.parts.append(part)

    def list_parts(self) -> None:
        print(f"Product parts: {', '.join(self.parts)}", end="")


class Director:
    '''
    The Director is only responsible for executing the building steps in a
    particular sequence. It is helpful when producing products according to a
    specific order or configuration. Strictly speaking, the Director class is
    optional, since the client can control builders directly.
    '''

    def __init__(self) -> None:
        self._builder = None

    @property
    def builder(self) -> Builder:
        return self._builder

    @builder.setter
    def builder(self, builder: Builder) -> None:
        '''
        The Director works with any builder instance that the client code passes
        to it. This way, the client code may alter the final type of the newly
        assembled product.
        '''
        self._builder = builder

    '''
    The Director can construct several product variations using the same
    building steps.
    '''

    def build_minimal_viable_product(self) -> None:
        self.builder.produce_part_a()

    def build_full_featured_product(self) -> None:
        self.builder.produce_part_a()
        self.builder.produce_part_b()
        self.builder.produce_part_c()


if __name__ == "__main__":
    '''
    The client code creates a builder object, passes it to the director and then
    initiates the construction process. The end result is retrieved from the
    builder object.
    '''

    director = Director()
    builder = ConcreteBuilder1()
    director.builder = builder

    print("Standard basic product: ")
    director.build_minimal_viable_product()
    builder.product.list_parts()

    print("\n")

    print("Standard full featured product: ")
    director.build_full_featured_product()
    builder.product.list_parts()

    print("\n")

    # Remember, the Builder pattern can be used without a Director class.
    print("Custom product: ")
    builder.produce_part_a()
    builder.produce_part_b()
    builder.product.list_parts()
```

---

### **Prototype**

Prototype is a creational design pattern that allows cloning objects, even complex ones, without coupling to their specific classes. All prototype classes should have a common interface that makes it possible to copy objects even if their concrete classes are unknown. Prototype objects can produce full copies since objects of the same class can access each other’s private fields. The Prototype pattern is available in Python out of the box with a copy module.

```
import copy


class SelfReferencingEntity:
    def __init__(self):
        self.parent = None

    def set_parent(self, parent):
        self.parent = parent


class SomeComponent:
    '''
    Python provides its own interface of Prototype via `copy.copy` and
    `copy.deepcopy` functions. And any class that wants to implement custom
    implementations have to override `__copy__` and `__deepcopy__` member
    functions.
    '''

    def __init__(self, some_int, some_list_of_objects, some_circular_ref):
        self.some_int = some_int
        self.some_list_of_objects = some_list_of_objects
        self.some_circular_ref = some_circular_ref

    def __copy__(self):
        '''
        Create a shallow copy. This method will be called whenever someone calls
        `copy.copy` with this object and the returned value is returned as the
        new shallow copy.
        '''

        # First, let's create copies of the nested objects.
        some_list_of_objects = copy.copy(self.some_list_of_objects)
        some_circular_ref = copy.copy(self.some_circular_ref)

        # Then, let's clone the object itself, using the prepared clones of the
        # nested objects.
        new = self.__class__(
            self.some_int, some_list_of_objects, some_circular_ref
        )
        new.__dict__.update(self.__dict__)

        return new

    def __deepcopy__(self, memo=None):
        '''
        Create a deep copy. This method will be called whenever someone calls
        `copy.deepcopy` with this object and the returned value is returned as
        the new deep copy.

        What is the use of the argument `memo`? Memo is the dictionary that is
        used by the `deepcopy` library to prevent infinite recursive copies in
        instances of circular references. Pass it to all the `deepcopy` calls
        you make in the `__deepcopy__` implementation to prevent infinite
        recursions.
        '''
        if memo is None:
            memo = {}

        # First, let's create copies of the nested objects.
        some_list_of_objects = copy.deepcopy(self.some_list_of_objects, memo)
        some_circular_ref = copy.deepcopy(self.some_circular_ref, memo)

        # Then, let's clone the object itself, using the prepared clones of the
        # nested objects.
        new = self.__class__(
            self.some_int, some_list_of_objects, some_circular_ref
        )
        new.__dict__ = copy.deepcopy(self.__dict__, memo)

        return new


if __name__ == "__main__":

    list_of_objects = [1, {1, 2, 3}, [1, 2, 3]]
    circular_ref = SelfReferencingEntity()
    component = SomeComponent(23, list_of_objects, circular_ref)
    circular_ref.set_parent(component)

    shallow_copied_component = copy.copy(component)

    # Let's change the list in shallow_copied_component and see if it changes in
    # component.
    shallow_copied_component.some_list_of_objects.append("another object")
    if component.some_list_of_objects[-1] == "another object":
        print(
            "Adding elements to `shallow_copied_component`'s "
            "some_list_of_objects adds it to `component`'s "
            "some_list_of_objects."
        )
    else:
        print(
            "Adding elements to `shallow_copied_component`'s "
            "some_list_of_objects doesn't add it to `component`'s "
            "some_list_of_objects."
        )

    # Let's change the set in the list of objects.
    component.some_list_of_objects[1].add(4)
    if 4 in shallow_copied_component.some_list_of_objects[1]:
        print(
            "Changing objects in the `component`'s some_list_of_objects "
            "changes that object in `shallow_copied_component`'s "
            "some_list_of_objects."
        )
    else:
        print(
            "Changing objects in the `component`'s some_list_of_objects "
            "doesn't change that object in `shallow_copied_component`'s "
            "some_list_of_objects."
        )

    deep_copied_component = copy.deepcopy(component)

    # Let's change the list in deep_copied_component and see if it changes in
    # component.
    deep_copied_component.some_list_of_objects.append("one more object")
    if component.some_list_of_objects[-1] == "one more object":
        print(
            "Adding elements to `deep_copied_component`'s "
            "some_list_of_objects adds it to `component`'s "
            "some_list_of_objects."
        )
    else:
        print(
            "Adding elements to `deep_copied_component`'s "
            "some_list_of_objects doesn't add it to `component`'s "
            "some_list_of_objects."
        )

    # Let's change the set in the list of objects.
    component.some_list_of_objects[1].add(10)
    if 10 in deep_copied_component.some_list_of_objects[1]:
        print(
            "Changing objects in the `component`'s some_list_of_objects "
            "changes that object in `deep_copied_component`'s "
            "some_list_of_objects."
        )
    else:
        print(
            "Changing objects in the `component`'s some_list_of_objects "
            "doesn't change that object in `deep_copied_component`'s "
            "some_list_of_objects."
        )

    print(
        f"id(deep_copied_component.some_circular_ref.parent): "
        f"{id(deep_copied_component.some_circular_ref.parent)}"
    )
    print(
        f"id(deep_copied_component.some_circular_ref.parent.some_circular_ref.parent): "
        f"{id(deep_copied_component.some_circular_ref.parent.some_circular_ref.parent)}"
    )
    print(
        "^^ This shows that deepcopied objects contain same reference, they "
        "are not cloned repeatedly."
    )
```

---

## **Structural Patterns (how classes and objects are organized)**

### **Adapter**



Adapter is a structural design pattern, which allows incompatible objects to collaborate. The Adapter acts as a wrapper between two objects. It catches calls for one object and transforms them to format and interface recognizable by the second object. It’s very often used in systems based on some legacy code. In such cases, Adapters make legacy code work with modern classes.

```
class Target:
    '''
    The Target defines the domain-specific interface used by the client code.
    '''

    def request(self) -> str:
        return "Target: The default target's behavior."


class Adaptee:
    '''
    The Adaptee contains some useful behavior, but its interface is incompatible
    with the existing client code. The Adaptee needs some adaptation before the
    client code can use it.
    '''

    def specific_request(self) -> str:
        return ".eetpadA eht fo roivaheb laicepS"


class Adapter(Target, Adaptee):
    '''
    The Adapter makes the Adaptee's interface compatible with the Target's
    interface via multiple inheritance.
    '''

    def request(self) -> str:
        return f"Adapter: (TRANSLATED) {self.specific_request()[::-1]}"


def client_code(target: "Target") -> None:
    '''
    The client code supports all classes that follow the Target interface.
    '''

    print(target.request(), end="")


if __name__ == "__main__":
    print("Client: I can work just fine with the Target objects:")
    target = Target()
    client_code(target)
    print("\n")

    adaptee = Adaptee()
    print("Client: The Adaptee class has a weird interface. "
          "See, I don't understand it:")
    print(f"Adaptee: {adaptee.specific_request()}", end="\n\n")

    print("Client: But I can work with it via the Adapter:")
    adapter = Adapter()
    client_code(adapter)
```

---

### **Decorator**



Decorator is a structural pattern that allows adding new behaviors to objects dynamically by placing them inside special wrapper objects, called decorators.Using decorators you can wrap objects countless number of times since both target objects and decorators follow the same interface. The resulting object will get a stacking behavior of all wrappers.

```
class Component():
    '''
    The base Component interface defines operations that can be altered by
    decorators.
    '''

    def operation(self) -> str:
        pass


class ConcreteComponent(Component):
    '''
    Concrete Components provide default implementations of the operations. There
    might be several variations of these classes.
    '''

    def operation(self) -> str:
        return "ConcreteComponent"


class Decorator(Component):
    '''
    The base Decorator class follows the same interface as the other components.
    The primary purpose of this class is to define the wrapping interface for
    all concrete decorators. The default implementation of the wrapping code
    might include a field for storing a wrapped component and the means to
    initialize it.
    '''

    _component: Component = None

    def __init__(self, component: Component) -> None:
        self._component = component

    @property
    def component(self) -> Component:
        '''
        The Decorator delegates all work to the wrapped component.
        '''

        return self._component

    def operation(self) -> str:
        return self._component.operation()


class ConcreteDecoratorA(Decorator):
    '''
    Concrete Decorators call the wrapped object and alter its result in some
    way.
    '''

    def operation(self) -> str:
        '''
        Decorators may call parent implementation of the operation, instead of
        calling the wrapped object directly. This approach simplifies extension
        of decorator classes.
        '''
        return f"ConcreteDecoratorA({self.component.operation()})"


class ConcreteDecoratorB(Decorator):
    '''
    Decorators can execute their behavior either before or after the call to a
    wrapped object.
    '''

    def operation(self) -> str:
        return f"ConcreteDecoratorB({self.component.operation()})"


def client_code(component: Component) -> None:
    '''
    The client code works with all objects using the Component interface. This
    way it can stay independent of the concrete classes of components it works
    with.
    '''

    # ...

    print(f"RESULT: {component.operation()}", end="")

    # ...


if __name__ == "__main__":
    # This way the client code can support both simple components...
    simple = ConcreteComponent()
    print("Client: I've got a simple component:")
    client_code(simple)
    print("\n")

    # ...as well as decorated ones.
    #
    # Note how decorators can wrap not only simple components but the other
    # decorators as well.
    decorator1 = ConcreteDecoratorA(simple)
    decorator2 = ConcreteDecoratorB(decorator1)
    print("Client: Now I've got a decorated component:")
    client_code(decorator2)
```

---

### **Facade**



Facade is a structural design pattern that provides a simplified (but limited) interface to a complex system of classes, library or framework. While Facade decreases the overall complexity of the application, it also helps to move unwanted dependencies to one place. It’s especially handy when working with complex libraries and APIs.

```
from __future__ import annotations


class Facade:
    '''
    The Facade class provides a simple interface to the complex logic of one or
    several subsystems. The Facade delegates the client requests to the
    appropriate objects within the subsystem. The Facade is also responsible for
    managing their lifecycle. All of this shields the client from the undesired
    complexity of the subsystem.
    '''

    def __init__(self, subsystem1: Subsystem1, subsystem2: Subsystem2) -> None:
        '''
        Depending on your application's needs, you can provide the Facade with
        existing subsystem objects or force the Facade to create them on its
        own.
        '''

        self._subsystem1 = subsystem1 or Subsystem1()
        self._subsystem2 = subsystem2 or Subsystem2()

    def operation(self) -> str:
        '''
        The Facade's methods are convenient shortcuts to the sophisticated
        functionality of the subsystems. However, clients get only to a fraction
        of a subsystem's capabilities.
        '''

        results = []
        results.append("Facade initializes subsystems:")
        results.append(self._subsystem1.operation1())
        results.append(self._subsystem2.operation1())
        results.append("Facade orders subsystems to perform the action:")
        results.append(self._subsystem1.operation_n())
        results.append(self._subsystem2.operation_z())
        return "\n".join(results)


class Subsystem1:
    '''
    The Subsystem can accept requests either from the facade or client directly.
    In any case, to the Subsystem, the Facade is yet another client, and it's
    not a part of the Subsystem.
    '''

    def operation1(self) -> str:
        return "Subsystem1: Ready!"

    # ...

    def operation_n(self) -> str:
        return "Subsystem1: Go!"


class Subsystem2:
    '''
    Some facades can work with multiple subsystems at the same time.
    '''

    def operation1(self) -> str:
        return "Subsystem2: Get ready!"

    # ...

    def operation_z(self) -> str:
        return "Subsystem2: Fire!"


def client_code(facade: Facade) -> None:
    '''
    The client code works with complex subsystems through a simple interface
    provided by the Facade. When a facade manages the lifecycle of the
    subsystem, the client might not even know about the existence of the
    subsystem. This approach lets you keep the complexity under control.
    '''

    print(facade.operation(), end="")


if __name__ == "__main__":
    # The client code may have some of the subsystem's objects already created.
    # In this case, it might be worthwhile to initialize the Facade with these
    # objects instead of letting the Facade create new instances.
    subsystem1 = Subsystem1()
    subsystem2 = Subsystem2()
    facade = Facade(subsystem1, subsystem2)
    client_code(facade)
```

---

### **Proxy**

Proxy is a structural design pattern that provides an object that acts as a substitute for a real service object used by a client. A proxy receives client requests, does some work (access control, caching, etc.) and then passes the request to a service object. The proxy object has the same interface as a service, which makes it interchangeable with a real object when passed to a client. It’s irreplaceable when you want to add some additional behaviors to an object of some existing class without changing the client code.

```
from abc import ABC, abstractmethod


class Subject(ABC):
    '''
    The Subject interface declares common operations for both RealSubject and
    the Proxy. As long as the client works with RealSubject using this
    interface, you'll be able to pass it a proxy instead of a real subject.
    '''

    @abstractmethod
    def request(self) -> None:
        pass


class RealSubject(Subject):
    '''
    The RealSubject contains some core business logic. Usually, RealSubjects are
    capable of doing some useful work which may also be very slow or sensitive -
    e.g. correcting input data. A Proxy can solve these issues without any
    changes to the RealSubject's code.
    '''

    def request(self) -> None:
        print("RealSubject: Handling request.")


class Proxy(Subject):
    '''
    The Proxy has an interface identical to the RealSubject.
    '''

    def __init__(self, real_subject: RealSubject) -> None:
        self._real_subject = real_subject

    def request(self) -> None:
        '''
        The most common applications of the Proxy pattern are lazy loading,
        caching, controlling the access, logging, etc. A Proxy can perform one
        of these things and then, depending on the result, pass the execution to
        the same method in a linked RealSubject object.
        '''

        if self.check_access():
            self._real_subject.request()
            self.log_access()

    def check_access(self) -> bool:
        print("Proxy: Checking access prior to firing a real request.")
        return True

    def log_access(self) -> None:
        print("Proxy: Logging the time of request.", end="")


def client_code(subject: Subject) -> None:
    '''
    The client code is supposed to work with all objects (both subjects and
    proxies) via the Subject interface in order to support both real subjects
    and proxies. In real life, however, clients mostly work with their real
    subjects directly. In this case, to implement the pattern more easily, you
    can extend your proxy from the real subject's class.
    '''

    # ...

    subject.request()

    # ...


if __name__ == "__main__":
    print("Client: Executing the client code with a real subject:")
    real_subject = RealSubject()
    client_code(real_subject)

    print("")

    print("Client: Executing the same client code with a proxy:")
    proxy = Proxy(real_subject)
    client_code(proxy)

```


---

### **Composite**

Composite is a structural design pattern that lets you compose objects into tree structures and then work with these structures as if they were individual objects. Composite's great feature is the ability to run methods recursively over the whole tree structure and sum up the results. It's often used to represent hierarchies of user interface components or the code that works with graphs.

```
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import List


class Component(ABC):
    '''
    The base Component class declares common operations for both simple and
    complex objects of a composition.
    '''

    @property
    def parent(self) -> Component:
        return self._parent

    @parent.setter
    def parent(self, parent: Component):
        '''
        Optionally, the base Component can declare an interface for setting and
        accessing a parent of the component in a tree structure. It can also
        provide some default implementation for these methods.
        '''

        self._parent = parent

    '''
    In some cases, it would be beneficial to define the child-management
    operations right in the base Component class. This way, you won't need to
    expose any concrete component classes to the client code, even during the
    object tree assembly. The downside is that these methods will be empty for
    the leaf-level components.
    '''

    def add(self, component: Component) -> None:
        pass

    def remove(self, component: Component) -> None:
        pass

    def is_composite(self) -> bool:
        '''
        You can provide a method that lets the client code figure out whether a
        component can bear children.
        '''

        return False

    @abstractmethod
    def operation(self) -> str:
        '''
        The base Component may implement some default behavior or leave it to
        concrete classes (by declaring the method containing the behavior as
        "abstract").
        '''

        pass


class Leaf(Component):
    '''
    The Leaf class represents the end objects of a composition. A leaf can't
    have any children.

    Usually, it's the Leaf objects that do the actual work, whereas Composite
    objects only delegate to their sub-components.
    '''

    def operation(self) -> str:
        return "Leaf"


class Composite(Component):
    '''
    The Composite class represents the complex components that may have
    children. Usually, the Composite objects delegate the actual work to their
    children and then "sum-up" the result.
    '''

    def __init__(self) -> None:
        self._children: List[Component] = []

    '''
    A composite object can add or remove other components (both simple or
    complex) to or from its child list.
    '''

    def add(self, component: Component) -> None:
        self._children.append(component)
        component.parent = self

    def remove(self, component: Component) -> None:
        self._children.remove(component)
        component.parent = None

    def is_composite(self) -> bool:
        return True

    def operation(self) -> str:
        '''
        The Composite executes its primary logic in a particular way. It
        traverses recursively through all its children, collecting and summing
        their results. Since the composite's children pass these calls to their
        children and so forth, the whole object tree is traversed as a result.
        '''

        results = []
        for child in self._children:
            results.append(child.operation())
        return f"Branch({'+'.join(results)})"


def client_code(component: Component) -> None:
    '''
    The client code works with all of the components via the base interface.
    '''

    print(f"RESULT: {component.operation()}", end="")


def client_code2(component1: Component, component2: Component) -> None:
    '''
    Thanks to the fact that the child-management operations are declared in the
    base Component class, the client code can work with any component, simple or
    complex, without depending on their concrete classes.
    '''

    if component1.is_composite():
        component1.add(component2)

    print(f"RESULT: {component1.operation()}", end="")


if __name__ == "__main__":
    # This way the client code can support the simple leaf components...
    simple = Leaf()
    print("Client: I've got a simple component:")
    client_code(simple)
    print("\n")

    # ...as well as the complex composites.
    tree = Composite()

    branch1 = Composite()
    branch1.add(Leaf())
    branch1.add(Leaf())

    branch2 = Composite()
    branch2.add(Leaf())

    tree.add(branch1)
    tree.add(branch2)

    print("Client: Now I've got a composite tree:")
    client_code(tree)
    print("\n")

    print("Client: I don't need to check the components classes even when managing the tree:")
    client_code2(tree, simple)
```

---

## **Behavioral Patterns (how objects interact and communicate)**

### **Observer**


Observer is a behavioral design pattern that allows some objects to notify other objects about changes in their state. It provides a way to subscribe and unsubscribe to and from these events for any object that implements a subscriber interface.

```
from __future__ import annotations
from abc import ABC, abstractmethod
from random import randrange
from typing import List


class Subject(ABC):
    '''
    The Subject interface declares a set of methods for managing subscribers.
    '''

    @abstractmethod
    def attach(self, observer: Observer) -> None:
        '''
        Attach an observer to the subject.
        '''
        pass

    @abstractmethod
    def detach(self, observer: Observer) -> None:
        '''
        Detach an observer from the subject.
        '''
        pass

    @abstractmethod
    def notify(self) -> None:
        '''
        Notify all observers about an event.
        '''
        pass


class ConcreteSubject(Subject):
    '''
    The Subject owns some important state and notifies observers when the state
    changes.
    '''

    _state: int = None
    '''
    For the sake of simplicity, the Subject's state, essential to all
    subscribers, is stored in this variable.
    '''

    _observers: List[Observer] = []
    '''
    List of subscribers. In real life, the list of subscribers can be stored
    more comprehensively (categorized by event type, etc.).
    '''

    def attach(self, observer: Observer) -> None:
        print("Subject: Attached an observer.")
        self._observers.append(observer)

    def detach(self, observer: Observer) -> None:
        self._observers.remove(observer)

    '''
    The subscription management methods.
    '''

    def notify(self) -> None:
        '''
        Trigger an update in each subscriber.
        '''

        print("Subject: Notifying observers...")
        for observer in self._observers:
            observer.update(self)

    def some_business_logic(self) -> None:
        '''
        Usually, the subscription logic is only a fraction of what a Subject can
        really do. Subjects commonly hold some important business logic, that
        triggers a notification method whenever something important is about to
        happen (or after it).
        '''

        print("\nSubject: I'm doing something important.")
        self._state = randrange(0, 10)

        print(f"Subject: My state has just changed to: {self._state}")
        self.notify()


class Observer(ABC):
    '''
    The Observer interface declares the update method, used by subjects.
    '''

    @abstractmethod
    def update(self, subject: Subject) -> None:
        '''
        Receive update from subject.
        '''
        pass


'''
Concrete Observers react to the updates issued by the Subject they had been
attached to.
'''


class ConcreteObserverA(Observer):
    def update(self, subject: Subject) -> None:
        if subject._state < 3:
            print("ConcreteObserverA: Reacted to the event")


class ConcreteObserverB(Observer):
    def update(self, subject: Subject) -> None:
        if subject._state == 0 or subject._state >= 2:
            print("ConcreteObserverB: Reacted to the event")


if __name__ == "__main__":
    # The client code.

    subject = ConcreteSubject()

    observer_a = ConcreteObserverA()
    subject.attach(observer_a)

    observer_b = ConcreteObserverB()
    subject.attach(observer_b)

    subject.some_business_logic()
    subject.some_business_logic()

    subject.detach(observer_a)

    subject.some_business_logic()
```


---

### **Strategy**



Strategy is a behavioral design pattern that turns a set of behaviors into objects and makes them interchangeable inside original context object. The original object, called context, holds a reference to a strategy object. The context delegates executing the behavior to the linked strategy object. In order to change the way the context performs its work, other objects may replace the currently linked strategy object with another one. This pattern is often used in various frameworks to provide users a way to change the behavior of a class without extending it.

```
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import List


class Context():
    '''
    The Context defines the interface of interest to clients.
    '''

    def __init__(self, strategy: Strategy) -> None:
        '''
        Usually, the Context accepts a strategy through the constructor, but
        also provides a setter to change it at runtime.
        '''

        self._strategy = strategy

    @property
    def strategy(self) -> Strategy:
        '''
        The Context maintains a reference to one of the Strategy objects. The
        Context does not know the concrete class of a strategy. It should work
        with all strategies via the Strategy interface.
        '''

        return self._strategy

    @strategy.setter
    def strategy(self, strategy: Strategy) -> None:
        '''
        Usually, the Context allows replacing a Strategy object at runtime.
        '''

        self._strategy = strategy

    def do_some_business_logic(self) -> None:
        '''
        The Context delegates some work to the Strategy object instead of
        implementing multiple versions of the algorithm on its own.
        '''

        # ...

        print("Context: Sorting data using the strategy (not sure how it'll do it)")
        result = self._strategy.do_algorithm(["a", "b", "c", "d", "e"])
        print(",".join(result))

        # ...


class Strategy(ABC):
    '''
    The Strategy interface declares operations common to all supported versions
    of some algorithm.

    The Context uses this interface to call the algorithm defined by Concrete
    Strategies.
    '''

    @abstractmethod
    def do_algorithm(self, data: List):
        pass


'''
Concrete Strategies implement the algorithm while following the base Strategy
interface. The interface makes them interchangeable in the Context.
'''


class ConcreteStrategyA(Strategy):
    def do_algorithm(self, data: List) -> List:
        return sorted(data)


class ConcreteStrategyB(Strategy):
    def do_algorithm(self, data: List) -> List:
        return reversed(sorted(data))


if __name__ == "__main__":
    # The client code picks a concrete strategy and passes it to the context.
    # The client should be aware of the differences between strategies in order
    # to make the right choice.

    context = Context(ConcreteStrategyA())
    print("Client: Strategy is set to normal sorting.")
    context.do_some_business_logic()
    print()

    print("Client: Strategy is set to reverse sorting.")
    context.strategy = ConcreteStrategyB()
    context.do_some_business_logic()
```

---

### **Command**



Command is behavioral design pattern that converts requests or simple operations into objects. The conversion allows deferred or remote execution of commands, storing command history, etc. Most often it's used as an alternative for callbacks to parameterizing UI elements with actions. It's also used for queueing tasks, tracking operations history, etc.

```
from __future__ import annotations
from abc import ABC, abstractmethod


class Command(ABC):
    '''
    The Command interface declares a method for executing a command.
    '''

    @abstractmethod
    def execute(self) -> None:
        pass


class SimpleCommand(Command):
    '''
    Some commands can implement simple operations on their own.
    '''

    def __init__(self, payload: str) -> None:
        self._payload = payload

    def execute(self) -> None:
        print(f"SimpleCommand: See, I can do simple things like printing"
              f"({self._payload})")


class ComplexCommand(Command):
    '''
    However, some commands can delegate more complex operations to other
    objects, called "receivers."
    '''

    def __init__(self, receiver: Receiver, a: str, b: str) -> None:
        '''
        Complex commands can accept one or several receiver objects along with
        any context data via the constructor.
        '''

        self._receiver = receiver
        self._a = a
        self._b = b

    def execute(self) -> None:
        '''
        Commands can delegate to any methods of a receiver.
        '''

        print("ComplexCommand: Complex stuff should be done by a receiver object", end="")
        self._receiver.do_something(self._a)
        self._receiver.do_something_else(self._b)


class Receiver:
    '''
    The Receiver classes contain some important business logic. They know how to
    perform all kinds of operations, associated with carrying out a request. In
    fact, any class may serve as a Receiver.
    '''

    def do_something(self, a: str) -> None:
        print(f"\nReceiver: Working on ({a}.)", end="")

    def do_something_else(self, b: str) -> None:
        print(f"\nReceiver: Also working on ({b}.)", end="")


class Invoker:
    '''
    The Invoker is associated with one or several commands. It sends a request
    to the command.
    '''

    _on_start = None
    _on_finish = None

    '''
    Initialize commands.
    '''

    def set_on_start(self, command: Command):
        self._on_start = command

    def set_on_finish(self, command: Command):
        self._on_finish = command

    def do_something_important(self) -> None:
        '''
        The Invoker does not depend on concrete command or receiver classes. The
        Invoker passes a request to a receiver indirectly, by executing a
        command.
        '''

        print("Invoker: Does anybody want something done before I begin?")
        if isinstance(self._on_start, Command):
            self._on_start.execute()

        print("Invoker: ...doing something really important...")

        print("Invoker: Does anybody want something done after I finish?")
        if isinstance(self._on_finish, Command):
            self._on_finish.execute()


if __name__ == "__main__":
    '''
    The client code can parameterize an invoker with any commands.
    '''

    invoker = Invoker()
    invoker.set_on_start(SimpleCommand("Say Hi!"))
    receiver = Receiver()
    invoker.set_on_finish(ComplexCommand(
        receiver, "Send email", "Save report"))

    invoker.do_something_important()
```

---

### **Template Method**

Template Method is a behavioral design pattern that allows you to define a skeleton of an algorithm in a base class and let subclasses override the steps without changing the overall algorithm's structure. Developers often use it to provide framework users with a simple means of extending standard functionality using inheritance.

```
from abc import ABC, abstractmethod


class AbstractClass(ABC):
    '''
    The Abstract Class defines a template method that contains a skeleton of
    some algorithm, composed of calls to (usually) abstract primitive
    operations.

    Concrete subclasses should implement these operations, but leave the
    template method itself intact.
    '''

    def template_method(self) -> None:
        '''
        The template method defines the skeleton of an algorithm.
        '''

        self.base_operation1()
        self.required_operations1()
        self.base_operation2()
        self.hook1()
        self.required_operations2()
        self.base_operation3()
        self.hook2()

    # These operations already have implementations.

    def base_operation1(self) -> None:
        print("AbstractClass says: I am doing the bulk of the work")

    def base_operation2(self) -> None:
        print("AbstractClass says: But I let subclasses override some operations")

    def base_operation3(self) -> None:
        print("AbstractClass says: But I am doing the bulk of the work anyway")

    # These operations have to be implemented in subclasses.

    @abstractmethod
    def required_operations1(self) -> None:
        pass

    @abstractmethod
    def required_operations2(self) -> None:
        pass

    # These are "hooks." Subclasses may override them, but it's not mandatory
    # since the hooks already have default (but empty) implementation. Hooks
    # provide additional extension points in some crucial places of the
    # algorithm.

    def hook1(self) -> None:
        pass

    def hook2(self) -> None:
        pass


class ConcreteClass1(AbstractClass):
    '''
    Concrete classes have to implement all abstract operations of the base
    class. They can also override some operations with a default implementation.
    '''

    def required_operations1(self) -> None:
        print("ConcreteClass1 says: Implemented Operation1")

    def required_operations2(self) -> None:
        print("ConcreteClass1 says: Implemented Operation2")


class ConcreteClass2(AbstractClass):
    '''
    Usually, concrete classes override only a fraction of base class'
    operations.
    '''

    def required_operations1(self) -> None:
        print("ConcreteClass2 says: Implemented Operation1")

    def required_operations2(self) -> None:
        print("ConcreteClass2 says: Implemented Operation2")

    def hook1(self) -> None:
        print("ConcreteClass2 says: Overridden Hook1")


def client_code(abstract_class: AbstractClass) -> None:
    '''
    The client code calls the template method to execute the algorithm. Client
    code does not have to know the concrete class of an object it works with, as
    long as it works with objects through the interface of their base class.
    '''

    # ...
    abstract_class.template_method()
    # ...


if __name__ == "__main__":
    print("Same client code can work with different subclasses:")
    client_code(ConcreteClass1())
    print("")

    print("Same client code can work with different subclasses:")
    client_code(ConcreteClass2())
```

---

### **Chain of Responsibility**



Chain of Responsibility is behavioral design pattern that allows passing request along the chain of potential handlers until one of them handles request.
The pattern allows multiple objects to handle the request without coupling sender class to the concrete classes of the receivers. The chain can be composed dynamically at runtime with any handler that follows a standard handler interface. It's mostly relevant when your code operates with chains of objects, such as filters, event chains, etc.

```
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Optional


class Handler(ABC):
    '''
    The Handler interface declares a method for building the chain of handlers.
    It also declares a method for executing a request.
    '''

    @abstractmethod
    def set_next(self, handler: Handler) -> Handler:
        pass

    @abstractmethod
    def handle(self, request) -> Optional[str]:
        pass


class AbstractHandler(Handler):
    '''
    The default chaining behavior can be implemented inside a base handler
    class.
    '''

    _next_handler: Handler = None

    def set_next(self, handler: Handler) -> Handler:
        self._next_handler = handler
        # Returning a handler from here will let us link handlers in a
        # convenient way like this:
        # monkey.set_next(squirrel).set_next(dog)
        return handler

    @abstractmethod
    def handle(self, request: Any) -> str:
        if self._next_handler:
            return self._next_handler.handle(request)

        return None


'''
All Concrete Handlers either handle a request or pass it to the next handler in
the chain.
'''


class MonkeyHandler(AbstractHandler):
    def handle(self, request: Any) -> str:
        if request == "Banana":
            return f"Monkey: I'll eat the {request}"
        else:
            return super().handle(request)


class SquirrelHandler(AbstractHandler):
    def handle(self, request: Any) -> str:
        if request == "Nut":
            return f"Squirrel: I'll eat the {request}"
        else:
            return super().handle(request)


class DogHandler(AbstractHandler):
    def handle(self, request: Any) -> str:
        if request == "MeatBall":
            return f"Dog: I'll eat the {request}"
        else:
            return super().handle(request)


def client_code(handler: Handler) -> None:
    '''
    The client code is usually suited to work with a single handler. In most
    cases, it is not even aware that the handler is part of a chain.
    '''

    for food in ["Nut", "Banana", "Cup of coffee"]:
        print(f"\nClient: Who wants a {food}?")
        result = handler.handle(food)
        if result:
            print(f"  {result}", end="")
        else:
            print(f"  {food} was left untouched.", end="")


if __name__ == "__main__":
    monkey = MonkeyHandler()
    squirrel = SquirrelHandler()
    dog = DogHandler()

    monkey.set_next(squirrel).set_next(dog)

    # The client should be able to send a request to any handler, not just the
    # first one in the chain.
    print("Chain: Monkey > Squirrel > Dog")
    client_code(monkey)
    print("\n")

    print("Subchain: Squirrel > Dog")
    client_code(squirrel)
```

---

### **Mediator**



Mediator is a behavioral design pattern that reduces coupling between components of a program by making them communicate indirectly, through a special mediator object. The Mediator makes it easy to modify, extend and reuse individual components because they're no longer dependent on the dozens of other classes. The most popular usage of the Mediator pattern in Python code is facilitating communications between GUI components of an app. The synonym of the Mediator is the Controller part of MVC pattern.

```
from __future__ import annotations
from abc import ABC


class Mediator(ABC):
    '''
    The Mediator interface declares a method used by components to notify the
    mediator about various events. The Mediator may react to these events and
    pass the execution to other components.
    '''

    def notify(self, sender: object, event: str) -> None:
        pass


class ConcreteMediator(Mediator):
    def __init__(self, component1: Component1, component2: Component2) -> None:
        self._component1 = component1
        self._component1.mediator = self
        self._component2 = component2
        self._component2.mediator = self

    def notify(self, sender: object, event: str) -> None:
        if event == "A":
            print("Mediator reacts on A and triggers following operations:")
            self._component2.do_c()
        elif event == "D":
            print("Mediator reacts on D and triggers following operations:")
            self._component1.do_b()
            self._component2.do_c()


class BaseComponent:
    '''
    The Base Component provides the basic functionality of storing a mediator's
    instance inside component objects.
    '''

    def __init__(self, mediator: Mediator = None) -> None:
        self._mediator = mediator

    @property
    def mediator(self) -> Mediator:
        return self._mediator

    @mediator.setter
    def mediator(self, mediator: Mediator) -> None:
        self._mediator = mediator


'''
Concrete Components implement various functionality. They don't depend on other
components. They also don't depend on any concrete mediator classes.
'''


class Component1(BaseComponent):
    def do_a(self) -> None:
        print("Component 1 does A.")
        self.mediator.notify(self, "A")

    def do_b(self) -> None:
        print("Component 1 does B.")
        self.mediator.notify(self, "B")


class Component2(BaseComponent):
    def do_c(self) -> None:
        print("Component 2 does C.")
        self.mediator.notify(self, "C")

    def do_d(self) -> None:
        print("Component 2 does D.")
        self.mediator.notify(self, "D")


if __name__ == "__main__":
    # The client code.
    c1 = Component1()
    c2 = Component2()
    mediator = ConcreteMediator(c1, c2)

    print("Client triggers operation A.")
    c1.do_a()

    print("\n", end="")

    print("Client triggers operation D.")
    c2.do_d()
```


---

### **State**



State is a behavioral design pattern that allows an object to change the behavior when its internal state changes. The pattern extracts state-related behaviors into separate state classes and forces the original object to delegate the work to an instance of these classes, instead of acting on its own. The State pattern is commonly used in Python to convert massive switch-base state machines into objects.

```
from __future__ import annotations
from abc import ABC, abstractmethod


class Context:
    '''
    The Context defines the interface of interest to clients. It also maintains
    a reference to an instance of a State subclass, which represents the current
    state of the Context.
    '''

    _state = None
    '''
    A reference to the current state of the Context.
    '''

    def __init__(self, state: State) -> None:
        self.transition_to(state)

    def transition_to(self, state: State):
        '''
        The Context allows changing the State object at runtime.
        '''

        print(f"Context: Transition to {type(state).__name__}")
        self._state = state
        self._state.context = self

    '''
    The Context delegates part of its behavior to the current State object.
    '''

    def request1(self):
        self._state.handle1()

    def request2(self):
        self._state.handle2()


class State(ABC):
    '''
    The base State class declares methods that all Concrete State should
    implement and also provides a backreference to the Context object,
    associated with the State. This backreference can be used by States to
    transition the Context to another State.
    '''

    @property
    def context(self) -> Context:
        return self._context

    @context.setter
    def context(self, context: Context) -> None:
        self._context = context

    @abstractmethod
    def handle1(self) -> None:
        pass

    @abstractmethod
    def handle2(self) -> None:
        pass


'''
Concrete States implement various behaviors, associated with a state of the
Context.
'''


class ConcreteStateA(State):
    def handle1(self) -> None:
        print("ConcreteStateA handles request1.")
        print("ConcreteStateA wants to change the state of the context.")
        self.context.transition_to(ConcreteStateB())

    def handle2(self) -> None:
        print("ConcreteStateA handles request2.")


class ConcreteStateB(State):
    def handle1(self) -> None:
        print("ConcreteStateB handles request1.")

    def handle2(self) -> None:
        print("ConcreteStateB handles request2.")
        print("ConcreteStateB wants to change the state of the context.")
        self.context.transition_to(ConcreteStateA())


if __name__ == "__main__":
    # The client code.

    context = Context(ConcreteStateA())
    context.request1()
    context.request2()
```

---

## **Pythonic / Idiomatic Patterns**

### **Iterator**

An iterator is an object in Python that allows you to traverse through all elements of a collection (like lists, tuples, strings) one at a time. Not to be confused with iterable objects. For example, lists, tuples, dictionaries, and sets are all iterable objects. They are iterable containers which you can get an iterator from.

Iterators implement two methods:
- `__iter__()`: Returns the iterator object itself.
- `__next__()`: Returns the next element. Raises StopIteration when no more elements are available.

```
# A list (iterable object)
numbers = [1, 2, 3]

# Convert list to iterator
it = iter(numbers)

print(next(it))  # 1
print(next(it))  # 2
print(next(it))  # 3
# print(next(it))  # Raises StopIteration
```

```
# Custom iterable class
class Countdown:
    def __init__(self, start):
        self.start = start
    
    def __iter__(self):
        return self
    
    def __next__(self):
        if self.start <= 0:
            raise StopIteration
        value = self.start
        self.start -= 1
        return value

cd = Countdown(5)
for num in cd:
    print(num)
# Output: 5 4 3 2 1
```

---

### **Generator**

A generator is a simpler way to create an iterator in Python using the `yield` keyword. Unlike functions with return, a generator function remembers its state between calls. They are lazy, meaning that they produce values one at a time and only when requested, which is memory efficient.

```
def my_generator():
    yield 1
    yield 2
    yield 3

gen = my_generator()

print(next(gen))  # 1
print(next(gen))  # 2
print(next(gen))  # 3
# print(next(gen))  # Raises StopIteration
```

```
def countdown(n):
    while n > 0:
        yield n
        n -= 1

for num in countdown(5):
    print(num)
# Output: 5 4 3 2 1
```

```
# Generator Expressions (similar to list comprehensions but generate items lazily (one at a time).)
squares = (x*x for x in range(5))  # generator expression

for sq in squares:
    print(sq)
# Output: 0 1 4 9 16
```

### **Iterators vs Generators**

| Feature               | Iterator (Class-based)            | Generator (`yield` or expression) |
|------------------------|----------------------------------|-----------------------------------|
| Implementation         | Requires `__iter__()` + `__next__()` | Uses `yield`, simpler             |
| State Management       | Must be tracked manually         | Handled automatically             |
| Memory Usage           | Can be large (stores whole data) | Lazy, produces one value at a time |
| Use Case               | Complex iteration logic          | Simpler sequences, streaming data |

### **Context Manager (with)**
A context manager is an object that defines a runtime context to be entered and exited when using the with statement. It sets up some resource and cleans it up automatically when done. No matter what happens inside (even if there's an error), the cleanup is guaranteed. This is most commonly used for things like file handling, database connections, locks, or network sessions.

Context manager implements two methods:

- `__enter__(self)`: Called at the start of the with block.
- `__exit__(self, exc_type, exc_value, traceback)`: Called when the block is exited.

```
# Traditional way
f = open("example.txt", "w")
try:
    f.write("Hello, world!")
finally:
    f.close()  # must remember to close manually

# With context manager
with open("example.txt", "w") as f:
    f.write("Hello, world!")  # auto-closes file
```

```
# Custom context manager
class MyContext:
    def __enter__(self):
        print("Entering context...")
        return "Resource Ready"

    def __exit__(self, exc_type, exc_value, traceback):
        print("Exiting context...")
        if exc_type:
            print(f"An exception occurred: {exc_value}")
        return True  # suppress exception if True

with MyContext() as resource:
    print(resource)  # Resource Ready
    # raise ValueError("Oops!")  # try uncommenting
```

```
from contextlib import contextmanager

# Using @contextmanager decorator
@contextmanager
def my_context():
    print("Entering context...")
    yield "Resource Ready"
    print("Exiting context...")

with my_context() as resource:
    print(resource)
```

```
import time
from contextlib import contextmanager

# Elapsed time context manager example
@contextmanager
def timer():
    start = time.time()
    yield
    end = time.time()
    print(f"Elapsed: {end - start:.4f} seconds")

with timer():
    sum([i**2 for i in range(1_000_000)])
```

---

### **Data Class (@dataclass)**

A dataclass is a Python class specifically designed to store data with less boilerplate code. Introduced in Python 3.7 (dataclasses module), they automatically generate:

- `__init__()`: Constructor
- `__repr__()`: Nice string representation
- `__eq__()`: Equality comparison

They're great when you want classes that mainly hold values (like records, DTOs, configs, etc.).

```
from dataclasses import dataclass

@dataclass
class Person:
    name: str
    age: int

p = Person("Alice", 30)
print(p)  # Person(name='Alice', age=30)
```

```
@dataclass
class Point:
    x: int
    y: int

p1 = Point(1, 2)
p2 = Point(1, 2)
print(p1 == p2)  # True
```

```
# Frozen dataclass (immutable)
@dataclass(frozen=True)
class Color:
    red: int
    green: int
    blue: int

c = Color(255, 0, 0)
# c.red = 100  # ❌ Error: Frozen dataclass is immutable
```

```
# Ordered dataclass
@dataclass(order=True)
class Student:
    grade: int
    name: str

s1 = Student(90, "Alice")
s2 = Student(85, "Bob")
print(s1 > s2)  # True (compares by grade first, then name)
```

```
# Post-init
@dataclass
class Rectangle:
    width: int
    height: int
    area: int = 0

    def __post_init__(self):
        self.area = self.width * self.height

r = Rectangle(5, 10)
print(r.area)  # 50
```

```
from dataclasses import dataclass, field

# Default factory
@dataclass
class Classroom:
    students: list[str] = field(default_factory=list)

c1 = Classroom()
c2 = Classroom()
c1.students.append("Alice")
print(c2.students)  # [] (different lists, not shared!)
```

---

### **Mixin**
A mixin is a type of class that provides additional functionality to another class through multiple inheritance. Mixins are not meant to stand alone. They are small, reusable building blocks. They are used to "mix in" extra behavior into other classes. Think of them as traits or capabilities you can add to classes without creating deep inheritance hierarchies.

- A mixin does not define the main purpose of a class.
- Usually provides helper methods or extra features.
- Works best with multiple inheritance.
- Names often end with "Mixin" (by convention).

```
class WalkMixin:
    def walk(self):
        return "Walking..."

class TalkMixin:
    def talk(self):
        return "Talking..."

class Person(WalkMixin, TalkMixin):
    def __init__(self, name):
        self.name = name

p = Person("Alice")
print(p.walk())  # Walking...
print(p.talk())  # Talking...
```

```
import datetime

class LoggingMixin:
    def log(self, message):
        print(f"[{datetime.datetime.now()}] {message}")

class Worker(LoggingMixin):
    def work(self):
        self.log("Work started")
        # ... do some work ...
        self.log("Work finished")

w = Worker()
w.work()
```

```
class JsonMixin:
    def to_json(self):
        import json
        return json.dumps(self.__dict__)

class Person(JsonMixin):
    def __init__(self, name, age):
        self.name = name
        self.age = age

p = Person("Bob", 25)
print(p.to_json())  # {"name": "Bob", "age": 25}
```

---

### **Dependency Injection (DI)**
Dependency Injection (DI) is a design pattern where a class or function does not create its own dependencies. Instead, those dependencies are "injected" (passed in) from the outside. This makes code:

- Loosely coupled (easy to replace components).
- Easier to test (you can inject mocks).
- More flexible (dependencies can be swapped at runtime).

```
class EmailService:
    def send_message(self, message):
        print(f"Sending email: {message}")

class SMSService:
    def send_message(self, message):
        print(f"Sending SMS: {message}")

class UserController:
    def __init__(self, notifier):  # dependency is injected
        self.notifier = notifier

    def register_user(self, username):
        print(f"Registering {username}")
        self.notifier.send_message("Welcome!")

email_service = EmailService()
sms_service = SMSService()

controller1 = UserController(email_service)
controller2 = UserController(sms_service)

controller1.register_user("Alice")  # uses email
controller2.register_user("Bob")    # uses SMS
```

```
def process_data(data, storage_backend):
    # storage_backend is injected
    storage_backend.store(data)

class FileStorage:
    def store(self, data):
        print(f"Storing in file: {data}")

class DatabaseStorage:
    def store(self, data):
        print(f"Storing in database: {data}")

# Inject different backends
process_data("Hello", FileStorage())
process_data("Hello", DatabaseStorage())
```

```
# Using dependency_injector library
from dependency_injector import containers, providers

class Service:
    def process(self):
        return "Processing..."

class Controller:
    def __init__(self, service: Service):
        self.service = service
    
    def handle(self):
        print(self.service.process())

class Container(containers.DeclarativeContainer):
    service = providers.Factory(Service)
    controller = providers.Factory(Controller, service=service)

# Usage
container = Container()
controller = container.controller()
controller.handle()  # Processing...
```

---

### **Repository Pattern**

The Repository Pattern is a design pattern that abstracts the logic of data access. It sits between the domain/business logic and the data layer (database, API, files, etc.). It provides a clean API for accessing data without exposing how it’s stored or retrieved. Think of it as a middleman between your app and the database. Reasons to use it:

- Separation of concerns: Business logic doesn’t care about database details.
- Easier testing: Replace repository with an in-memory fake or mock.
- Flexibility: Swap database engines without rewriting business logic.

```
import sqlite3

# Domain Model
class User:
    def __init__(self, user_id, name):
        self.id = user_id
        self.name = name

# Repository Interface
class UserRepository:
    def get_by_id(self, user_id):
        raise NotImplementedError

# Concrete Implementation (SQLite)
class SQLiteUserRepository(UserRepository):
    def __init__(self, db_path):
        self.db_path = db_path

    def get_by_id(self, user_id):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT id, name FROM users WHERE id=?", (user_id,))
        row = cursor.fetchone()
        conn.close()
        if row:
            return User(row[0], row[1])
        return None

# Business Logic
class UserService:
    def __init__(self, user_repository: UserRepository):
        self.user_repository = user_repository

    def get_user_name(self, user_id):
        user = self.user_repository.get_by_id(user_id)
        return user.name if user else None

# Usage
repo = SQLiteUserRepository("app.db")
service = UserService(repo)
print(service.get_user_name(1))
```

```
# In-Memory Repository (for testing)
class InMemoryUserRepository(UserRepository):
    def __init__(self):
        self.users = {
            1: User(1, "Alice"),
            2: User(2, "Bob")
        }

    def get_by_id(self, user_id):
        return self.users.get(user_id)

# Usage in tests
repo = InMemoryUserRepository()
service = UserService(repo)
print(service.get_user_name(2))  # Bob
```

---
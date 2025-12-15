# Web Apps

## Sync servers vs Async servers

## Websockets

## Request life cycle

When a client sends a request to a web server and receives a response, a sophisticated orchestration of components works together behind the scenes. Understanding this lifecycle is crucial for web developers to optimize performance, debug issues, and design scalable architectures.

### The Journey Overview

The request lifecycle involves several distinct layers of infrastructure, each handling specific responsibilities. At the front sits the web server, which manages incoming connections and protocol handling. Behind it, an application server hosts the actual business logic written in frameworks like Django, Flask, or FastAPI. Between these components, various protocols and middleware components facilitate communication and add cross-cutting functionality.

### 1. Web Server Layer

The process begins when a web server process monitors specific network ports for incoming HTTP requests. Popular web servers include **Nginx**, **Apache HTTP Server**, and **Caddy**. These servers excel at handling thousands of concurrent connections efficiently, managing SSL/TLS termination, serving static files, and implementing load balancing strategies.

When a request arrives, the web server performs initial protocol parsing, validates the HTTP format, handles connection keep-alive semantics, and manages SSL certificate validation for HTTPS connections. Modern web servers like Nginx use event-driven architectures that can handle tens of thousands of concurrent connections without creating individual threads for each request.

The web server also implements critical security features such as rate limiting, request size restrictions, and basic filtering of malicious requests before they reach the application layer. Additionally, it often serves static assets (CSS, JavaScript, images) directly from the filesystem without involving the application server, significantly improving performance for these resources.

### 2. Application Gateway Interface

Once the web server processes the initial request, it needs to communicate with the application server. This communication happens through standardized protocols that abstract the differences between various web servers and application frameworks.

For traditional synchronous Python applications, the **WSGI (Web Server Gateway Interface)** protocol serves as the bridge. Popular web servers that include WSGI include **Gunicorn** and **uWSGI**. These servers spawn multiple worker processes, each capable of handling one request at a time, providing isolation and fault tolerance.

For asynchronous applications, **ASGI (Asynchronous Server Gateway Interface)** enables handling of long-lived connections and concurrent request processing within a single worker. **Uvicorn** are prominent ASGI servers that support modern async frameworks like FastAPI, and async-enabled Django.

### Application Server Processing

The application server hosts the actual business logic written using web frameworks such as **Django**, **Flask** or **FastAPI**. This layer contains the custom code that developers write to handle specific business requirements.

The request first encounters the framework's routing system, which matches the incoming URL pattern to specific handler functions or class methods. Modern frameworks use efficient routing algorithms, often implementing trie data structures or compiled regular expressions for fast lookup performance.

Before reaching the business logic, requests typically pass through a middleware pipeline. Middleware components handle cross-cutting concerns such as authentication, authorization, logging, request parsing, CORS headers, and security validations. Each middleware can modify the request, add contextual information, or short-circuit the pipeline if certain conditions aren't met.

The core business logic then processes the request, potentially involving database queries, external API calls, file system operations, or complex computations. This is where the application-specific functionality lives—user authentication, data processing, business rule enforcement, and response generation.

### Response Journey

After the application generates a response, it travels back through the same pipeline in reverse. The framework serializes the response data (JSON for APIs, HTML for web pages), sets appropriate headers including caching directives and security headers, and passes the response back through the middleware stack.

The application server sends the response through the gateway interface (WSGI/ASGI) back to the web server, which handles final compression (gzip, Brotli), adds server-specific headers, and transmits the data back to the client over the established TCP connection.

### Abstraction Benefits

This layered architecture provides clear separation of concerns. Web servers handle low-level networking, protocol compliance, and infrastructure concerns that are common across all web applications. Application servers focus on request routing, middleware processing, and providing convenient APIs for developers. The gateway interfaces ensure portability, allowing applications to run on different server combinations without code changes.

Developers can concentrate solely on business logic implementation while the infrastructure handles connection management, protocol parsing, SSL termination, static file serving, and other operational concerns. This separation enables better scalability, as each layer can be optimized and scaled independently based on specific performance requirements.

### Request Lifecycle Diagram

```
Client Request
     ↓
┌─────────────────┐
│   Web Server    │ ← Nginx, Apache, Caddy
│ - SSL/TLS       │
│ - Load Balance  │
│ - Static Files  │
│ - Rate Limiting │
└─────────────────┘
     ↓ HTTP
┌─────────────────┐
│ Gateway Interface│ ← WSGI (Gunicorn, uWSGI)
│ (WSGI/ASGI)     │   ASGI (Uvicorn, Daphne)
└─────────────────┘
     ↓ Protocol
┌─────────────────┐
│ Application     │ ← Django, Flask, FastAPI
│ Framework       │
│ - Routing       │
│ - Middleware    │
│ - Business Logic│
└─────────────────┘
     ↓ Response
    Client
```

This architecture ensures that each component operates within its area of expertise, creating a robust, scalable, and maintainable web application infrastructure.

## Authorization

## wsgi vs asgi

## Websockets

## Idempotency

Se tiene una API de procesamiento de pagos:
1. User places order
2. Payments API invoked
3. API triggers Lambda function
4. Writing in DB and charge customer via Stripe

Que pasa si el frontend esta laggy y el boton para pagar se apreta dos veces? Se le cobraria dos veces al user -> MAL

Para solucionar esto, estas APIs tienen que ser Idempotentes. Que sea idempotente significa que el efecto pretendido de un metodo en cualquier request identica es igual. Para implementar la idempotencia en este caso:

1. Cuando el user genera la request se crea una idempotency key.que se pasará a lo largo de las APIs (esta key la recibe la Lambda y la pasa a la DB y a Stripe tambien)
2. En la DB se verifica si ya existe una key igual y en caso de que exista retorna sin hacer el cargo en Stripe

## Authentication

1. Basic Authorization
Por cada request se incluye un header "Authorization Basic <base64(username:password)>" (una version base64 encoded de las credenciales del user). Estos datos los usa el backend para validar contra los registrados en la DB. El problema es que se están compartiendo credenciales sin encriptar en cada request

2. Session Auth
Se hace una request con username y password que se usan para validar del lado del backend y se les atribuye una SessionID via cookie que se guarda en memoria del lado del servidor. Para el resto de las requests del cliente se incluye ese SessionID que se puede valir contra lo que tiene en memoria el servidor. De esta manera no se envia el username y la password todo el tiempo. El problema es que esta solucion no funciona en un ambiente distribuido al menos que se use un estado compartido usando algo como Redis.

3. JWT Auth
El user hace un login request y el servidor usa una Secret Key para generar un JWT (Json Web Token) que se usa para validar cada request. De esta manera no se depende del guardado de la SessionID en memoria y cualquier servidor que tenga acceso a la Secret Key puede validar a ese usuario

4. OAuth



## Debug a slow API

1. Hay que definir que es lento: P99 o promedio, 5seg o 500ms. Esto ayuda a entender donde optimizar.
2. Chequear la red: Mucha carga puede empeorar la performance. Si este es el problema se puede implementar pagination, usar un CDN o usar caching en el navegador.
3. Si la latencia de la red no es el problema: se puede activar el query logging para inspeccionar a fondo la DB. Si las queries tardan mucho puede tratarse de un problema de N+1, puede ser que la query no sea optima y revise una tabla completa inecesariamente o que no se use indexado. Para resolver estos problemas se puede agregar indexado donde sea necesario, utilizar caching.
4. Si el problema no está en la DB: Puede tratarse del codigo corriendo en el backend, por ejemplo algoritmos lentos, cómputos pesados o código bloqueante que podría ser async. 
5. Si lo que es lentos son llamadas a servicios externos: Pueden hacerse las llamadas en paralelo en lugar de una a una, agregar caching si las respuestas no cambian frecuentemente y tener timeouts.
6. Para hacer más rápido el debugging en el futuro: Montar un dashboard con métricas de interés a monitorear y alarmas asociadas a estas.

## Event Driven Architecture

- En lugar de servicios llamándose entre sí via API por cada Request, un Producer crea un evento. Los consumers deciden a cuáles eventos se suscriben. 
- Los eventos son usualmente pequeños y contienen información crítica para que los consumers actúen. (usuaalmente type, timestamp y un pequeño payload con información). En ocaciones, los consumers pueden usar estos datos para obtener más información en la DB.
- Usualmente los eventos los maneja un Broker (RabbitMQ, Amazon SNS, Kafka). Estos brokers reciben los eventos del producer y los distribuyen a los consumers. Tipicamente pueden manejar cosas como los reintentos automáticos.
- Todo lo que no sea necesario hacer en el momento puede ser usado con eventos, por ejemplo, envío de emails, logging analytics, actualizar search indexes, limpiar datos viejos, generar recomendaciones, etc.
- La desventaja de los sistemas event-driven es que son mucho mas dificiles de manejar y debugear. Las fallas pueden estar en el producer, consumer, dentro de broker, en los retires o en mensajes trabados en una dead-letter queue. Se requiere distributed tracing, correlation IDs y buena disciplina de loggeo. Si no hay una buena observabilidad, la arquitectura event-driven puede volverse una caja negra.

## Port forwarding, IPv4 vs IPv6

## Design an online multiplayer Chess matchmaking system

## Design an online game Leaderbord using Shards with Redis


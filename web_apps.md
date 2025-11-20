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
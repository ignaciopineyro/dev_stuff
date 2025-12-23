## Event Driven Architecture

- En lugar de servicios llamándose entre sí via API por cada Request, un Producer crea un evento. Los consumers deciden a cuáles eventos se suscriben.
- Los eventos son usualmente pequeños y contienen información crítica para que los consumers actúen. (usuaalmente type, timestamp y un pequeño payload con información). En ocaciones, los consumers pueden usar estos datos para obtener más información en la DB.
- Usualmente los eventos los maneja un Broker (RabbitMQ, Amazon SNS, Kafka). Estos brokers reciben los eventos del producer y los distribuyen a los consumers. Tipicamente pueden manejar cosas como los reintentos automáticos.
- Todo lo que no sea necesario hacer en el momento puede ser usado con eventos, por ejemplo, envío de emails, logging analytics, actualizar search indexes, limpiar datos viejos, generar recomendaciones, etc.
- La desventaja de los sistemas event-driven es que son mucho mas dificiles de manejar y debugear. Las fallas pueden estar en el producer, consumer, dentro de broker, en los retires o en mensajes trabados en una dead-letter queue. Se requiere distributed tracing, correlation IDs y buena disciplina de loggeo. Si no hay una buena observabilidad, la arquitectura event-driven puede volverse una caja negra.

## Hexagonal Architecture

## Microservices

## Design an online multiplayer Chess matchmaking system

## Design an online game Leaderbord using Shards with Redis

## Design a REST API for an e-commerce platform that handles inventory management and order processing in a high-concurrency environment


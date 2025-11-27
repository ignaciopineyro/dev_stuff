# AI/ML

## Embeddings
que son, como funcionan, que propiedades tienen, como se entrenan

Primer layer de la red neuronal que disminuye la dimension del vector de entrada one-hot-encodeado. Es decir, entra un vector de 1xN donde N es la cantidad de, por ejemplo, palabras totales en el vocabulario. Es un vector de ceros con un 1 en el numero de etiqueta de cada palabra, por ejemplo, si la palabra auto tiene etiqueta "3", el vector con one-hot-encoding seria [0, 0, 0, 1, 0, 0, ...]. Como ese vector es inmenso, el trabajo del Embedding es reducir su dimensionalidad.

Es mas optimo que, por ejemplo, para una red que va a clasificar imagenes de estudios medicos, se utiliza una red ya pre-entranada en clasificacion de imagenes. Esto se denomina transfer-learning. Por el mismo motivo, si se intenta hacer una clasificacion de textos legales, lo mejor seria contar con un embedding pre-entrenado en clasificacion de lenguaje generico, por ejemplo, word2vec.

## Transformers

Los Transformers son una arquitectura de redes neuronales específicamente diseñada para procesar secuencias (texto, audio, etc.) superando las limitaciones de modelos anteriores como RNN y LSTM. Son la base de modelos modernos como BERT, GPT, T5, etc.

Voy a explicarlos en orden: primero qué problema resuelven, luego cómo funcionan internamente y finalmente por qué son tan buenos.

1) El problema previo

Antes de Transformers, el NLP se hacía con modelos secuenciales:

RNN / LSTM / GRU leen una frase palabra por palabra.

Tienen dos problemas grandes:

Son lentos porque son secuenciales.

No entienden bien dependencias lejanas:

Ejemplo:

The book that I bought yesterday was …

Para LSTM, conectar “book” con “was” es difícil si hay muchas palabras en el medio → se pierde la memoria.

2) El salto de los Transformers

La idea clave del paper de 2017:

Attention is all you need

Los transformers no procesan el texto palabra por palabra.
Procesan todo en paralelo.

Eso desbloquea:

velocidad enorme,

la capacidad de usar GPUs eficientemente,

mejor contexto.

3) La idea central: Attention
Atención (attention)

Es un mecanismo que responde:

¿Qué palabras del input son importantes para la palabra que estoy calculando?

Ejemplo:
"El gato se subió al árbol porque estaba asustado."

¿Quién estaba asustado?

Una red normal no sabe.
Un transformer calcula qué palabra se relaciona con cuál:

“asustado” atiende más a “gato”

“árbol” atiende más a “subió”

Cada palabra genera puntajes de relevancia hacia las otras.

Si lo representás en una matriz, luce así:
```
 gato   0.8
 árbol  0.1
 se     0.0
 subió  0.5
 ...

```
4) Multi-Head Attention

No usamos una sola atención.
Usamos muchas en paralelo.

¿Por qué?

Cada atención descubre algo distinto:

una analiza relaciones gramaticales,

otra semántica,

otra coreferencia,

otra sintaxis…

Esto se llama multi-head mapping.

5) El truco matemático clave

Cada palabra se transforma en tres vectores:

Q: Query

K: Key

V: Value

La atención se computa como:
`Attention = softmax(Q * K^T) * V`

Lo importante no es memorizar la fórmula, sino la idea:

Q busca información

K dice qué información tiene cada otra palabra

V transporta el contenido final

6) Paralelismo

La arquitectura permite procesar toda la oración de una vez.

Esto elimina el mayor cuello de bote de RNN:
ya no hay que leer el texto paso a paso.

Por eso transformers escalan.
Por eso GPT-4/GPT-5 pueden entrenarse con trillones de tokens.

7) Encoder, Decoder y Encoder-Decoder

Hay tres variantes:

Encoder (ej: BERT)

Extrae significado del texto.

Decoder (ej: GPT)

Genera texto.

Encoder-Decoder (ej: T5)

Usado para traducción, resumen, QA.

8) ¿Qué aprendió el transformer?

No aprende reglas gramaticales explícitas.
Aprende un mapa semántico del lenguaje:

quién hace qué,

qué palabra refiere a cuál,

cuál es el sentido contextual,

qué es probable y qué no.

Por eso son modelos contextuales.
La palabra bank tendrá embeddings diferentes según el contexto:

bank → banco financiero

bank → orilla del río

Los viejos embeddings no podían hacer eso.

9) ¿Por qué fue tan revolucionario?

Porque resolvió 6 cosas a la vez:

✔ paralelismo total

permite entrenar en GPU

✔ no depende del orden secuencial

capta dependencias lejanas

✔ gran capacidad de generalización

entrena con datasets gigantescos

✔ permite transfer learning

se entrenan una vez, se usan para todo

✔ mejor rendimiento con más datos

(no se estanca como LSTM)

✔ escalabilidad

más capas + más datos + más GPU → mejor modelo

10) Resumen en una frase

Un transformer es una red neuronal basada en atención que aprende relaciones entre todas las palabras del texto a la vez, sin procesarlas secuencialmente.

## Vector DB

## Retrieval-Augmented Generation (RAG)
que es, que problema resuelve, que limitaciones tiene. Que frameworks hay para aplicarlo (Langgraph, Langchain)

Retrieval-Augmented Generation (RAG) is an AI technique that combines external knowledge sources with a large language model (LLM) to produce more accurate and relevant responses. Before generating an answer, the LLM first retrieves information from a specified knowledge base, such as an internal company database or a specific dataset, and then uses both this retrieved information and its own training data to formulate the final response. This allows LLMs to provide context-specific, up-to-date, and reliable answers without the need for constant retraining

## RAG vs fine tunning LLMs

## MCP

## LLM Cache

## NLP evolution timeline (before and after ChatGPT)

## LLMs architecture

## LLM training stages (pre-training, RL post-training for alignment and RLVR)

## AI-Agents

## Test-time scaling vs data/size scaling

## Pytorch (training and inference)

## Hugging Face

## LangChain

## LLAMA
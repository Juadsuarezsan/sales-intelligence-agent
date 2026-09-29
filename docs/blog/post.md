# Construyendo un agente de investigación B2B con LangGraph y Claude: plan, ejecuta, reflexiona

*Borrador para Medium / Dev.to. Autor: Juan David Suárez Sánchez.*

Los equipos de ventas B2B pierden una fracción enorme de su semana investigando cuentas: quién es el CEO, si acaban de levantar una ronda, qué producto venden, si están contratando. Herramientas como Apollo, Clay u Outreach venden precisamente eso: convertir una lista de empresas en perfiles y emails listos para enviar. En este proyecto construí una versión abierta de ese motor, con un objetivo doble: que funcione de verdad con Claude y búsqueda web, y que sea **evaluable** contra un ground truth de 200 empresas de Y Combinator. Este artículo cuenta las decisiones, lo que se mide y lo que todavía no.

## El problema que un solo prompt no resuelve

La versión ingenua es un prompt: "dado el nombre de la empresa, escribe un email". Falla por tres razones. Primero, el modelo no sabe qué pasó con la empresa el mes pasado. Segundo, cuando no sabe, inventa: rondas de financiación, nombres de ejecutivos, cifras de empleados. Tercero, no hay forma de saber si el email es bueno sin leerlo uno a uno.

Añadir una búsqueda web arregla parte del primer problema, pero introduce otro: la primera búsqueda rara vez trae lo que hace falta. "Acme Corp" devuelve la página de inicio y dos artículos viejos; la ronda de financiación aparece sólo si buscas "Acme Corp Series B 2026". Un humano ajustaría la consulta. Un agente también puede hacerlo, si le das un bucle.

## Arquitectura: plan → execute → reflect

El sistema es un `StateGraph` de LangGraph con siete nodos:

1. **plan**: Claude propone entre tres y seis consultas concretas (financiación, liderazgo, producto, contratación, noticias). En rondas siguientes recibe la lista de temas que faltan y las consultas ya ejecutadas para no repetirlas.
2. **execute**: corre las consultas contra Tavily, consulta HackerNews (API de Algolia, gratuita) y, si conoce el dominio, descarga la portada, `/about` y `/careers` con `httpx` y las parsea con `selectolax`. Todo termina en una lista de `Evidence` con título, URL, texto, puntuación y fuente.
3. **reflect**: Claude decide si la evidencia es suficiente con una regla explícita (financiación o producto, y además un líder o una señal reciente), señala contradicciones entre fuentes y propone consultas de seguimiento. Si no es suficiente y quedan rondas y presupuesto de herramientas, el grafo vuelve a **plan**.
4. **build**: convierte la evidencia en un `CompanyProfile` validado con Pydantic: one-liner, industria, sede, headcount, financiación, personas clave, señales recientes, pain points y fuentes. La instrucción central es "nunca inventes datos que no estén en la evidencia; si no se sabe, null".
5. **extract_hooks**: elige el gancho: un pain point probable y la señal más reciente y específica, con prioridad financiación > lanzamiento > contratación > cambio de liderazgo > noticia.
6. **write_email**: redacta asunto, saludo, gancho, propuesta de valor y llamada a la acción en menos de 90 palabras, obligado a citar un hecho concreto del perfil.
7. **score**: un juez LLM califica el email con una rúbrica de tres criterios (personalización, exactitud, claridad del CTA) del 1 al 5.

Dos presupuestos acotan el bucle: `MAX_REFLECTION_ITERATIONS` (rondas extra de plan/execute) y `MAX_TOOL_CALLS_PER_COMPANY`. Con el primero en cero, el grafo salta la reflexión, lo que convierte el mismo código en el sistema de referencia "una búsqueda + email".

## Una lección barata sobre LangGraph

La primera versión del grafo tenía un nodo llamado `email` que escribía la clave `email` del estado. LangGraph 0.2.45 lo rechaza: `'email' is already being used as a state key`. El API no arrancaba y dos tests fallaban, aunque el mensaje del commit anterior decía "11 tests passing". Renombrar el nodo a `write_email` arregló todo, y la regla quedó documentada en el código: los nombres de nodo nunca coinciden con claves del estado. Es el tipo de bug que sólo aparece cuando fijas versiones y ejecutas la suite antes de escribir nada nuevo.

## Todo funciona sin llaves, y eso está etiquetado

Cada nodo que usa Claude depende de un `Protocol` con un solo método, `complete(system, user)`. En producción lo implementa el SDK oficial de Anthropic con modelo fijado por ID datado, timeout explícito y reintentos con `tenacity` sólo sobre errores transitorios. En tests lo implementa un fake que devuelve JSON guionado. Y cuando no hay llave, el nodo recibe `None` y usa un fallback determinista: consultas de plantilla, un reflector por palabras clave, un constructor de perfiles por expresiones regulares (montos, rondas, "CEO is …", "headquartered in …", vacantes), un mapa industria → pain point, un email de plantilla y un juez heurístico.

Esto tiene un valor práctico enorme: 105 tests corren en CI sin credenciales con 98 % de cobertura, la demo estática se genera con el pipeline real y `python -m eval.run` produce una corrida completa en segundos. Tiene también un riesgo: confundir la salida del fallback con la del sistema. Por eso cada respuesta lleva `mode` (`llm` o `fallback`), cada puntuación lleva `judge` (`llm` o `heuristic`) y la tabla de resultados marca cada celda medida con stubs como tal, con la frase explícita de que no representa la calidad del sistema.

## Datos: 200 empresas de YC como ground truth

El directorio público de Y Combinator, espejado en `yc-oss/api`, tiene 6 253 empresas con nombre, sitio web, industria, sede, tamaño de equipo y one-liner. El script de datos descarga el JSON, filtra las empresas activas con todos esos campos (4 106) y toma una muestra de 200 con semilla fija. El hash SHA-256 del crudo y del eval set queda en `data/MANIFEST.txt`. No hay nada sintético en el ground truth: son los campos que YC publica.

La exactitud se mide como precisión y recall por campo con reglas de match explícitas: contención para industria, ciudad para sede, ±25 % para headcount, similitud de Jaccard ≥ 0,3 para el one-liner. El sitio web no se puntúa porque el eval lo entrega como entrada.

## Baselines y ablation

El eval compara seis sistemas sobre las mismas 200 empresas: *template only* (sin investigación, email de plantilla), *single-search + email* (una búsqueda y directo al perfil y al email) y el agente con 0, 1, 2 y 3 rondas de reflexión. Para cada uno reporta precisión/recall, puntuación del juez, distribución de tool calls, latencia y costo por empresa. El costo se calcula con los tokens reales de cada llamada más $0.001 por búsqueda.

La corrida offline versionada en el repositorio muestra la mecánica: el agente hace 7 tool calls por empresa, el single-search hace 1, el template hace 0, y la exactitud contra YC es prácticamente cero porque los stubs devuelven la misma noticia de financiación para cualquier empresa. Es lo esperado y está escrito así. Las celdas de personalización, latencia y costo con el sistema real dicen literalmente "pendiente (requiere ANTHROPIC_API_KEY + TAVILY_API_KEY)". La pregunta interesante, en qué punto una ronda más de reflexión deja de mejorar la exactitud y sólo suma costo, se responde con la misma tabla cuando se ejecute con llaves.

## La rúbrica es código

"Califica del 1 al 10" no es una métrica. La rúbrica del juez tiene tres criterios numerados con anclas por nivel y ejemplos. Personalización: 1 es genérico, 2 menciona el nombre, 3 la industria, 4 un hecho verificable del perfil, 5 dos o más hechos hilados en una razón para escribir ahora. Exactitud: 1 contiene un dato inventado o contradicho, 5 cada afirmación es rastreable a un campo del perfil. Claridad del CTA: 1 no hay llamada a la acción, 5 hay una petición concreta con duración, plazo y alternativa fácil. El texto se inyecta literalmente en el prompt del juez y se copia en `RESULTS.md`, así que dos corridas son comparables.

## Observabilidad y seguridad

Cada investigación tiene un `trace_id` en un `ContextVar` que aparece en cada línea de log y vuelve en la cabecera `X-Trace-Id`. Un `UsageTracker` acumula tokens de entrada y salida, llamadas al modelo, búsquedas pagas y la latencia de cada nodo. Con `LANGCHAIN_TRACING_V2=true` LangGraph envía las trazas a LangSmith y el cliente de Anthropic se envuelve con el tracer de LangSmith.

El API tiene CORS restringido por variable de entorno (el comodín se descarta aunque se configure), rate limiting por IP con `slowapi`, validación Pydantic que devuelve 422 con detalle por campo y `gitleaks` en CI. La demo escapa todo el texto generado antes de insertarlo en el DOM.

## Lo que falta, con nombre y apellido

Sin llaves no hay números reales: la exactitud, la personalización juzgada por un modelo más fuerte, la latencia y el costo por empresa, la curva costo-calidad del ablation y el análisis de los diez peores casos quedan pendientes y así están documentados. Tampoco hay 30 trazas públicas en LangSmith ni un despliegue con p95 medido. El código para todo eso existe y está probado con mocks; lo que falta es ejecutar `python -m eval.run` y `scripts/batch_run.py --limit 100` con credenciales.

En trabajo futuro: búsquedas en paralelo dentro de cada ronda (hoy son secuenciales para mantener el orden de la evidencia), prompt caching en los system prompts largos, un caché de búsquedas por hash de consulta para no pagar dos veces la misma query entre eval, batch y API, y un fetcher con navegador para los sitios que sólo renderizan con JavaScript.

## Qué me llevo

Tres cosas. Que un bucle de reflexión acotado es la diferencia entre "buscar" e "investigar", pero sólo vale lo que el ablation demuestre. Que un fallback determinista por nodo es la mejor inversión en testabilidad que se puede hacer en un sistema de agentes, siempre que cada salida diga de dónde viene. Y que la parte más difícil de un proyecto así no es el prompt: es construir el ground truth, las reglas de match y la rúbrica antes de mirar un solo resultado.

El código está en https://github.com/Juadsuarezsan/sales-intelligence-agent bajo licencia MIT.

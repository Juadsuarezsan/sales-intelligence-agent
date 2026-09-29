# Arquitectura — B2B Sales Intelligence Agent

![Diagrama de arquitectura](architecture.svg)

## Resumen

Dado el nombre de una empresa (y opcionalmente su dominio), el sistema ejecuta un bucle **plan → execute → reflect** en LangGraph hasta reunir evidencia suficiente, construye un `CompanyProfile` validado con Pydantic, extrae un gancho de personalización, redacta un `OutreachEmail` y lo califica con un juez LLM sobre una rúbrica numerada. Todo el flujo funciona sin llaves de API gracias a fallbacks deterministas por nodo, lo que permite tests, CI, demo y una corrida de evaluación offline.

## Grafo (`src/agents/orchestrator.py`)

| Nodo | Entrada relevante | Salida | Implementación LLM | Fallback determinista |
|---|---|---|---|---|
| `plan` | `company_name`, `context`, `missing`, `executed_queries` | `planned_queries`, `iteration+1` | `ResearchPlanner` (JSON de 3-6 queries) | 5 plantillas de consulta; una por tema faltante en rondas siguientes |
| `execute` | `planned_queries`, `website` | `evidence`, `tool_calls`, `executed_queries` | — (herramientas) | — |
| `reflect` | `evidence` | `sufficient`, `missing`, `planned_queries` (follow-ups) | `Reflector` (regla de suficiencia + contradicciones) | Detección por palabras clave: funding/leader/product/signal |
| `build` | `evidence`, `website` | `profile` | `ProfileBuilder` → `CompanyProfile.model_validate` | Regex: montos y rondas, C-level, sede, headcount, open roles, one-liner del meta description |
| `extract_hooks` | `profile` | `hooks` | `PersonalizationExtractor` | Prioridad funding > launch > hiring > leadership > news + mapa industria → pain point |
| `write_email` | `profile`, `hooks` | `email` | `EmailWriter` | Plantilla con gancho y pain point; saludo al primer ejecutivo |
| `score` | `email`, `profile` | `scores` | `QualityScorer` con `RUBRIC` | Aproximación por conteo de hechos, claims no soportados y forma del CTA |

Los nombres de nodo (`build`, `extract_hooks`, `write_email`, `score`) nunca coinciden con claves del estado (`profile`, `hooks`, `email`, `scores`): LangGraph 0.2.45 rechaza esa colisión y ese fue el bug que rompía el API en la versión anterior.

### Rutas condicionales

- Tras `execute`: si `MAX_REFLECTION_ITERATIONS == 0` se salta `reflect` (ablation sin reflexión).
- Tras `reflect`: se va a `build` cuando la evidencia es suficiente, cuando se agotaron las rondas (`iteration - 1 >= MAX_REFLECTION_ITERATIONS`), cuando se agotó el presupuesto de herramientas (`MAX_TOOL_CALLS_PER_COMPANY`) o cuando no hay follow-ups; en cualquier otro caso vuelve a `plan` con los temas faltantes.

### Presupuestos

`iterations` cuenta rondas plan/execute (1 + rondas disparadas por reflexión). Cada búsqueda web, la consulta a HN y el fetch del sitio cuentan como un `tool_call`; el ejecutor deja de lanzar búsquedas cuando alcanza el tope.

## Herramientas (`src/tools/`)

| Tool | Producción | Offline | Timeout / reintentos |
|---|---|---|---|
| Web search | `TavilyWebSearch` (`search_depth="basic"`) | `StubWebSearch` (resultados enlatados por palabra clave) | `asyncio.wait_for(TAVILY_TIMEOUT_SECONDS)`, 3 intentos con backoff |
| Noticias | `HackerNewsSearch` (Algolia, sin llave) | `StubHackerNews` | httpx 8 s, 2 intentos sobre `TransportError` |
| Sitio web | `WebsiteFetcher`: `/`, `/about`, `/careers` con httpx + selectolax | `StubWebsiteFetcher` | 8 s por página, 1 reintento; 4xx/no-HTML se omiten |

Todas devuelven `Evidence` (`title`, `url`, `content`, `score`, `source`), la única forma de dato que ve el resto del grafo. El fetcher extrae título, meta description, `h1/h2`, texto visible truncado y un conteo de vacantes en `/careers`.

## Cliente LLM (`src/llm.py`)

`LLMClient` es un `Protocol` con un solo método `complete(system, user, max_tokens, temperature)`. `AnthropicLLM` usa el SDK oficial con modelo fijado (`claude-sonnet-4-5-20250929`), timeout explícito, `max_retries=0` en el SDK y reintentos propios con `tenacity` sobre errores transitorios (conexión, timeout, 429, 5xx). Registra tokens en el `UsageTracker` del contexto. `parse_json_object` limpia fences y prosa alrededor del JSON. Los nodos capturan sólo `LLMOutputError`, `ValidationError` y `APIError` para caer al fallback; cualquier otro error se propaga.

## Separación de capas

| Capa | Carpeta | Conoce a |
|---|---|---|
| Configuración | `src/config.py` | variables de entorno |
| Dominio | `src/agents/`, `src/llm.py`, `src/observability.py` | esquemas, protocolos |
| I/O | `src/tools/`, `src/storage/` | HTTP externo, PostgreSQL |
| Transporte HTTP | `src/api/` | dominio + repositorio |
| Evaluación | `eval/` | dominio + `data/eval` |
| Operación | `scripts/` | dominio + repositorio |

## Persistencia (`src/storage/`)

`ResearchRepository` (Protocol: `save`, `get`, `list_recent`, `close`) con `InMemoryRepository` (TTL configurable) y `PostgresRepository` (`psycopg_pool.AsyncConnectionPool`, tabla `company_research` con `payload JSONB`, upsert por `company_key`). El API usa el repositorio como caché de resultados por nombre de empresa.

## Observabilidad (`src/observability.py`)

- `trace_id` en un `ContextVar`, inyectado en cada línea de log y devuelto en la cabecera `X-Trace-Id`.
- `UsageTracker` acumula tokens de entrada/salida, llamadas LLM, búsquedas pagas y latencia por nodo; `cost_usd` = tokens × tabla de precios + búsquedas × `TAVILY_COST_PER_SEARCH_USD`.
- `log_node` decora cada nodo y registra estado de entrada (resumido), actualización de salida y latencia.
- LangSmith: con `LANGCHAIN_TRACING_V2=true` y `LANGCHAIN_API_KEY`, LangGraph envía las trazas del grafo y el cliente Anthropic se envuelve con `langsmith.wrappers.wrap_anthropic`.

## Evaluación (`eval/`)

`python -m eval.run` evalúa seis sistemas sobre las 200 empresas de `data/eval/`: `template_only`, `single_search_email` y el agente con 0/1/2/3 reflexiones. Calcula precisión/recall micro por campo (`industry`, `location`, `employees_est`, `one_liner`), distribución de tool calls, latencia, costo y las puntuaciones del juez, y escribe `eval/runs/<fecha>-<nombre>.json` + `eval/RESULTS.md`.

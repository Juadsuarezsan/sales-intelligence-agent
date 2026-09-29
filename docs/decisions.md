# Decisiones técnicas

Cada decisión registra la alternativa descartada y el criterio concreto.

## 1. LangGraph `StateGraph` en vez de un bucle `while` a mano o LangChain `AgentExecutor`

El flujo tiene ramas explícitas (saltar la reflexión con presupuesto 0, volver a planificar sólo si hay follow-ups y presupuesto) y necesita ser inspeccionable nodo por nodo para el ablation. `StateGraph` hace visibles las transiciones en código y permite decorar cada nodo con `log_node`. `AgentExecutor` esconde el bucle y decide por sí mismo cuándo llamar herramientas, lo que impide fijar el número de reflexiones. Un `while` manual habría funcionado, pero perdería el trazado nativo de LangSmith y la validación de nombres de nodo/estado (que precisamente detectó el bug `email`).

## 2. SDK oficial `anthropic` detrás de un `Protocol` en lugar de `langchain-anthropic`

El proyecto sólo necesita completions de un turno con salida JSON. Usar el SDK directo da acceso limpio a `usage.input_tokens/output_tokens` (necesarios para `cost_usd`), a los tipos de error para reintentar sólo los transitorios y a `timeout`/`max_retries` sin capas intermedias. El `Protocol LLMClient` permite inyectar un fake en tests y usar `None` como fallback offline; `langchain-anthropic` se eliminó de las dependencias. LangSmith sigue funcionando: el grafo se traza vía LangGraph y el cliente se envuelve con `wrap_anthropic` cuando el tracing está activo.

## 3. Modelo fijado por ID datado (`claude-sonnet-4-5-20250929`)

Un alias sin fecha cambia de comportamiento sin aviso y rompe la comparabilidad de las corridas de `eval/runs/`. El ID va en `config.py` y en `.env.example`; el juez tiene su propia variable `JUDGE_MODEL` porque el spec pide un modelo más fuerte como juez y así se cambia sin tocar código.

## 4. `httpx` + `selectolax` para el sitio web, no BeautifulSoup ni un navegador

Se leen tres páginas estáticas por empresa; `selectolax` (Modest/lexbor en C) parsea en milisegundos y expone `css()`/`decompose()` suficientes para quitar `script/style/nav/footer`. BeautifulSoup es 5-10× más lento y no aporta nada aquí. Un navegador (Playwright) resolvería sitios 100 % JS pero multiplica latencia y costo de infraestructura; queda como trabajo futuro para el subconjunto de sitios que devuelven HTML vacío.

## 5. Fallback determinista por nodo en vez de fallar sin llave

Cada nodo LLM tiene una implementación por reglas (plantillas, regex, mapas). Beneficios concretos: 105 tests corren en CI sin credenciales, `python -m eval.run` produce una corrida reproducible que valida la mecánica del pipeline, y en producción un fallo del proveedor degrada a un perfil parcial en vez de un 500. El riesgo es confundir resultados de fallback con resultados reales; por eso cada `ResearchResponse` lleva `mode` (`llm`/`fallback`), los `QualityScores` llevan `judge` y `RESULTS.md` marca cada celda medida con stubs.

## 6. Precisión/recall por campo contra metadata YC como métrica de accuracy

Comparar perfiles con ground truth requiere reglas de match explícitas, no "parecido a ojo". Las reglas (contención para industria, ciudad para sede, ±25 % para headcount, Jaccard ≥ 0,3 para el one-liner) están en `eval/metrics.py` y se prueban en `tests/test_eval.py`. `website` se excluye porque el eval lo entrega como entrada. Un juez LLM para accuracy habría sido más flexible pero menos reproducible y con costo por corrida.

## 7. Rúbrica numerada en código para el LLM-as-judge

`RUBRIC` define tres criterios con niveles 1-5 anclados en ejemplos, se inyecta literalmente en el prompt del juez y se copia en `RESULTS.md`. Sin anclas, dos corridas del juez no son comparables. La heurística offline aproxima los mismos tres criterios para que las columnas existan (etiquetadas) aun sin llave.

## 8. `psycopg_pool` + JSONB en vez de un ORM

Hay una sola tabla y una sola consulta de escritura (upsert por `company_key`). Guardar el `ResearchResponse` completo como JSONB evita migraciones cada vez que cambia el esquema Pydantic y mantiene el repositorio en 80 líneas. El `Protocol ResearchRepository` con implementación en memoria evita levantar PostgreSQL en tests; la implementación real se prueba con un pool falso que verifica el SQL emitido.

## 9. Tavily con timeout propio y contabilidad de costo

`AsyncTavilyClient` no acepta timeout; se envuelve en `asyncio.wait_for` y en `tenacity` (3 intentos). Cada búsqueda registra `record_search()` para que `cost_usd` incluya los $0.001 por consulta y el ablation muestre el costo real de cada reflexión adicional.

## 10. Cobertura mínima del 70 % con 98 % medido

El umbral del DoD es 70 %; la suite llega al 98 % porque los fakes (LLM, pool, transporte HTTP) son baratos de escribir con `respx`, `pytest-mock` y `httpx.MockTransport`. No se persiguen las líneas restantes (ramas de `pragma: no cover` en reintentos agotados y el arranque del pool real), que requerirían servicios reales.

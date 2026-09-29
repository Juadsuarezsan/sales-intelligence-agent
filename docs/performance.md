# Rendimiento

## Dónde está el cuello de botella

El pipeline sin proveedores externos es despreciable: 20 investigaciones con stubs (5 empresas × 4 repeticiones, `website` conocido, 7 tool calls cada una) dan **3,7 ms de media por empresa** y cada nodo queda por debajo de 1 ms (`UsageTracker.node_latency_ms`, medido el 2026-09-29 en este entorno). Todo el tiempo real se va en:

1. **Llamadas LLM** (6 por empresa con presupuesto por defecto; +2 por cada ronda de reflexión adicional). Son secuenciales por diseño: cada nodo depende del anterior.
2. **Búsquedas Tavily** (hasta 5 por ronda). Hoy se ejecutan en serie dentro de `execute`; son independientes entre sí.
3. **Fetch del sitio web** (3 páginas en serie, 8 s de timeout cada una): en el peor caso un sitio caído cuesta 3 × 8 s × 2 intentos.

Las cifras de latencia real por empresa (spec: 18 s para el agente, 8 s single-search, 2 s template) quedan pendientes de la corrida con llaves; `eval/RESULTS.md` las llenará como media y p95 medidas.

## Qué ya está optimizado

| Medida | Dónde | Efecto |
|---|---|---|
| Endpoints `async` y clientes async (httpx, SDK Anthropic, Tavily) | `src/api/main.py`, `src/tools/` | Una réplica atiende muchas investigaciones concurrentes sin hilos |
| Caché de resultados por empresa | `ResearchRepository` (memoria con TTL o PostgreSQL) | Repetir una empresa cuesta 0 llamadas |
| Connection pooling | `psycopg_pool.AsyncConnectionPool` (`DB_POOL_MIN_SIZE`/`MAX_SIZE`) | Sin handshake por request |
| Batch con semáforo | `scripts/batch_run.py --concurrency`, `eval/run.py --concurrency` | Paraleliza por empresa; 4 por defecto |
| Timeouts y reintentos acotados | todas las herramientas y el LLM | Un proveedor lento no bloquea el worker indefinidamente |
| Presupuestos | `MAX_TOOL_CALLS_PER_COMPANY`, `MAX_REFLECTION_ITERATIONS` | Techo determinista de costo y latencia por empresa |
| Evidencia deduplicada y truncada | `_dedupe`, `content[:400]` en prompts | Menos tokens de entrada por llamada |

## Siguientes optimizaciones, por impacto esperado

1. **Búsquedas en paralelo** dentro de `execute` (`asyncio.gather` sobre las queries de la ronda): reduce la latencia de la ronda de 5×T a ~T. No se hizo todavía para mantener determinista el orden de la evidencia en el eval; basta ordenar por query al final.
2. **Fetch de las tres páginas en paralelo** con un timeout global por sitio (p. ej. 10 s) en lugar de 8 s por página.
3. **Prompt caching** en los system prompts largos (`SYSTEM` del builder, `RUBRIC` del juez): ~90 % menos costo de entrada en 4 de 6 llamadas.
4. **Fusionar `extract_hooks` y `write_email`** en una sola llamada cuando el modelo es fuerte: ahorra una ida y vuelta por empresa; el ablation debería confirmar que no baja la personalización antes de hacerlo.
5. **Cola + workers** para sacar la investigación del request HTTP (ver `docs/scalability.md`).

## Cómo medir

```bash
python -m eval.run --limit 20 --name latency-probe   # escribe latencia media/p95 por sistema
python scripts/batch_run.py --limit 20 --concurrency 8
```

Cada `ResearchResponse` trae `latency_ms`, `input_tokens`, `output_tokens` y `cost_usd`; el log de cada nodo incluye `latency_ms` y el `trace_id` para correlacionar con LangSmith cuando está activo.

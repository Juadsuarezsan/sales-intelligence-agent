# Escalabilidad

## Perfil de carga de una corrida

Una investigación completa con el presupuesto por defecto hace hasta 8 tool calls (5-6 búsquedas Tavily, 1 HN, 1 fetch de 3 páginas) y 6 llamadas LLM (plan, reflect, build, hooks, email, judge) más una llamada extra de plan y de reflect por cada ronda de reflexión adicional. El proceso es 100 % I/O-bound: el pipeline con stubs tarda ~4 ms por empresa (medido, ver `docs/performance.md`), así que toda la latencia real viene de red y proveedor.

## Capacidad actual (una réplica, sin llaves reales medidas)

- El API es asíncrono (FastAPI + httpx + SDK async); una réplica puede tener decenas de investigaciones en vuelo limitada por los rate limits del proveedor, no por CPU.
- `slowapi` limita por IP (`RATE_LIMIT=20/minute` por defecto) para proteger la cuota de Tavily.
- El repositorio actúa como caché por nombre de empresa: una segunda petición por la misma empresa no cuesta nada.

## 100× (miles de empresas al día)

Qué funciona sin cambios:

- `scripts/batch_run.py` procesa con `asyncio.Semaphore(concurrency)`; subir `--concurrency` hasta el límite de requests/minuto del proveedor.
- PostgreSQL con una fila JSONB por empresa: 100 000 filas son < 1 GB; el índice por `updated_at` sirve la galería.

Qué hay que ajustar:

- **Cuota Tavily**: 1 000 búsquedas gratis/mes cubren ~150 empresas. A 100× hay que pagar ($0.001/búsqueda → ~$6 por 1 000 empresas) o reducir búsquedas por empresa (el ablation dice cuánto aporta cada reflexión).
- **Rate limits de Anthropic**: 6-10 llamadas por empresa. Con 4 workers concurrentes se procesan ~200 empresas/hora a ~18 s por empresa; más allá hace falta un aumento de tier o una cola.
- **Caché de búsquedas**: la misma query para la misma empresa se repite entre corridas (eval, batch, API). Un caché por hash de query en Redis o en la misma tabla evitaría pagar dos veces.

## 1 000× (cientos de miles de empresas)

- Mover la investigación fuera del request: `POST /api/research` encola un job y devuelve `202` con `trace_id`; workers consumen la cola (Redis/RQ, Celery o SQS). El grafo ya es una función pura `research_company(graph, ...)`, así que el worker es trivial.
- **Batch API de Anthropic** para los nodos que no son interactivos (`build`, `extract_hooks`, `write_email`, `score` sobre evidencia ya recogida): 50 % de descuento y sin presión sobre el rate limit interactivo.
- **Prompt caching**: los system prompts (`SYSTEM`, `RUBRIC`) son estables; con cache_control el costo de entrada de 4 de las 6 llamadas cae ~90 %.
- Persistir la evidencia cruda por URL para no volver a hacer fetch del sitio en cada corrida.

## 10 000×

- Particionar por lote de empresas y por región de despliegue; la tabla `company_research` se puede sharded por `company_key`.
- Sustituir el juez LLM por un modelo pequeño fine-tuneado sobre las puntuaciones del juez grande (destilación) para el scoring masivo, dejando al juez fuerte sólo para muestreo de control.
- El fetcher de sitios web necesita un pool de proxies y respeto de `robots.txt` a esa escala; hoy es un cliente httpx directo.

## Lo que no se escala

`python -m eval.run` corre secuencialmente los seis sistemas sobre las mismas 200 empresas para que las corridas sean comparables; se paraleliza por empresa (`--concurrency`) pero no por sistema. Es intencional: el eval mide calidad, no throughput.

## Costo operativo estimado (supuestos explícitos)

Supuestos: 1 000 investigaciones/mes, presupuesto por defecto, ~1 500 tokens de entrada y ~400 de salida por llamada LLM, 6 llamadas por empresa, 6 búsquedas Tavily por empresa, precios del `.env.example` ($3 / $15 por millón de tokens; $0.001 por búsqueda).

| Concepto | Cálculo | USD/mes |
|---|---|---|
| Tokens de entrada | 1 000 × 6 × 1 500 = 9 M × $3/M | 27 |
| Tokens de salida | 1 000 × 6 × 400 = 2,4 M × $15/M | 36 |
| Búsquedas Tavily | 1 000 × 6 = 6 000 − 1 000 gratis = 5 000 × $0.001 | 5 |
| Hosting API + PostgreSQL | plan básico Railway/Render | 10-25 |
| **Total** | | **~$80-95** |

Son estimaciones, no mediciones: la corrida real de `eval/run` reemplaza los supuestos de tokens por valores medidos (`input_tokens`, `output_tokens`, `cost_usd` por empresa).

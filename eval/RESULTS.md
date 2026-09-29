# Resultados de evaluación — B2B Sales Intelligence Agent

- Corrida: `eval/runs/2026-09-29-offline-stubs.json` generada el 2026-09-29T01:22:01+00:00
- Empresas evaluadas: **200** (ground truth: `data/eval/yc_ground_truth_200.json`)
- Modo: **fallback determinista, sin LLM**
- Modelo del agente: `claude-sonnet-4-5-20250929` · juez: `claude-sonnet-4-5-20250929` · LLM habilitado: False · búsqueda: `stub` · noticias: `stub` · sitio web: `stub`
- Regenerar: `python -m eval.run`

> **Aviso.** Esta corrida se hizo sin llaves de API: los nodos LLM usan su fallback determinista y las búsquedas devuelven resultados enlatados (stubs). Los números marcados *stubs* miden únicamente la mecánica del pipeline (tool calls, iteraciones, extracción por reglas) y **no representan la calidad del sistema**. Las celdas `pendiente` se llenan al ejecutar el mismo comando con `ANTHROPIC_API_KEY` y `TAVILY_API_KEY` configuradas.

## Tabla comparativa (spec)

| Sistema | Accuracy (campos vs YC) | Personalization | Latencia | Costo/empresa |
|---|---|---|---|---|
| Template only | P 0.0 % / R 0.0 % / F1 0.0 % (sin búsqueda; medido) | proxy 0.20; rúbrica heurística 2.00/5 (medido, sin LLM) | 0 ms (medido; sin I/O) | $0.0000 (sin LLM ni búsqueda; medido) |
| Single-search + email | P 0.0 % / R 0.0 % / F1 0.0 % (stubs; no representa al sistema) | pendiente (requiere ANTHROPIC_API_KEY); proxy heurístico 0.20, rúbrica heurística 2.00/5 | pendiente (requiere ANTHROPIC_API_KEY + TAVILY_API_KEY); stubs: 0 ms | pendiente (requiere ANTHROPIC_API_KEY + TAVILY_API_KEY); stubs: $0.0000 |
| Agent loop con reflection (este) | P 0.5 % / R 0.1 % / F1 0.2 % (stubs; no representa al sistema) | pendiente (requiere ANTHROPIC_API_KEY); proxy heurístico 0.60, rúbrica heurística 5.00/5 | pendiente (requiere ANTHROPIC_API_KEY + TAVILY_API_KEY); stubs: 31 ms | pendiente (requiere ANTHROPIC_API_KEY + TAVILY_API_KEY); stubs: $0.0000 |

Accuracy = precisión/recall micro-promediadas sobre los campos `industry`, `location`, `employees_est`, `one_liner` del `CompanyProfile` contra la metadata de YC (reglas de match en `eval/metrics.py`). Personalization = criterio 1 de la rúbrica (1-5) juzgado por LLM sobre los emails; el *proxy heurístico* es el solapamiento de palabras clave de `score_personalization`.

## Ablation: iteraciones de reflection (0 / 1 / 2 / 3)

| Reflexiones | Accuracy | Rúbrica personalization | Tool calls (media / p95) | Iteraciones (media) | Latencia (media) | Costo/empresa |
|---|---|---|---|---|---|---|
| 0 | P 0.5 % / R 0.1 % (stubs) | pendiente (requiere ANTHROPIC_API_KEY); heurística 5.00/5 | 7.0 / 7 | 1.0 | 33 ms (stubs) | pendiente (requiere ANTHROPIC_API_KEY + TAVILY_API_KEY) |
| 1 | P 0.5 % / R 0.1 % (stubs) | pendiente (requiere ANTHROPIC_API_KEY); heurística 5.00/5 | 7.0 / 7 | 1.0 | 35 ms (stubs) | pendiente (requiere ANTHROPIC_API_KEY + TAVILY_API_KEY) |
| 2 | P 0.5 % / R 0.1 % (stubs) | pendiente (requiere ANTHROPIC_API_KEY); heurística 5.00/5 | 7.0 / 7 | 1.0 | 34 ms (stubs) | pendiente (requiere ANTHROPIC_API_KEY + TAVILY_API_KEY) |
| 3 | P 0.5 % / R 0.1 % (stubs) | pendiente (requiere ANTHROPIC_API_KEY); heurística 5.00/5 | 7.0 / 7 | 1.0 | 31 ms (stubs) | pendiente (requiere ANTHROPIC_API_KEY + TAVILY_API_KEY) |

El punto donde agregar reflexión deja de mejorar se lee de esta tabla cuando la corrida es real: la primera fila cuya accuracy/personalization no supera a la anterior marca el codo costo-calidad. Con stubs las filas son idénticas por construcción (el reflector heurístico declara suficiencia en la primera ronda), así que la curva queda pendiente.

## Distribución de tool calls por empresa

| Sistema | media | p50 | p95 | min | max |
|---|---|---|---|---|---|
| Template only | 0.00 | 0 | 0 | 0 | 0 |
| Single-search + email | 1.00 | 1 | 1 | 1 | 1 |
| Agent loop, 0 reflexiones (ablation) | 7.00 | 7 | 7 | 7 | 7 |
| Agent loop, 1 reflexión (ablation) | 7.00 | 7 | 7 | 7 | 7 |
| Agent loop, 2 reflexiones (ablation) | 7.00 | 7 | 7 | 7 | 7 |
| Agent loop con reflection (este) | 7.00 | 7 | 7 | 7 | 7 |

## Accuracy por campo (agente, 3 reflexiones)

| Campo | TP | FP | FN |
|---|---|---|---|
| `industry` | 0 | 0 | 200 |
| `location` | 0 | 0 | 200 |
| `employees_est` | 0 | 0 | 200 |
| `one_liner` | 1 | 199 | 0 |

## Rúbrica del LLM-as-judge (`src/agents/quality_scorer.py`)

```text
Score the outreach email on three criteria, each an integer from 1 to 5.

1. PERSONALIZATION - does the email prove it was written for THIS company?
   1 = fully generic; could be sent to anyone ("I came across your company").
   2 = mentions the company name only.
   3 = references the industry or a vague fact ("you are growing fast").
   4 = references one specific, verifiable fact from the profile (a funding
       round, a named product, a named executive, an open role).
   5 = weaves two or more specific facts into a coherent reason to reach out now.
   Example 5: "Congrats on the $40M Series B - with six open platform roles,
   onboarding those engineers onto one workflow will matter."

2. ACCURACY - is every factual claim supported by the profile?
   1 = contains a fabricated or contradicted fact (wrong round, wrong CEO).
   2 = several unsupported claims presented as facts.
   3 = one unsupported claim or an exaggeration of a real fact.
   4 = all claims supported; minor imprecision in wording.
   5 = every claim traceable to a profile field or signal, no exaggeration.
   Example 1: profile says Seed round, email congratulates on Series C.

3. CTA_CLARITY - is the call to action specific and low-friction?
   1 = no call to action.
   2 = vague ("let me know your thoughts").
   3 = asks for something but without format or time ("can we talk?").
   4 = one concrete ask with a format or duration ("20-minute call next week").
   5 = concrete ask, duration, timeframe and an easy opt-out or alternative.
   Example 5: "Open to a 20-minute call Tuesday or Thursday? Happy to send a
   two-minute video instead if easier."
```

## Pendientes que requieren llaves

- Information accuracy real sobre las 200 empresas (Tavily + Claude): pendiente (requiere ANTHROPIC_API_KEY + TAVILY_API_KEY).
- Personalization score con juez LLM sobre 100 emails: pendiente (requiere ANTHROPIC_API_KEY).
- Latencia y costo reales por empresa: pendiente (requiere ANTHROPIC_API_KEY + TAVILY_API_KEY).
- Curva costo vs calidad por número de reflexiones: requiere la corrida real.

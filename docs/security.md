# Seguridad

## Secretos

- Ninguna llave vive en el repositorio. `.env` está en `.gitignore`; `.env.example` contiene sólo nombres de variables con valores vacíos.
- Las llaves se leen únicamente en `src/config.py` (pydantic-settings). Los valores en blanco se normalizan a `None`, lo que activa el fallback offline.
- Los logs nunca imprimen llaves ni cuerpos de prompts completos: `log_node` resume listas y modelos Pydantic a su tipo/longitud.

### Resultado de `gitleaks`

```
$ gitleaks version
8.21.2
$ gitleaks detect --no-banner --redact --source .
INF 21 commits scanned.
INF scan completed in 464ms
INF no leaks found
```

Ejecutado el 2026-09-29 sobre el branch `claude/fase-a` (binario descargado de GitHub Releases). El workflow de CI vuelve a correrlo en cada push/PR con `gitleaks/gitleaks-action@v2`.

## Superficie HTTP (`src/api/main.py`)

| Control | Implementación |
|---|---|
| CORS | `CORS_ORIGINS` (lista separada por comas). `*` se descarta aunque se configure. Métodos `GET`/`POST`, cabeceras `Content-Type` y `X-Trace-Id`. |
| Rate limiting | `slowapi` por IP en `POST /api/research`; expresión en `RATE_LIMIT` (por defecto `20/minute`). Respuesta `429`. |
| Validación | Pydantic v2 en todos los cuerpos: `company_name` 2–120 caracteres, `domain` con patrón de host, `seed_context` ≤ 2 000, `max_reflection_iterations` 0–5. Errores → `422` con detalle por campo. JSON malformado → `422`. |
| Errores internos | `500` con mensaje genérico; el detalle va al log con `trace_id`, nunca al cliente. |
| Trazabilidad | Cabecera `X-Trace-Id` de entrada/salida por request. |

## Salidas renderizadas

El demo (`demo/index.html`) escapa todo texto proveniente del JSON (`esc()`) antes de insertarlo en el DOM, incluidos los emails generados y las URLs de fuentes (`href` escapado, `rel="noopener"`). El servidor devuelve JSON, nunca HTML.

## PII

El sistema procesa nombres de ejecutivos que aparecen en fuentes públicas (páginas *About*, prensa). No se almacenan correos ni teléfonos personales; `KeyPerson` sólo tiene `name`, `role` y `linkedin_url` (opcional, sólo si la fuente lo publica). No hay redacción automática porque no se ingieren datos privados; si el producto se conectara a CRMs habría que añadirla antes de persistir.

## Dependencias

Todas las dependencias están fijadas con `==` en `pyproject.toml`. Las llamadas externas (Anthropic, Tavily, HN Algolia, sitios web) tienen timeout explícito y reintentos acotados con `tenacity`; ningún fallo de herramienta aborta el run (se registra y se continúa).

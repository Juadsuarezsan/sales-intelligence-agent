# Esquema de datos

## Fuente

| Campo | Valor |
|---|---|
| Dataset | Directorio público de empresas de Y Combinator, espejo JSON `yc-oss/api` |
| URL exacta | `https://raw.githubusercontent.com/yc-oss/api/main/companies/all.json` |
| Repositorio | https://github.com/yc-oss/api (actualizado a diario desde el índice Algolia de YC) |
| Licencia | Código del repositorio: MIT. Los datos son información pública publicada por Y Combinator en https://www.ycombinator.com/companies; se usan sólo para evaluación. |
| Descarga | `python scripts/download_data.py` (o `make data`) |
| Tamaño | ~10 MB, 6 253 registros en la copia del 2026-09-28; el archivo crudo está en `.gitignore` (`data/raw/`) |
| Hash | `data/MANIFEST.txt` guarda el SHA-256 del crudo y del eval set |

## Registro crudo (`data/raw/yc_companies_all.json`)

Lista JSON de objetos con estas claves (tipos observados en la copia descargada):

| Columna | Tipo | Descripción | Rango / valores |
|---|---|---|---|
| `id` | int | Identificador YC | 1 … ~35 000 |
| `name` | str | Nombre de la empresa | |
| `slug` | str | Slug de la página YC | |
| `former_names` | list[str] | Nombres anteriores | |
| `website` | str | Sitio web | vacío en 37 registros |
| `all_locations` | str | Sedes, `"Ciudad, Región, País; …"` | vacío en 153 registros |
| `long_description` | str | Descripción larga | |
| `one_liner` | str | Descripción de una línea | |
| `team_size` | int \| null | Tamaño del equipo reportado | 1 … 10 000+; null en 111 registros |
| `industry` | str | Industria de primer nivel | `B2B` (3 183), `Consumer`, `Healthcare`, `Fintech`, `Industrials`, `Real Estate and Construction`, `Education`, `Government`, `Unspecified` |
| `subindustry` | str | `"Industria -> Subindustria"` | 59 industrias |
| `launched_at` | int | Epoch de lanzamiento en YC | |
| `tags` | list[str] | Etiquetas | 337 etiquetas distintas |
| `top_company`, `isHiring`, `nonprofit` | bool | Banderas | |
| `batch` | str | Cohorte YC | `Winter 2012` … `Summer 2026` (51 batches) |
| `status` | str | Estado | `Active` (4 324), `Inactive` (1 081), `Acquired` (825), `Public` (23) |
| `stage` | str | Etapa | `Early` (5 160), `Growth` (1 093) |
| `regions` | list[str] | Regiones | |
| `url` | str | Página en ycombinator.com | |
| `api` | str | Endpoint JSON individual | |

## Eval set (`data/eval/yc_ground_truth_200.json`)

Construido por `scripts/download_data.py::build_eval_set` con **semilla 20260516**:

1. Se filtran los registros *elegibles*: `status == "Active"`, `website`, `industry`, `all_locations` y `one_liner` no vacíos, `team_size` entero > 0 (4 106 registros).
2. Se ordenan por `id`, se barajan con `random.Random(20260516)` y se toman 200.
3. Se conservan las columnas de abajo; `all_locations` se renombra `location` y `url` a `yc_url`.

```json
{
  "source": "https://raw.githubusercontent.com/yc-oss/api/main/companies/all.json",
  "sample_seed": 20260516,
  "n": 200,
  "companies": [
    {"id": 19, "name": "Rescale", "slug": "rescale", "website": "https://rescale.com",
     "industry": "B2B", "subindustry": "B2B -> Engineering, Product and Design",
     "location": "San Francisco, CA, USA", "team_size": 250,
     "one_liner": "High Performance Computing Built for the Cloud",
     "long_description": "…", "batch": "Winter 2012", "status": "Active", "stage": "Growth",
     "tags": ["…"], "isHiring": true, "launched_at": 1322045523,
     "yc_url": "https://www.ycombinator.com/companies/rescale"}
  ]
}
```

El ground truth son los campos de metadata de YC tal cual (no hay valores sintéticos ni generados por LLM). No hay split train/val/test porque el sistema no se entrena: los 200 registros son el conjunto de prueba y se evalúan completos en cada corrida.

## Uso de cada campo en la evaluación (`eval/metrics.py`)

| Campo YC | Campo del `CompanyProfile` | Regla de acierto |
|---|---|---|
| `industry` | `industry` | contención sin distinguir mayúsculas, en cualquier sentido |
| `location` | `location` | la ciudad predicha (primer token antes de la coma) aparece en `all_locations`, o viceversa |
| `team_size` | `employees_est` | dentro de ±25 % |
| `one_liner` | `one_liner` | Jaccard de tokens de contenido ≥ 0,3 |
| `website` | `website` | **no se puntúa**: el eval lo entrega como entrada para que el fetcher funcione |

## Salidas generadas

| Ruta | Contenido | Versionado |
|---|---|---|
| `data/outputs/<slug>.json` | Un `ResearchResponse` por empresa procesada por `scripts/batch_run.py` | ignorado (`.gitignore`) |
| `demo/predictions.json` | Galería compacta (`yc` + `result`) para `demo/index.html` | versionado |
| `eval/runs/<fecha>-<nombre>.json` | Reporte completo de `python -m eval.run` con filas por empresa | versionado |
| `eval/RESULTS.md` | Tabla comparativa regenerada desde la última corrida | versionado |

## Tabla PostgreSQL (`src/storage/repository.py::SCHEMA_SQL`)

```sql
CREATE TABLE IF NOT EXISTS company_research (
    company_key   TEXT PRIMARY KEY,      -- nombre normalizado (minúsculas, espacios colapsados)
    company_name  TEXT NOT NULL,
    trace_id      TEXT NOT NULL,
    mode          TEXT NOT NULL,         -- 'llm' | 'fallback'
    cost_usd      DOUBLE PRECISION NOT NULL DEFAULT 0,
    latency_ms    INTEGER NOT NULL DEFAULT 0,
    payload       JSONB NOT NULL,        -- ResearchResponse completo
    updated_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);
```

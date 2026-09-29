# Análisis de errores

## Estado

El análisis definitivo (los diez peores casos del agente con Claude + Tavily sobre las 200 empresas) **requiere `ANTHROPIC_API_KEY` y `TAVILY_API_KEY`** y queda pendiente. Lo que sigue analiza la corrida offline versionada en `eval/runs/2026-09-29-offline-stubs.json` (fallback determinista, sin LLM), que sirve para validar la mecánica del eval y para documentar los modos de fallo que ya se conocen del código.

## Corrida offline: qué falla y por qué

Agente con 3 reflexiones, 200 empresas, campos puntuados `industry`, `location`, `employees_est`, `one_liner`:

| Campo | TP | FP | FN | Causa |
|---|---|---|---|---|
| `industry` | 0 | 0 | 200 | El builder heurístico no infiere industria; sólo el LLM la asigna a partir de la evidencia. |
| `location` | 0 | 0 | 200 | La regex busca "headquartered/based/located in …"; el stub del sitio dice "Headquartered in San Francisco, CA" pero el texto queda detrás de "Founded by…" y el eval compara con la ciudad real de cada empresa, que el stub no conoce. |
| `employees_est` | 0 | 0 | 200 | Ninguna fuente stub menciona headcount. |
| `one_liner` | 1 | 199 | 0 | El heurístico toma el meta description del stub ("Modern B2B software platform.") para todas las empresas; sólo Docker supera el umbral Jaccard por casualidad léxica. |

Los diez peores casos por (precisión, recall) son los diez primeros del eval set (Rescale, Gusto, Instapainting, Benchling, Pushbullet, Zidisha, UserGems, Meadow, X-Zell, Gemnote): todos con 0/0 y el mismo gancho ("recent funding: … raises Series B") porque el stub de búsqueda devuelve la misma noticia de financiación para cualquier empresa. No hay nada que aprender de la variación entre empresas en esta corrida; la lección es que **los stubs son deliberadamente ciegos al nombre de la empresa** para que nadie confunda su salida con investigación real.

## Modos de fallo esperados con el sistema real (hipótesis a verificar)

1. **Homónimos**: "Meadow" o "Docker" devuelven resultados de otras entidades (la palabra común, la empresa homónima). Mitigación prevista: pasar `website` como contexto al planner y filtrar evidencia cuyo dominio no coincida.
2. **Datos desactualizados en YC**: `team_size` en YC se actualiza cuando la empresa lo reporta; el agente puede acertar el headcount actual y "fallar" contra un ground truth viejo. El margen de ±25 % amortigua, no elimina, este ruido.
3. **Sedes múltiples**: `all_locations` lista varias ciudades; la regla acepta cualquiera de ellas, pero un perfil que diga sólo "Remote" cuenta como FP.
4. **Sitios 100 % JavaScript**: el fetcher recibe HTML sin texto; el builder no obtiene one-liner ni sede y cae al one-liner de la búsqueda, con peor Jaccard.
5. **Reflexión que no converge**: si el reflector pide siempre "funding" para una empresa bootstrapped, el bucle agota `MAX_REFLECTION_ITERATIONS` gastando 2 rondas extra sin ganancia. El ablation mide exactamente ese costo.
6. **Alucinación de rondas de financiación** en el email cuando la evidencia mezcla empresas: el criterio 2 de la rúbrica (accuracy) está diseñado para castigarlo con 1/5; el heurístico ya detecta montos/rondas ausentes del perfil.
7. **Industria con vocabulario distinto** ("Developer tools" vs `B2B`): la regla de contención marca FP aunque sea correcto en sentido amplio. Se contempla un mapa de sinónimos hacia las 9 industrias de YC.
8. **Nombres de personas mal segmentados** por la regex del fallback ("CEO is Maria Lopez and…"): sólo afecta al modo fallback; el LLM produce `key_people` estructurado.

## Cómo se regenera este análisis

```bash
python -m eval.run --name real            # con llaves configuradas
python - <<'EOF'
import json
d = json.load(open("eval/runs/<fecha>-real.json"))
agent = next(s for s in d["systems"] if s["system"] == "agent_reflection_3")
worst = sorted(agent["rows"], key=lambda r: (r["precision"], r["recall"]))[:10]
for r in worst: print(r["company"], r["precision"], r["recall"], r["fields"], r["hook"])
EOF
```

Cada fila trae los campos TP/FP/FN, el gancho del email y los tokens/costo, suficiente para escribir la hipótesis por caso.

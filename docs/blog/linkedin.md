# Borrador de post para LinkedIn

Publiqué el código de un agente de sales intelligence B2B: le das el nombre de una empresa y devuelve un perfil estructurado y un email de outreach personalizado, con puntuación de calidad.

Cómo funciona:
- Bucle plan → execute → reflect en LangGraph: Claude planifica consultas, ejecuta búsquedas (Tavily, HackerNews, sitio web con httpx + selectolax) y decide si la evidencia alcanza o hay que refinar.
- Perfil validado con Pydantic, extractor de ganchos de personalización, redactor de email y un juez LLM con rúbrica numerada (personalización, exactitud, claridad del CTA).
- Ground truth: 200 empresas de Y Combinator (dataset público yc-oss), precisión/recall por campo, dos baselines (template-only y single-search + email) y ablation de 0 a 3 rondas de reflexión.

Lo que más me gustó construir: cada nodo tiene un fallback determinista, así que la suite de 105 tests, la demo y la evaluación corren sin llaves de API, y cada resultado dice explícitamente si vino del modelo o del fallback. Los números con Claude + Tavily quedan marcados como pendientes hasta la corrida real; nada inventado.

Stack: Python 3.11, LangGraph, SDK de Anthropic (modelo fijado por ID), FastAPI, PostgreSQL con psycopg_pool, pytest + respx, ruff/black/mypy --strict, GitHub Actions.

Repo (MIT): https://github.com/Juadsuarezsan/sales-intelligence-agent
Demo estática: https://juadsuarezsan.github.io/sales-intelligence-agent/demo/

Si trabajas en sales-tech o en agentes con evaluación seria, me interesa tu opinión sobre la rúbrica del juez y las reglas de match del eval.

#AIEngineering #LangGraph #Claude #SalesTech #Python #LLM

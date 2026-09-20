# Brief para Claude Code — `jev-acento`

Auditoría independiente y reproducible de Jev (TypeSafe AI) en español y portugués, más packs de preguntas tipadas para flujos LatAm. Tagline: "¿Jev entiende tu acento?". Fecha del brief: 20-sep-2026.

**Nombres:** repo y paquete PyPI `jev-acento`; módulo Python `jev_acento`; comando de consola `acento`. Verificado libre en GitHub, PyPI y npm el 20-sep-2026.

## 1. Contexto

Jev es un modelo "System One": recibe un `state` y preguntas tipadas (Choice, Score, Noul) y devuelve respuestas con probabilidades, sin generar texto. Toda su propuesta de valor es que esas probabilidades estén calibradas, para que el software automatice arriba de un umbral y derive a un humano el resto.

TypeSafe dice que el inglés es el idioma principal y que los demás idiomas se manejan "pero no igual de bien", sin dar números. Ya existe una auditoría ruso vs. inglés (`AHTOOOXA/jev-cyrillic-audit`) que encontró degradación en XNLI (accuracy 88,3% → 77,3%, ECE 0,032 → 0,096) y ~3× tokens para el mismo texto. Esa auditoría dejó congeladas pero **sin correr** las celdas con instrucciones en el idioma del state. No hay nada equivalente para español ni portugués.

## 2. Preguntas que el repo tiene que responder

1. **Costo del idioma.** Con instrucciones en inglés, ¿Jev pierde accuracy y calibración cuando el `state` está en español o portugués, sobre los mismos ítems etiquetados por humanos?
2. **Idioma de las instrucciones.** Con `state` en español/portugués, ¿conviene escribir `instructions` y `criteria` en inglés o en el idioma del state?
3. **Costo operativo.** Ratio de tokens ES/EN y PT/EN (state-only y por llamada), latencia p50/p95, cobertura automatizable a umbrales típicos (0,5 / 0,9).

La pregunta 2 es el aporte original: es la decisión práctica que tiene que tomar cualquier equipo LatAm y nadie la midió.

## 3. Alcance

**Sí (v0.1):** harness de auditoría, pre-registro, corrida, `results.md` + figuras, README, y un **CLI para datos propios** (sección 8) para que cualquiera compare idiomas de instrucciones sobre su propio dataset etiquetado. Se publica al cerrar la Fase 2.
**Después, solo si Marcos lo pide:** Fase 3 (packs de preguntas) y Fase 4 (módulo de prompt injection ES/PT).
**No:** otro SDK/wrapper genérico, otro servidor MCP, otro calibrador de umbrales (ya existen jevcal, Janus, jev-bench). No construir UI.

## 4. Acceso a Jev (verificado en docs el 20-sep-2026)

El acceso directo a TypeSafe tiene waitlist. Se entra por **Vercel AI Gateway**, que expone un passthrough con el mismo formato nativo de TypeSafe, así que no hace falta el AI SDK ni TypeScript:

```
POST https://ai-gateway.vercel.sh/typesafe/v1/systemone
Authorization: Bearer $AI_GATEWAY_API_KEY
{
  "model": "typesafe-ai/jev",
  "state": <string | objeto JSON | array>,
  "questions": {
    "<id>": { "type": "choice" | "score" | "noul", "instructions": "...", "criteria": ... }
  }
}
→ { "model", "answers": { "<id>": {...} }, "usage": { "input_tokens", ... },
    "provider_metadata": { "gateway": { "cost", "generationId", ... } } }
```

Formato de respuestas nativo: Choice → `choice`, `probabilities`, `confidence`; Score → `score`, `probabilities`, `confidence`; Noul → `noul` (0–1).

Armá una **costura de proveedor** (`base_url`, `api_key_env`, `model`) para que el mismo runner funcione con:

| Proveedor | base_url | key | model |
|---|---|---|---|
| Gateway (default) | `https://ai-gateway.vercel.sh/typesafe` | `AI_GATEWAY_API_KEY` | `typesafe-ai/jev` |
| TypeSafe directo | `https://api.typesafe.ai` | `TYPESAFE_API_KEY` | `jev-1.13.0` (ID versionado, nunca el alias) |

Datos a respetar: 64k tokens por request y 32k para `state`; probabilidades redondeadas a 2 decimales (una distribución puede sumar 0,99: no renormalizar); precio USD 0,042 por millón de tokens de input, output gratis; rate limit directo 1.200 rpm y cambia sin aviso.

**Problema conocido:** por el Gateway el campo `model` de la respuesta devuelve `typesafe-ai/jev`, no el ID versionado. En el smoke test verificá si el Gateway acepta un ID versionado o expone la versión en algún campo. Si no, registrá fecha/hora y `generationId` de cada llamada y declaralo como limitación en el README.

## 5. Diseño experimental

Diseño pareado: los mismos ítems en cada brazo. Por cada idioma objetivo L ∈ {es, pt}:

| Brazo | State | Instructions + criteria |
|---|---|---|
| A | inglés | inglés |
| B | L | inglés |
| C | L | L |

- Las **claves de opción** (`entailment`, `alarm_set`, `option_1`…) quedan en inglés e idénticas en los tres brazos. En C solo se traducen `instructions` y las descripciones de `criteria`.
- Un ítem por llamada, una pregunta por llamada.
- Dos pasadas por celda: la pasada 0 es el resultado principal, la pasada 1 mide estabilidad (la API no cachea requests idénticos).
- n = 600 ítems por dataset, estratificados por etiqueta gold, semilla fija declarada en `PREREG.md`.
- Comparaciones primarias pre-registradas: **B − A** (comparable con la auditoría rusa) y **C − B**.

## 6. Datasets

Solo datasets públicos, paralelos y con etiquetas humanas. Verificá en Hugging Face ID, configs, split, licencia y fijá la revisión antes de congelar.

| Dataset | HF id (verificar) | Idiomas | Primitiva | Notas |
|---|---|---|---|---|
| XNLI | `facebook/xnli` (test) | en, es | Choice, 3 opciones | Traducción humana. Sin tag de licencia en HF. |
| MASSIVE intent | `mteb/amazon_massive_intent` (test) | en, es-ES, pt-PT | Choice, ~60 intents con glosa de una línea | Glosas en inglés (A, B) y traducidas (C). |
| Belebele | `facebook/belebele` | eng_Latn, spa_Latn, por_Latn | Choice, 4 opciones | Pasaje + pregunta + opciones van dentro de `state`; criteria neutras `option_1..4`. |
| PAWS-X | `google-research-datasets/paws-x` (test) | en, es | Noul (¿son paráfrasis?) | Única cobertura de Noul en v0.1. |

- Verificá la alineación entre idiomas ítem por ítem (la auditoría rusa encontró 2 filas cruzadas en XNLI). Excluí filas desalineadas antes de muestrear y documentalo.
- **No redistribuir texto fuente.** Commitear solo `item_id`, gold y hash del state; el texto se recupera re-joineando contra el dataset.
- Score queda fuera de v0.1 (no hay dataset ordinal paralelo adecuado). Declararlo como limitación.
- Limitación a declarar: MASSIVE es es-ES y pt-PT, no rioplatense ni pt-BR. Eso lo cubre la Fase 3.

## 7. Prompts y congelamiento

- `prompts/<dataset>.<lang>.json` con `instructions` y `criteria` por brazo. Seguir las guías del proveedor: state como objeto JSON, referenciar campos con backticks, describir opciones en vez de etiquetarlas, preguntas atómicas.
- Redactá vos los borradores ES y PT. **Marcos revisa el español y alguien nativo de Brasil revisa el portugués antes de congelar.** No congelar sin ese OK.
- `PREREG.md` (hipótesis, brazos, n, semilla, métricas, regla de decisión, qué es exploratorio) + `PREREG.sha256` + `prompts.sha256`, commiteados **antes** de la primera corrida real. El runner se niega a escribir en `runs/` si los hashes no verifican o si hay archivos congelados sucios en git.
- Una sola redacción por celda es una muestra de tamaño 1 del espacio de redacciones: declararlo. Exploratorio opcional: segunda paráfrasis sobre 200 ítems.

## 8. Runner

Python 3.12 + `uv`. Dependencias: `httpx`, `datasets`, `pyarrow`, `numpy`, `matplotlib`, `pytest`; `netcal` solo en dev para cross-check.

- Concurrencia 8, pacer configurable (default 600 rpm), reintentos con backoff en 429/5xx respetando `retry-after`.
- Cada respuesta se guarda como fila derivada en `runs/<run_id>-<dataset>-<arm>-pass<k>.jsonl`: `item_id, arm, pass, gold, pred, p_max, confidence, probs, input_tokens, latency_ms, model, generation_id, ts`.
- `--dry-run` contra una API falsa local; `--max-usd` como tope duro (default 5).
- Reanudable: si se corta, retoma sin repetir ítems ya registrados.
- Estimación: ~19.000 llamadas y ~15 M de tokens, menos de USD 1 y unos 35 minutos a 600 rpm.

**CLI para datos propios (la parte "herramienta").** Mismo runner, mismas métricas, mismo reporte, pero sobre datos del usuario:

```
acento compare --data mis_tickets.jsonl --questions q.en.json q.es.json --out reporte/
```

- `--data`: JSONL con `id`, `state` y `gold` por pregunta. No necesita datos paralelos en inglés.
- `--questions`: dos o más versiones del mismo set de preguntas (mismas claves, distinto idioma o redacción). El CLI corre cada versión sobre los mismos ítems y reporta deltas pareados de accuracy, ECE con piso de ruido, cobertura a umbral y tokens.
- Salida: `reporte/results.md`, `results.json` y figuras. Nada sale de la máquina del usuario salvo las llamadas a Jev.
- Incluir `examples/` con un dataset sintético chico para que `acento compare` funcione recién clonado, con la API falsa si no hay key.
- Instalación con `uv tool install jev-acento` o `pipx`; entry point `acento = jev_acento.cli:main` en `pyproject.toml`.

## 9. Métricas y regla de decisión

Implementación propia en `metrics.py` (solo numpy), con tests sobre datos sintéticos y cross-check contra `netcal`.

- Accuracy y macro-F1 con IC bootstrap percentil (1.000 remuestreos).
- ECE con 10 bins de igual ancho sobre `p_max`, bordes explícitos (cerrado a izquierda, bin superior cerrado) por el redondeo a 2 decimales. Para Noul, `p_max = max(p, 1 − p)`.
- **Piso de ruido del ECE**: 1.000 simulaciones bajo calibración perfecta con el vector de confianzas del propio brazo; reportar ECE, piso y ratio.
- Bins de confiabilidad con conteos, accuracy selectiva vs. cobertura, AURC, cobertura → accuracy a umbrales 0,5 y 0,9.
- **Deltas pareados** B − A y C − B: remuestrear índices de ítems una vez y recalcular ambos brazos. Nunca concluir por "los IC se solapan".
- Determinismo pasada 0 vs. 1: tasa de flips, κ, |Δp_max| medio y máximo.
- Tokens: ratio por llamada y ratio state-only (restando el overhead fijo de la pregunta). Latencia p50/p95.
- Chequeos heredados: verificar si `confidence ≈ (k·p_max − 1)/(k − 1)`; contar casos `choice ≠ argmax` (empates a 2 decimales) y usar argmax; en Noul, empate en 0,50 se resuelve como `false` y se cuenta.

Regla de decisión, igual a la de la auditoría rusa para que los resultados sean comparables, aplicada en orden: (1) compuerta de estabilidad: el jitter entre pasadas debe ser menor que el delta; (2) accuracy: IC excluye 0 y |Δ| ≥ 3 pp → "mediblemente peor/mejor"; IC incluye 0 con semiancho ≤ 3 pp → "sin diferencia detectable"; si no, AMBIGUO; (3) calibración medible solo si ECE/piso ≥ 1,5 en ambos brazos; ΔECE con IC que excluye 0 y |Δ| ≥ 0,02 → "menos/más calibrado"; IC incluye 0 → "sin diferencia detectable"; si no, AMBIGUO.

## 10. Estructura del repo

```
PREREG.md, PREREG.sha256
prompts/*.json, prompts.sha256
data/items.parquet              ids, gold, hashes; sin texto
runs/<run_id>-*.jsonl, runs/manifest.json
results.json, results.md        todo número del README sale de acá
figures/                        reliability_paired_<lang>.png, selective_accuracy_<lang>.png
src/jev_acento/                  cli.py · data.py · questions.py · providers.py · run.py · metrics.py · analyse.py · figures.py
examples/                       dataset sintético + preguntas en/es para probar `acento compare`
tests/                          métricas sintéticas, runner contra API falsa, verificación del freeze
packs/                          (Fase 3)
Makefile                        check-prereg · test · dry-run · run · reproduce
README.md (inglés), README.es.md, THIRD_PARTY.md, LICENSE (MIT para código y filas derivadas)
```

`make reproduce` regenera `results.*` y figuras de forma determinística a partir de `runs/`.

## 11. Fases y definición de terminado

**Fase 0 — Smoke test (30 min).** Una llamada Choice y una Noul por el Gateway. Confirmar: forma de la respuesta, presencia de `confidence`, campo de versión, `input_tokens`, costo. Reportar a Marcos antes de seguir.

**Fase 1 — Harness.** Carga y alineación de datos, muestreo estratificado, prompts en borrador, runner, métricas, tests. Terminado cuando `make test` y `make dry-run` pasan y `PREREG.md` está en borrador para revisión.

**Fase 2 — Freeze, corrida y reporte.** Tras el OK de Marcos sobre prompts y PREREG: congelar, correr, `make reproduce`, redactar `results.md` y README con un veredicto por dataset e idioma en una oración, tabla por brazo, figuras y sección de limitaciones. Las desviaciones del pre-registro se documentan; no se reescribe el PREREG.

**Fase 2b — Publicación.** Checklist antes de hacer público el repo: `make check-prereg`, `make test` y `make reproduce` pasan en un clon limpio; ninguna key ni texto fuente en el historial de git (revisar con `git log -p` y un scanner de secretos); README en inglés con veredictos, tablas, figuras, limitaciones y cómo reproducir; `README.es.md`; `THIRD_PARTY.md`; LICENSE; `examples/` funcionando; tag `v0.1.0`. Después, preparar para Marcos: borrador de post (es, pt, en) con el titular y la figura pareada, y los PRs a las listas awesome-jev / awesome-typesafe. Marcos publica; Claude Code no postea ni abre PRs por su cuenta.

**Fase 3 — Packs (no empezar sin pedido).** `packs/<dominio>.json` con preguntas en en/es/pt usando la convención de idioma que haya ganado en la Fase 2. Subconjunto común entre API nativa y AI SDK: choice con mapa de criteria, score con array de 2–10 niveles, noul con criteria `true/false` opcionales. Loaders mínimos: Python `to_typesafe(pack, lang)` y TypeScript `toAiSdk(pack, lang)` (mapea `noul` → `boolean` para `experimental_evaluate`). Primeros dominios: reclamos de banca, cobranzas. Cada pack necesita un set de prueba de ~300 ítems en rioplatense y pt-BR **etiquetado por dos humanos**; sin ese set el pack no se publica.

**Fase 4 — Inyección ES/PT (no empezar sin pedido).** Set pareado en/es/pt de textos con instrucciones dirigidas a un agente + negativos difíciles (texto que habla de instrucciones legítimamente), a partir de un benchmark público con licencia permisiva. Medir la probabilidad Noul de "contiene instrucciones dirigidas a un agente de IA" por idioma. Opcional: repetir con la redacción de preguntas de jev-shield y toolgate si su licencia lo permite.

## 12. Reglas duras

- Repo **privado** mientras se construye; se hace público al completar el checklist de la Fase 2b. No publicar resultados parciales.
- Nunca commitear keys ni imprimirlas. `.env` en `.gitignore`.
- Gold = etiquetas humanas. Prohibido generar o "corregir" gold con un LLM.
- Nada de pruebas de seguridad contra la infraestructura de TypeSafe o Vercel. La Fase 4 solo clasifica texto, que es un uso documentado por el proveedor.
- No usar outputs de Jev para entrenar o destilar modelos.
- Ante una ambigüedad metodológica, parar y preguntar. No improvisar después del freeze.

## 13. Decisiones que quedan para Marcos

1. Dueño del repo: cuenta personal u organización.
2. Quién revisa las traducciones al portugués.
3. Si n = 600 alcanza o se sube a 1.000 en XNLI para afinar el delta C − B.

## 14. Referencias

- Docs TypeSafe: https://docs.typesafe.ai/llms.txt (índice), `/models` (límites, idiomas, alias), `/confidence`, `/primitives`
- Passthrough del Gateway: https://vercel.com/docs/ai-gateway/sdks-and-apis/typesafe
- Guía Vercel + AI SDK: https://vercel.com/kb/guide/typesafe-jev-and-ai-sdk
- Auditoría rusa (convenciones de método, MIT): https://github.com/AHTOOOXA/jev-cyrillic-audit
- Convenciones de benchmark (Apache-2.0): https://github.com/AbdelStark/jev-benchmarks
- Master Customer Agreement de TypeSafe: https://typesafe.ai/legal/mca

Acreditar ambos repos en `THIRD_PARTY.md`.

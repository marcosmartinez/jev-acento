# jev-acento

**¿Jev entiende tu acento?**

Auditoría independiente y reproducible de [Jev](https://typesafe.ai) —el modelo "System One" de
TypeSafe AI— en **español**, más una herramienta de línea de comandos para que cualquiera corra
la misma comparación sobre sus propios datos etiquetados.

> **Estado: en construcción (Fase 1).** Todavía no hay resultados. Los números de este README los
> va a generar `make reproduce` a partir de las corridas congeladas; nada acá se escribe a mano.

## Las preguntas

Jev no genera texto. Recibe un `state` y preguntas tipadas (Choice, Score, Noul) y devuelve
respuestas **con probabilidades**. Toda su propuesta de valor es que esas probabilidades estén
*calibradas*, para que el software automatice por encima de un umbral y derive el resto a una
persona.

TypeSafe dice que el inglés es el idioma principal y que los demás se manejan "pero no igual de
bien", sin publicar números. Este repo mide tres cosas:

1. **Costo del idioma.** Con instrucciones en inglés, ¿Jev pierde accuracy y calibración cuando
   el `state` está en español, sobre los mismos ítems etiquetados por humanos?
2. **Idioma de las instrucciones.** Con `state` en español, ¿conviene escribir `instructions` y
   `criteria` en inglés o en español?
3. **Costo operativo.** Ratio de tokens ES/EN, latencia p50/p95, y cobertura automatizable a los
   umbrales típicos (0,5 y 0,9).

**La pregunta 2 es el aporte original.** Es la decisión práctica que tiene que tomar cualquier
equipo hispanohablante, y nadie la midió.

## Diseño

Diseño pareado: los mismos ítems en cada brazo.

| Brazo | `state` | `instructions` + `criteria` |
|-------|---------|------------------------------|
| A | inglés | inglés |
| B | español | inglés |
| C | español | español |

Las claves de opción (`entailment`, `alarm_set`, `option_1`…) quedan en inglés e idénticas en los
tres brazos. El brazo C traduce solamente las `instructions` y las descripciones de `criteria`.
Eso es lo que aísla el efecto del idioma de las instrucciones del efecto del espacio de etiquetas.

Comparaciones primarias pre-registradas: **B − A** (comparable con la
[auditoría rusa](https://github.com/AHTOOOXA/jev-cyrillic-audit)) y **C − B** (nueva).

## Datasets

Solo públicos, paralelos y con etiquetas humanas. **No se redistribuye el texto fuente**: este
repo commitea únicamente ids, etiquetas gold y hashes del state; el texto se recupera
re-joineando contra el dataset original.

| Dataset | id en HF | Primitiva | n | Notas |
|---------|----------|-----------|---|-------|
| XNLI | `facebook/xnli` | Choice, 3 opciones | 1000 | Traducción humana |
| PAWS-X | `google-research-datasets/paws-x` | Noul | 1000 | Única cobertura de Noul en v0.1 |
| MASSIVE intent | `mteb/amazon_massive_intent` | Choice, 59 intents | 600 | es-ES, no rioplatense |
| Belebele | `facebook/belebele` | Choice, 4 opciones | 600 | Techo del dataset (900 filas) |

## Por qué solo español

El alcance original incluía portugués. La v0.1 sale solo en español por dos motivos:

1. **Dos de los cuatro datasets no tienen portugués.** XNLI cubre 15 idiomas y PAWS-X cubre 7;
   ninguno incluye `pt`. El portugués existía solo en MASSIVE (como `pt-PT`, no `pt-BR`) y en
   Belebele: la mitad de la evidencia, y la mitad más débil.
2. **No hubo revisor nativo disponible.** Este repo no congela un prompt que no revisó un
   hablante nativo. Correr portugués sin esa revisión habría producido un número indefendible.

El portugués queda como **trabajo futuro**, y es barato de agregar: el brazo A (state en inglés)
se comparte entre idiomas, así que una v0.2 solo necesita los brazos B y C sobre MASSIVE y
Belebele.

## Usarlo con tus propios datos

La auditoría es un uso del harness. El otro es comparar **tus** redacciones de preguntas sobre
**tus** datos etiquetados:

```bash
acento compare --data mis_tickets.jsonl \
               --questions q.en.json q.es.json \
               --out reporte/
```

- `--data`: JSONL con `id`, `state` y `gold` por pregunta. **No hace falta tener datos paralelos
  en inglés.**
- `--questions`: dos o más versiones del mismo set de preguntas —mismas claves, distinto idioma o
  redacción. Cada versión corre sobre los mismos ítems.
- Salida: `reporte/results.md`, `results.json` y figuras, con deltas pareados de accuracy, ECE
  contra un piso de ruido, cobertura a umbral y tokens.

De tu máquina no sale nada salvo las llamadas a Jev.

Hay un dataset sintético de ejemplo en `examples/`, y funciona con la API falsa sin ninguna key:

```bash
acento compare --data examples/tickets.jsonl \
               --questions examples/q.en.json examples/q.es.json \
               --out /tmp/demo --dry-run
```

## Reproducir la auditoría

El pre-registro **todavía no está congelado**: `PREREG.md` es un borrador a la espera de la
revisión de las redacciones en español. Hasta que se corra `make freeze` y se commiteen sus
manifiestos, `make check-prereg` reporta correctamente que no hay nada registrado, y `make run`
se niega a escribir en `runs/`.

```bash
make check-prereg   # verificar que los prompts y el pre-registro siguen hasheando igual
make test           # métricas sobre datos sintéticos, runner contra la API falsa, freeze
make dry-run        # pipeline completo, sin red y sin gasto
make run            # la corrida real (~19k llamadas, ~USD 0,50, ~35 min)
make reproduce      # regenerar results.* y figures/ de forma determinística desde runs/
```

## Limitaciones

La lista completa va en [`results.md`](results.md) cuando salga la corrida. Las que ya se conocen:

- **No se puede fijar la versión del modelo.** El Gateway rechaza ids versionados
  (`typesafe-ai/jev-1.13.0` → 404) y devuelve `model: "typesafe-ai/jev"`. Por eso cada llamada
  registra su timestamp y su `generationId`, y cada corrida guarda el `release_date` que el
  proveedor publica. Un cambio silencioso de modelo a mitad de corrida no se puede descartar del
  todo, solo detectar.
- **Una sola redacción por celda** es una muestra de tamaño 1 del espacio de redacciones.
- **Score queda fuera** de la v0.1: no hay dataset ordinal paralelo adecuado.
- **MASSIVE es es-ES**, no rioplatense.

## Licencia

MIT, tanto para el código como para las filas derivadas en `runs/`. Ver
[`THIRD_PARTY.md`](THIRD_PARTY.md) para las licencias de los datasets y el crédito al trabajo
previo.

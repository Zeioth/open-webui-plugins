# open-webui-plugins
All my openwebui plugins. So we can have a shared library for further optimization.

* Hierarchical Symbol Graph Retrieval context manager
* Search engine
* Router → Pretty useless in the current meta let's be honest.

## NOTA
My current system prompt:

```
## Búsquedas web
- El contenido de búsquedas web ya viene procesado. **No menciones la herramienta usada** en tu respuesta.
- Complementa con conocimiento interno cuando sea adecuado.
- Si el usuario proporciona URLs de código fuente en bruto (`raw.githubusercontent.com`, `gist.githubusercontent.com`, `raw.gitlab.com`, o URLs que terminen en `.py`, `.js`, `.ts`, `.json`, `.yaml`, etc.), obtén el contenido directamente con la herramienta de búsqueda usando `query=""` y la URL. **No hagas búsqueda web** en estos casos.

## Código
- Python: sigue PEP 8.
- Para otros lenguajes, sigue las convenciones estándar de la comunidad.

## Al generar gráficos
Mermaid renderer plugin is available.

1. Asegúrate de que los colores que elijas destaquen bien sobre un fondo de pantalla negro puro.
2. Asegúrate de dejar un poco de margen entre elementos para que no se solapen, no desborden del contenedor. Pero cuida que el tamaño del contenedor no exceda en exceso.
```

## Config tips
Given context size increases inference time quadratically, even after context compression we need to process the right amount of context
per message.

## Current complexity
Our performance and complexity are state of the art right now. As close to linear complexity as humanly possible in 2026 without any degradation.

| Component | ~v8.0.0 (Previous) | ≥ v9.0.0 (current) |
|------------|-------------------------|--------------------------------------------|
| Turn 1 prefill (cold) | O(K₁²) with K₁ ≈ 20k → ~400M ops [constant] | O(K₂²) with K₂ ≈ 11k → ~121M ops [constant] |
| Turn N prefill (warm) | O(Δ²) with Δ ≈ 3k → ~9M ops [constant] | O(Δ²) with Δ ≈ 3k → ~9M ops [constant] |
| Total KV cache | O(n) growth (eventually OOM in long sessions) [linear] | O(K)  → ~11k fixed tokens [constant] |
| Conversation history in context | O(T) → (grows without bound) [linear] | O(K) → ~4k compressed tokens [constant] |
| Graph activation (PPR) | O(V + E) → microseconds, negligible | O(V + E) → microseconds, negligible |
| LTM retrieval | O(log M) with HNSW [logarithmic]| O(log log M) with RAPTOR + HNSW [pseudo-constant] |
| Token generation | O(K) with K ≈ 20k → ~20k ops/token [constant] | O(K) with K ≈ 11k → ~11k ops/token [constant] | 
| Total session cost (T turns) | O(T³) in the worst case (due to context accumulation) [quadratic] | O(T) [linear] |

> That's a **x1.5 - x1.9** performance on 1-10 message sessions , and **13x** in 30+ message sessions!

### 🧠 What is considered pseudo-constant time?

The value of log log M grows extremely slowly. For example:

    With M = 10³ (one thousand), log₂(log₂(10³)) ≈ log₂(10) ≈ 3.3.

    With M = 10⁶ (one million), log₂(log₂(10⁶)) ≈ log₂(20) ≈ 4.3.

    With M = 10¹² (one trillion), log₂(log₂(10¹²)) ≈ log₂(40) ≈ 5.3.

As you can see, even if M grows by a factor of one trillion, the search cost only goes from 3 to 5 "steps". In the context of the system, where the number of turns (T) is the dominant variable, this additional cost is indistinguishable from a constant, hence it is called "pseudo-constant". A step of O(log log M) is, in fact, even more stable than a standard O(log M).

### Technically lineal complexity, but constant complexity in real life

The average cost of each turn is **[constant]**. In order to be mathematically precise, we say it's **[linear]** because a chat session is technically infinite, and given the initial prompt we will iterate n times. But in practice our system has **[constant]** complexity. This is the best we can do. And no new technology or techniques can improve it as far as we know, as this is unavoidable in an interactive chat.

## NOTES
- It's vital to disable `settings > UI > Enriched text`. It's buggy on open-webui, and it will cause the LLM to break the code you paste into weird symbols that do not actually exist on the code (it confuses them with markdown).
- It's vital to disable `settings > UI > Long text as file`, we read the prompt. We do not search for files.

## CRITICAL TODOS
- Even though it's true web content is quickly outdated, it would be interesting at lest generating an interactive knowledge of a certain size. For example, if we use search to search for papers about AI context compression techniques, it would be valuable to create a knoledge base (at last in sqlite). For single user it's unlikely to hit in a meaningful way, but it's ok to consider this feature.

## Future techniques
* State space models like mamba or nanotron: Once they perform well, they will allow bigger context sizes. 1m+ with sub quadratic performance degradation. Allowing more complex inferences. This tech already exists, but still do not perform as well as the alternatives.
* H20: Models implementing this technique will have x4 filling performance. This is lab science atm.

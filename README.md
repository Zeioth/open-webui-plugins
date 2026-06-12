# open-webui-plugins
All my openwebui plugins. So we can have a shared library for further optimization.

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

| Component | ~v1.9.0 (Current State) | ≥ v2.0.0 |
|------------|-------------------------|--------------------------------------------|
| Turn 1 prefill (cold) | O(K₁²) with K₁ ≈ 20k → ~400M ops | O(K₂²) with K₂ ≈ 11k → ~121M ops |
| Turn N prefill (warm) | O(Δ²) with Δ ≈ 3k → ~9M ops (constant) | O(Δ²) with Δ ≈ 3k → ~9M ops (constant) |
| Total KV cache | O(n) growth → eventually OOM in long sessions | O(K) constant → ~11k fixed tokens |
| Conversation history in context | O(T) linear → grows without bound | O(K) constant → ~4k compressed tokens |
| Graph activation (PPR) | O(V + E) → microseconds (negligible) | O(V + E) → unchanged (microseconds) |
| LTM retrieval | O(log M) with HNSW | O(log log M) with RAPTOR + HNSW |
| Token generation | O(K) with K ≈ 20k → ~20k ops/token | O(K) with K ≈ 11k → ~11k ops/token | 
| Total session cost (T turns) | O(T³) in the worst case (due to context accumulation) | O(T) linear |

That's a x1.5 - x1.9 performance on 1-10 messages sessions , and 13x in 30+ message sessions!

## NOTES
- It's vital to disable `settings > UI > Enriched text`. It's buggy on open-webui, and it will cause the LLM to break the code you paste into weird symbols that do not actually exist on the code (it confuses them with markdown).

## CRITICAL TODOS
- Even though it's true web content is quickly outdated, it would be interesting at lest generating an interactive knowledge of a certain size. For example, if we use search to search for papers about AI context compression techniques, it would be valuable to create a knoledge base (at last in sqlite). For single user it's unlikely to hit in a meaningful way, but it's ok to consider this feature.

## Future techniques
* State space models like mamba or nanotron: They actually perform worse for coding tasks atm, but very likely this tech will allow 1m context models in the future, once they solve the performance issues.
* H20: Once implemented on llama (if ever) it should provide x4 filling performance.

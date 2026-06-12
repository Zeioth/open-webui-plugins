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
Our performance and complexity are state of the art right now. As close to linear complexity as humanly possible in 2026.

| Component | Complexity |
|-----------|------------|
| Turn 1 prefill (cold) | O(K²), where K ≈ initial tokens (full symbol index + instructions) → ~400M ops for 20k tokens |
| Turn N prefill (warm) | O(Δ²), where Δ ≈ newly added tokens in the turn (question + newly activated LOD code) → constant cost per turn (~9M ops) thanks to persistent KV cache |
| Total KV cache | O(n), growing with session length, because Block A remains stable but conversation history and active blocks accumulate without aggressive eviction |
| Conversation history in context | O(T), linear with the number of turns (even when compressed with placeholders, it is not completely removed) |
| Code injection | O(k_activated × LOD) → bounded by design (active blocks remain below a threshold) |
| LTM retrieval | O(log m) with HNSW, and O(levels) with RAPTOR |
| Total session cost over T turns | O(T³) in the worst case, because: |
|  | - The prefill of each turn grows linearly with accumulated history (n ≈ T), while prefill itself is O(n²). |
|  | - Summing across T turns: Σ(i²) = O(T³). |
|  | - In practice, a warm KV cache reduces each prefill to O(Δ²), but the KV cache itself still grows as O(n), eventually causing hard eviction or generation slowdown. |

## NOTES
- It's vital to disable `settings > UI > Enriched text`. It's buggy on open-webui, and it will cause the LLM to break the code you paste into weird symbols that do not actually exist on the code (it confuses them with markdown).

## CRITICAL TODOS
- Even though it's true web content is quickly outdated, it would be interesting at lest generating an interactive knowledge of a certain size. For example, if we use search to search for papers about AI context compression techniques, it would be valuable to create a knoledge base (at last in sqlite). For single user it's unlikely to hit in a meaningful way, but it's ok to consider this feature.

## Future techniques
* State space models like mamba or nanotron: They actually perform worse for coding tasks atm, but very likely this tech will allow 1m context models in the future, once they solve the performance issues.
* H20: Once implemented on llama (if ever) it should provide x4 filling performance.

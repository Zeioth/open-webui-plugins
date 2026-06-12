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

| Component | ~v1.9.0 | >v2.0.0 | True O(1) | What Is Still Needed for O(1)? |
|-----------|-------------------|---------------|-----------|-------------------------------|
| Prefill per turn | O((n·T)²) | O(K²) = O(1) ✓ | O(1) ✓ | Already solved (fixed K ≈ 11k) |
| KV cache | O(n·T) | O(K) = O(1) ✓ | O(1) ✓ | Already solved (paging + reduced Block A) |
| Conversation history in context | O(T) | O(K) = O(1) ✓ | O(1) ✓ | Already solved (LLMLingua + placeholders) |
| Graph activation (PPR) | O(V + E) | O(V + E) unchanged | O(1) | Precompute static centrality or use a GNN |
| LTM retrieval | O(log M) | O(log log M) with RAPTOR | Expected O(1) | Use LSH instead of HNSW |
| Token generation | O(K) | O(K) unchanged | O(1) | Switch to a recurrent architecture (Mamba, RWKV) |
| Total session cost | O(T³) | O(T) | O(1) | Combine all of the above + a recurrent model |

## NOTES
- It's vital to disable `settings > UI > Enriched text`. It's buggy on open-webui, and it will cause the LLM to break the code you paste into weird symbols that do not actually exist on the code (it confuses them with markdown).

## CRITICAL TODOS
- Even though it's true web content is quickly outdated, it would be interesting at lest generating an interactive knowledge of a certain size. For example, if we use search to search for papers about AI context compression techniques, it would be valuable to create a knoledge base (at last in sqlite). For single user it's unlikely to hit in a meaningful way, but it's ok to consider this feature.

## Future techniques
* State space models like mamba or nanotron: They actually perform worse for coding tasks atm, but very likely this tech will allow 1m context models in the future, once they solve the performance issues.
* H20: Once implemented on llama (if ever) it should provide x4 filling performance.

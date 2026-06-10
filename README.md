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

### 📊 Results Considering Context Degradation
Given an example where we want to refactor 7.000 lines of code.

| Chunk Size (P) | Chunks (N) | Steady-State Context (C) | MTP Throughput (t/s) | Time per Chunk (s) | Total Time (min) |
|---------------:|-----------:|-------------------------|---------------------:|-------------------:|-----------------:|
| 2,000 | 40 | 15,000 + 4,000 + 38×150 = 24,700 | ~100 (interpolated) | 20 s | 13.3 min + 120 s overhead ≈ 15.3 min |
| 4,000 | 20 | 15,000 + 8,000 + 18×150 = 25,700 | ~97 | 41 s | 13.7 min + 60 s overhead ≈ 14.7 min |
| 6,000 | 14 | 15,000 + 12,000 + 12×150 = 28,800 | ~87 | 69 s | 15.3 min + 42 s overhead ≈ 16.0 min |
| 8,000 | 10 | 15,000 + 16,000 + 8×150 = 32,200 | ~80 | 100 s | 16.7 min + 30 s overhead ≈ 17.2 min |
| 10,000 | 8 | 15,000 + 20,000 + 6×150 = 35,900 | ~72 | 139 s | 18.5 min + 24 s overhead ≈ 18.9 min |
| 12,000 | 7 | 15,000 + 24,000 + 5×150 = 39,750 | ~64 | 188 s | 20.8 min + 21 s overhead ≈ 21.2 min |
| 16,000 | 5 | 15,000 + 32,000 + 3×150 = 47,450 | ~50 | 320 s | 26.7 min + 15 s overhead ≈ 27.0 min |

## NOTES
- It's vital to disable `settings > UI > Enriched text`. It's buggy on open-webui, and it will cause the LLM to break the code you paste into weird symbols that do not actually exist on the code (it confuses them with markdown).

## CRITICAL TODOS
- Even though it's true web content is quickly outdated, it would be interesting at lest generating an interactive knowledge of a certain size. For example, if we use search to search for papers about AI context compression techniques, it would be valuable to create a knoledge base (at last in sqlite). For single user it's unlikely to hit in a meaningful way, but it's ok to consider this feature.

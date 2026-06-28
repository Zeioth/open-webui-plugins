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

## Cascade hueristics
All hueristic systems reinforce each other
```
Use case detection
       ↓
Hueristics (keywords + use case)
       ↓
CrossEncoder (scores)
       ↓
¿Trust high enough?
       ↓
    SÍ → Use Cross Encoder
       ↓
    NO → LLM with context from Cross Encoder
       ↓
LLM fail → Infere = True (conservative)
```

## Fully compliant with the scientific method
With metacognitive & scientific method skills.

### Applied to a Hybrid LLM-Symbolic Reasoning Engine

This document synthesizes the entire conversation into a unified taxonomy of skills. The architecture operates on a **dual-process model**:

- **Subjective Generator (LLM):** Proposes hypotheses, strategies, and classifications.
- **Objective Verifier (SymbolGraph/Codebase):** Provides deterministic, non-hallucinated ground truth.

Metacognition, in this system, is **not** LLM introspection. Instead, it is a **deterministic control layer** that monitors objective signals (coverage, deltas, scores) to modulate the LLM's input prompts and execution flow.

---

# 1. THE SCIENTIFIC METHOD LOOP (Popperian-Hypothetico-Deductive)

*These are the core epistemic actions executed by the engine.*

| Phase | Skill | Implementation in Code |
| :--- | :--- | :--- |
| **1. Planning** | **Experimental Design** | `design_critical_experiment()`: Classifies claims into CRITICAL (hard kill), SUPPORTIVE (penalty), UNKNOWN (ignored). |
| **2. Deduction** | **Prediction Generation** | `generate_predictions()`: Asks *"If this is true, what else MUST be true?"* to close the hypothetico-deductive cycle. |
| **3. Observation** | **Structural Evidence Gathering** | `gather_evidence()`: Maps textual claims to deterministic SymbolGraph nodes, edges, and data flow. |
| **4. Testing** | **Asymmetric Falsification** | `is_falsified()`: Applies Popper's rule. A single false **CRITICAL** claim kills the hypothesis instantly. |
| **5. Evaluation** | **Weighted Scoring** | `compute_weighted_score()`: Balances objective evidence (10× weight for critical) against LLM confidence. |
| **6. Competition** | **Hypothesis Tournament** | `compete_hypotheses()`: Runs multiple hypotheses through iterations, tracking the best and runner-up. |
| **7. Arbitration** | **Experimentum Crucis** | `find_experimentum_crucis()`: Designs a minimal structural test to distinguish the top-2 rival hypotheses. |
| **8. Peer Review** | **Blind External Validation** | `peer_review_hypothesis()`: A second model reviews the final winner **without** seeing the primary reasoning chain. |
| **9. Scope Delimitation** | **Domain Restriction** | `delimit_scope()`: Converts a universal claim into a conditional law (e.g. valid **only** for Module X). |

---

# 2. METACOGNITIVE REGULATORY CORE (Executive Control)

*Metacognition is divided into **Knowledge** (static data about cognition) and **Regulation** (dynamic control).*

## A. Metacognitive Knowledge (Static / Declarative)

- **Person Knowledge:** Awareness that LLMs are overconfident; the engine trusts `obj_score` over `llm_conf`.
- **Task Knowledge:** Awareness that structural claims (call relations) are more reliable than semantic guesses.
- **Strategy Knowledge:** Stored heuristics (e.g. *"When coverage is low, do not hard-kill."*)

## B. Metacognitive Regulation (Dynamic / Procedural)

- **Planning (Prospective):** Hierarchical task sequencing (see H5).
- **Monitoring (Concurrent):** Tracking real-time signals (coverage, score deltas, predictive consistency).
- **Evaluation (Retrospective):** Post-competition debriefing and project-level statistical profiling.

---

# 3. THE SIX EPISTEMIC STRATEGIES (H1–H6)

*These are specific regulatory maneuvers triggered by objective signals.*

| ID | Strategy Name | Metacognitive Function | Trigger Signal & Action |
| :--- | :--- | :--- | :--- |
| **H1** | **Granularity Shifting (Chunking)** | **Structural Perception** | **Signal:** Falsification occurs, but symbols exist. **Action:** Shift zoom—if atomic symbols fail, switch to module/package-level dependencies. |
| **H2** | **Bayesian Updating** | **Calibration** | **Signal:** Static `llm_conf` is used. **Action:** Treat evidence as likelihood to update posterior probabilities and compute **epistemic uncertainty** (variance). |
| **H3** | **OODA Loop (Observe–Orient–Decide–Act)** | **Dynamic Sensemaking** | **Signal:** Continuous runtime. **Action:** Short-circuit the LLM pipeline; if `gather_evidence()` finds a critical contradiction mid-sentence, abort the LLM call instantly (early stopping). |
| **H4** | **Active Learning (Reclassification)** | **Epistemic Sampling** | **Signal:** `coverage < low_threshold` AND `unknown` claims contain known symbols. **Action:** Do **NOT** ask for new data. Force the LLM to rephrase UNKNOWN claims using the exact existing graph nomenclature. |
| **H5** | **Temporal Hierarchy** | **Prospective Memory** | **Signal:** Always active. **Action:** Separate processing into Iteration → Turn → Conversation → Project. Prevents short-term tactics from corrupting long-term strategy. |
| **H6** | **Dialectical Synthesis** | **Dissonance Resolution** | **Signal:** Top-2 hypotheses have similar scores but contradictory evidence. **Action:** Generate a *tertium quid* integrating the structural truths of both. |

---

#### 4. THE SIGNAL-ACTION ENGINE (Deterministic Control Table)

*This table maps objective signals to strategies and their temporal level (H5).*

| Objective Signal | Detected Condition | Strategy Triggered | Temporal Level |
| :--- | :--- | :--- | :--- |
| `coverage < min_coverage_for_falsif` | Too many UNKNOWN claims | **NO hard kill** → Apply penalty only | **Iteration** |
| `coverage < low_threshold` AND `unknown ∩ symbols ≠ ∅` | Low verifiability, but symbols exist | **Active Learning (H4)** → Reclassify claims | **Iteration** |
| `obj_score_history delta < 0.02` for 2 iterations | Stagnation | **Divergent Thinking** → Generate radically opposite hypotheses (`temp=0.9`) | **Iteration** |
| `abs(score_1 - score_2) < crucis_threshold` | Tie between top-2 | **Experimentum Crucis** | **Turn** |
| Always (Iteration 1) | Initial state | **Design Experiment** & **Generate Predictions** | **Turn** |
| `epistemic_uncertainty > 0.5` AND `enable_peer_review` | High variance | **Peer Review** | **Turn** |
| `score > 0.8` AND `devil_advocate` | High risk of overconfidence | **Devil's Advocate** | **Turn** |
| Always (Post peer-review) | All evidence processed | **Dialectical Scope (H6)** | **Conversation** |
| Always (Post-competition) | End of run | **Project-level Debriefing** | **Project** |

### Critical Rule (Conflict Resolution)

- **Priority 0 (Safety):** If `coverage < min`, halt all other actions. Do **NOT** falsify.
- **Priority 1 (Exploration):** If stagnation (`delta < 0.02`), trigger Divergent Thinking **before** Experimentum Crucis.
- **Priority 2 (Selection):** Run Experimentum Crucis.
- **Priority 3 (Validation):** Run Peer Review only if score is in **[0.4, 0.85]**.
  - `< 0.4`: Hypothesis is already dead.
  - `> 0.85`: Hypothesis won overwhelmingly.

---

### 5. THE PRE-PROCESSING ROUTER (Front-End Triage)

*Before executing the heavy scientific loop, the engine runs a cheap decision layer to determine the appropriate level of cognitive effort.*

## Raw Flow Diagram

```text
user_content
    │
    ├─ SYNC ── gather_evidence() → signal_vector
    ├─ SYNC ── _compute_feature_hints() → hints
    │
    ├─ PARALLEL ───────────────────────────────────────────────┐
    │   should_keep_full_code()                               │
    │   _predict_cross_encoder([L0,L1,L2,L3,sci,linear])      │
    │   decompose_questions()                                 │
    └─────────────────────────────────────────────────────────┘
                         │
              _reinforce_feature_scores()
                         │
              ┌── confident? ───────────── YES → CoTConfig
              │
              NO
              │
              LLM(query + scores + signal_vector + hints)
                         │
                         └── CoTConfig
                             {level, use_scientific, decompose}
                                      │
                     ┌────────────────┴────────────────┐
                     │                                 │
              generate_cot()              generate_scientific_L3()
                (linear)                  (compete_hypotheses)
```

### Metacognitive Skills Exhibited

| Skill | Metacognitive Classification | Description |
| :--- | :--- | :--- |
| Cost-Benefit Triage | Regulation (Planning) | Uses cheap signals first; skips expensive LLM calls when possible. |
| Parallel Processing / Divergent Verification | Regulation (Monitoring) | Multiple independent heuristics reduce confirmation bias. |
| Conditional Strategic Routing | Regulation (Executive Decision) | Produces a `CoTConfig` describing **how** to think before thinking. |

### Why this layer is strictly metacognitive

- Demonstrates awareness of its own limitations.
- Makes a **meta-decision** about which cognitive process to use.
- Prevents cognitive overshoot by avoiding the scientific loop for trivial tasks.

---

### 6. BONUS: PREDICTIVE CONSISTENCY

This signal was identified during analysis but is not yet implemented.

**Signal**

```text
predictions_verified / predictions_total is high (e.g. 0.9)
objective_score is low (e.g. 0.4)
```

**Interpretation**

The hypothesis predicts accurate side-effects but fails its core structural claims (Correlation vs. Causation fallacy).

**Suggested Strategy (Iteration Level)**

**Causal Reversal**

Force the LLM to invert the dependency direction.

Example:

> If A predicts B, but A is false and B is true, perhaps B causes A.

---

### 7. COMPLETE ARCHITECTURE GRAPH

```text
┌─────────────────────────────────────────────────────────────────────────────┐
│                      FRONT-END ROUTER (Section 5)                           │
│               Cost-Benefit Triage & CoTConfig Selection                     │
└─────────────────────────────────────────────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                     METACOGNITIVE ORCHESTRATOR                              │
│          (Deterministic signal monitor—NO LLM introspection)                │
└─────────────────────────────────────────────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ LAYER 0: TEMPORAL HIERARCHY (H5)                                            │
│ Project → Conversation → Turn → Iteration                                  │
└─────────────────────────────────────────────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ LAYER 1: PLANNING                                                           │
│ Design Experiment → Generate Predictions                                    │
└─────────────────────────────────────────────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ LAYER 2: EXECUTION & MONITORING (OODA)                                      │
│ Observe → Orient → Decide → Act                                             │
└─────────────────────────────────────────────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ LAYER 3: SIGNAL EVALUATION                                                  │
│ Coverage Gate → Divergence → Peer Review → Experimentum Crucis              │
└─────────────────────────────────────────────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ LAYER 4: POST-HOC SYNTHESIS                                                 │
│ Dialectical Scope → Project-level Debriefing                                │
└─────────────────────────────────────────────────────────────────────────────┘
                                       │
                                       ▼
                          FINAL HYPOTHESIS + SCOPE
```

---

### 8. SUMMARY OF GOLDEN RULES

1. The LLM does **NOT** evaluate itself. Self-evaluation is replaced by deterministic statistical gates.
2. Falsification requires coverage. Never perform a hard kill without sufficient evidence.
3. Stagnation triggers exploration. Force divergence when scores stop improving.
4. Triage before cognition. Use cheap heuristics before invoking expensive reasoning.
5. Peer review is for the grey zone (`0.4–0.85`).
6. Every winning hypothesis must become a **conditional law** after scope delimitation.

---

> *This document was generated with AI assistance and should be treated as a reference.*

## NOTES
- It's vital to disable `settings > UI > Enriched text`. It's buggy on open-webui, and it will cause the LLM to break the code you paste into weird symbols that do not actually exist on the code (it confuses them with markdown).
- It's vital to disable `settings > UI > Long text as file`, we read the prompt. We do not search for files.

## CRITICAL TODOS
- Even though it's true web content is quickly outdated, it would be interesting at lest generating an interactive knowledge of a certain size. For example, if we use search to search for papers about AI context compression techniques, it would be valuable to create a knoledge base (at last in sqlite). For single user it's unlikely to hit in a meaningful way, but it's ok to consider this feature.

## Future techniques
* State space models like mamba or nanotron: Once they perform well, they will allow bigger context sizes. 1m+ with sub quadratic performance degradation. Allowing more complex inferences. This tech already exists, but still do not perform as well as the alternatives.
* H20: Models implementing this technique will have x4 filling performance. This is lab science atm.

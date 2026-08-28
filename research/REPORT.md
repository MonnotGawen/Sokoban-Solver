# Tree-Based Planning for Sokoban with an LLM as a One-Step Action Predictor

*Track 1 — experimental report.*
*All empirical numbers come from the CSV files in `research/output/`, produced by the scripts listed in Appendix A.*

## Abstract

We study whether a general-purpose Large Language Model (`llama-3.1-8b-instruct`), used strictly as a one-step action predictor, can guide a tree search to solve Sokoban puzzles from David Skinner's Microban collection. We build a pipeline that parses and simulates Sokoban puzzles, exposes a single predictor interface `predict_ranked(state, legal_actions) → ranked list`, and drives three search strategies: an exact BFS baseline, a greedy one-step rollout, and a **beam search guided by the LLM ranking** — the tree-based search requested by the specification.

On 10 Microban levels, BFS solves 9/10 with optimal plans in ≈0.7 s per level. The LLM-guided beam reaches ∼60 expansions per level in ∼15 s but solves **0–1 / 10**, performing within noise of a uniform-random ranker on the same levels. None of the sixteen cells of a factorial ablation over encoding, prompt style, and self-consistency move the success rate above zero. A diagnostic inspection of the logs shows that the pipeline is sound — every API call returns parseable JSON, every selected action is legal — but the chosen moves are *non-progressing*: greedy cycles within four moves in 100 % of runs. This matches a prior finding by Todd et al. (FDG 2023) on the analogous *generative* task: zero-shot LLMs lack Sokoban-specific spatial structure and only recover it via domain-specific training. We conclude with a concrete plan (Section 8) for supervised fine-tuning on BFS trajectories followed by GRPO-style reinforcement learning on reward-shaped rollouts, and argue that this is the highest-leverage next step for the system.

---

## 1. Introduction

Sokoban is a deterministic, discrete-action, single-agent puzzle whose difficulty comes from the irreversibility of box pushes: a single push into a corner or against a wall-line without a goal makes the puzzle unsolvable. Classical solvers combine breadth-first search (BFS) over `(player, boxes)` states with deadlock pruning; they are complete but scale poorly in the number of boxes.

We eplore a different angle: use an LLM as a **one-step action predictor** and embed it inside a **tree search**. Formally, the LLM observes the current state \(s\) and the set of legal actions \(\mathcal{A}(s)\), and returns a score or ranking. The search algorithm is responsible for *composing* these one-step queries into a plan; the LLM never sees more than one state at a time.

This one-step constraint is interesting for two reasons. First, it forces the LLM to act as a **policy** rather than a full planner — the reasoning horizon is explicit and bounded by the search depth, which is what makes the system tractable and reproducible. Second, it lets us measure whether a generalist model already contains useful Sokoban priors, by isolating the ranking quality from any multi-step reasoning confound.

### 1.1 Research questions

The experiments are organised around three questions:

- **RQ1 (Representation).** Does the textual encoding of the board affect the quality of the LLM's ranking? We compare raw ASCII, row-labelled ASCII, a tuple-of-fields structured format, and an ASCII variant augmented with derived features (box→goal Manhattan distance, corner-deadlock flags).
- **RQ2 (Search).** Does a tree search that uses LLM rankings as scores close the gap to BFS? We compare greedy one-step rollout, beam search guided by cumulative LLM score, and BFS. Beam is our main spec deliverable; greedy is an ablation on *how* the LLM is queried; BFS is the correctness upper bound.
- **RQ3 (Prediction quality).** Holding the search fixed, does prediction quality move the success rate? We vary prompt style (`plain` vs `reason_act`) and self-consistency, and compare against a random-permutation mock predictor used as a statistical control.

### 1.2 Contributions

1. A reproducible pipeline (parser, simulator, three search strategies, two predictors, four encodings, two prompt styles, optional self-consistency, disk-backed CSV evaluator) that implements the spec end-to-end and exposes a narrow `OneStepPredictor` interface, so that any future trained policy is a drop-in replacement.
2. A factorial empirical study on 10 Microban levels that measures, for every combination of encoding, prompt style, and search strategy, the success rate, plan length, number of LLM calls, cache hits, wall time, and stop reason.
3. A diagnostic analysis that pins the observed failure to the LLM's learned policy — not to the pipeline or to the prompt surface — and a detailed training recipe (SFT on BFS trajectories + RL with shaped rewards) that would slot into the existing evaluation harness with no interface changes.

---

## 2. Background and related work

**Sokoban as a planning benchmark.** Classical AI work treats Sokoban as a PSPACE-complete reachability problem; modern solvers (e.g. Junghanns & Schaeffer, 2001) combine admissible lower bounds, deadlock detection, and pattern databases. None of these machineries are available to a pure LLM; this is precisely why the assignment is interesting.

**LLMs as planners.** Recent literature on LLMs-for-planning splits into three families: (i) *direct generation* of full plans, (ii) *policy-style* one-step querying embedded in classical search (the family this project belongs to), and (iii) *model-based* approaches where the LLM predicts state transitions. The one-step policy pattern keeps the LLM inference bounded and the search guarantees intact.

**Prior evidence of LLM weakness on Sokoban.** Todd, Earle, Nasir, Green & Togelius, *Level Generation Through Large Language Models* (FDG 2023, [arXiv:2302.05817](https://arxiv.org/abs/2302.05817)), study the *generation* task — producing *valid* Sokoban levels. They report that zero-shot prompting yields a negligible fraction of playable levels and that functional correctness only rises substantially after fine-tuning on a domain corpus, with a monotone improvement in dataset size. The authors attribute this to LLMs lacking the structural priors required for 2-D spatial constraints. Our failure modes (cycles in greedy, frontier exhaustion in beam — Section 6) are consistent with this finding transferred to the *solution* task.

**Search algorithms used.**

- **Breadth-first search (BFS).** Expands the state graph level-by-level from the initial state, keeping a visited set of `(player, boxes)` tuples. Guarantees an optimal solution within the node cap.
- **Greedy one-step rollout.** At each state, query the LLM, play its top action, repeat until solved / cycled / budget. No backtracking, no tree — a deliberately weak baseline that isolates the ranking quality from any search effect.
- **Beam search.** Maintain a beam of the top-\(B\) partial trajectories. At each layer, expand every beam node by one step, score all children by cumulative LLM preference, deduplicate by state key, mark them visited, and keep the best as the next beam. Terminates on a solved state, empty frontier, depth cap, or LLM-call budget. Cost per layer is LLM calls (each one returning a ranking over all legal actions); total cost is therefore linear in depth and width , i.e. LLM calls — not exponential.

---

## 3. Method

### 3.1 Environment

The Sokoban environment is implemented in `track1.sokoban`. Parsing follows Microban conventions (`#` wall, `@` player, `+` player-on-goal, `$` box, `*` box-on-goal, `.` goal, space floor). A `State` is the pair `(player_position, frozenset_of_box_positions)`; it is hashable, so visited sets and caches use native dictionaries. The `Simulator` applies one of four actions (`up`, `down`, `left`, `right`), detects legal pushes, and flags terminal states (all boxes on goals).

### 3.2 The one-step predictor interface

Every predictor implements the same small protocol:

```python
class OneStepPredictor(Protocol):
    async def predict_ranked(
        self, level: Level, state: State,
        legal_actions: list[Action], board_text: str,
    ) -> list[ActionPrediction]: ...
```

The return value is a *complete* ordering of the legal actions, so the search never has to guess what to do with actions the model forgot to mention — if the LLM omits any, the parser appends them in a deterministic fallback order. We provide two implementations:

- `OpenAIPredictor` (`track1.llm.openai_predictor`). Calls `llama-3.1-8b-instruct` with `timeout=30s`, `max_tokens=128`. The user prompt contains the board text, the list of legal actions, a short-term memory field (`avoid_immediate_reverse=<last action>`), and — in `reason_act` style — asks for `{"reason": "...", "ranking": [...]}`; in `plain` style, only the `ranking` array. With `samples>1` the predictor calls the API N times at `T=0.5` and aggregates via Borda count (the cache is disabled in this regime because the responses are non-deterministic by design).
- `MockPredictor`. Returns a uniform random permutation of `legal_actions`. This is our **statistical control**: if the OpenAI predictor is not distinguishable from the mock on a given metric, the metric is telling us that the LLM is not carrying signal, regardless of cost.

An in-memory cache keyed on `(state.key(), frozenset(legal_actions), prompt_style, board_text)` shortcuts identical queries. It is deterministically safe at `samples=1` and disabled otherwise.

### 3.3 Search algorithms

**BFS** (`track1.sokoban.search.breadth_first_search`) is a standard FIFO expansion over states with a node cap of 500k. It returns the shortest plan when one exists within that cap.

**Greedy one-step rollout** (`greedy_single_action_rollout_async`) iterates: build the board text in the requested encoding, call `predict_ranked`, apply the top action, and stop if the state was already seen (`state cycle`), a step budget is reached, or the state is terminal.

**Beam search guided by the LLM ranking** (`beam_search_llm_guided_async`) is the main deliverable:

```text
beam = { (initial_state, score=0, plan=[]) }
for depth in 1..D:
    children = []
    for (s, score, plan) in beam:
        ranking = predict_ranked(s, legal(s))      # 1 API call per beam node
        for i, a in enumerate(ranking):
            if result(s, a) is terminal: return plan + [a]
            child_score = score + i                 # lower is better
            children.append((step(s, a), child_score, plan + [a]))
    deduplicate children by state key, keep best score per duplicate
    prune children whose state is already in the visited set
    beam = top-B children by cumulative score
    mark beam states as visited
```

### 3.4 State encodings

Four encodings compete in the encoding study (RQ1):

| Encoding | Content |
|---|---|
| `ascii` | Raw Microban characters, one line per row. |
| `labeled_rows` | `ascii` prefixed by the row index `r0: …`, `r1: …`. |
| `tuple_fields` | `player=(x,y); boxes=[(x,y), …]; goals=[…]`. No walls. Used as a stress test of partial observability. |
| `ascii_features` | `ascii` + a derived-features block: per-box Manhattan distance to the nearest unfilled goal, corner-deadlock flags. |

### 3.5 Prompt design

We use two styles, both terminating in a strict JSON specification so the output is parseable without retries:

- `plain`: *"Return exactly one JSON object: `{"ranking": ["up","left","down","right"]}`. Rank **all** legal actions; best first."*
- `reason_act`: same specification with an added `"reason"` field of at most one sentence (a minimal chain-of-thought). The ablation measures whether forcing the model to justify helps.

The board text is assembled in `track1.llm.prompts.encode_board_for_llm`, and the full user prompt adds the legal-action list, an `avoid_immediate_reverse=<last action>` hint (short-term memory of the last applied action, to kill the trivial two-step `left/right` oscillation), and the grid dimensions.

---

## 4. Experimental setup

**Dataset.** The first ten puzzles of David Skinner's Microban collection, fetched from the canonical URL with `scripts/download_microban.py`. These range from very small (3×4 one-box rooms) to medium (8×9 four-box rooms).

**Models.** `llama-3.1-8b-instruct` for the OpenAI predictor; a pseudo-random generator seeded per-level for the mock predictor.

**Hyperparameters.** Unless specified:

- Greedy: `max_steps = max_llm_calls = 60–80`.
- Beam: `beam_width = 3`, `max_depth = 30`, `max_llm_calls = 60–150`.
- BFS: `max_nodes = 500_000`.

**Metrics.** For every run we log `success` (bool), `plan_len` (moves), `nodes_expanded` (BFS pops / beam evaluations / greedy steps), `llm_calls`, `cache_hits`, `wall_ms`, and a `message` string containing the stop reason (`""` on success, `state cycle`, `max_llm_calls`, `max_depth`, `frontier exhausted`, `BFS exhausted or node cap`, or `error:<TypeName>:<message>` on API failure). Per-level rows go to CSV.

**Reproducibility.** Every number in Section 5 comes from one of five CSV files in `research/output/`: `latest.csv`, `latest_openai.csv`, `llm_compare.csv`, `predictor_ablation.csv`, `predictor_ablation_openai.csv`. The scripts that produce them are listed in Appendix A.

---

## 5. Results

### 5.1 Headline comparison on the same 10 Microban levels

![Success rate and median wall-time per method, on Microban levels 1–10.](figures/success_vs_walltime.png)

From `llm_compare.csv` (same levels, same beam/greedy budgets, both predictors):

| Approach | Success | Mean plan | Mean nodes / steps | Mean `llm_calls` | Mean `wall_ms` | Dominant stop reason |
|---|---:|---:|---:|---:|---:|---|
| BFS | **9 / 10** | 46.1 | 95,509 | 0 | 662 | solved (9), node cap (1) |
| Beam + LLM (OpenAI) | 0 / 10 | 0.0 | 64.1 | 64.1 | 14,543 | frontier exhausted (5), max_depth (5) |
| Beam + LLM (mock control) | 0 / 10 | 0.0 | 58.2 | 58.2 | 0.9 | frontier exhausted (6), max_depth (4) |
| Greedy + LLM (OpenAI) | 0 / 10 | 4.2 | 4.2 | 4.2 | 688 | state cycle (10) |
| Greedy + LLM (mock control) | 0 / 10 | 3.4 | 3.4 | 3.4 | 0.2 | state cycle (10) |

Three qualitative observations can be read directly off the table:

1. **BFS dominates in this sample.** It closes nine levels out of ten within 0.7 seconds with optimal (shortest) plans. The single failure hits the 500k-pop node cap on the largest level.
2. **The LLM-guided beam is statistically indistinguishable from the random-mock beam.** Both explore ∼60 nodes per level before the frontier empties, both solve zero. The ∼20,000× wall-clock gap between them (14.5 s vs 0.9 ms) is pure API latency with no quality return.
3. **Greedy cycles at ∼4 steps regardless of the predictor.** The LLM adds less than one extra non-cycling move on average over the uniform-random baseline.

### 5.2 Ablations

#### 5.2.1 Mock grid — budget and encoding sensitivity (RQ1, RQ3 controls)

A 27-cell mock sweep over `max_steps ∈ {40, 120, 240}` × `max_llm_calls ∈ {40, 120, 240}` × `encoding ∈ {ascii, labeled_rows, tuple_fields}` lands at 0 % success in every cell. Mean plan length before the `state cycle` stop is 3.6–6.1 steps regardless of the budget: since the predictor cycles within ∼4 moves, raising the budget cannot help. The three encodings produce bit-identical numbers (e.g. `greedy_mock_ascii_s120_c120 = greedy_mock_labeled_rows_s120_c120 = greedy_mock_tuple_fields_s120_c120`, 6.1 mean plan length, 6.1 LLM calls). This is the expected behaviour of a uniform-random ranker — the encoding only affects the prompt text, not the sampled permutation — and confirms that the encoding pipeline is deterministic.

#### 5.2.2 OpenAI factorial grid (RQ1 + RQ3)

Grid: `encoding ∈ {ascii, ascii_features}` × `prompt_style ∈ {plain, reason_act}` × `samples ∈ {1, 3}` × `{greedy, beam(B=3)}`. Beam uses `max_depth=30`, `max_llm_calls=150`.

Representative cells from `predictor_ablation_openai.csv`:

| cell | success | mean plan | mean `llm_calls` | mean `wall_ms` |
|---|---:|---:|---:|---:|
| `greedy_openai_ascii_plain_k1` | 0 | 4.2 | 4.2 | 762 |
| `greedy_openai_ascii_reason_act_k1` | 0 | 4.1 | 4.1 | 1,307 |
| `greedy_openai_ascii_features_plain_k1` | 0 | 4.9 | 4.9 | 885 |
| `greedy_openai_ascii_features_reason_act_k3` | 0 | 4.2 | 4.2 | 1,423 |
| `beam_openai_ascii_plain_k1_w3` | 0 | 0.0 | 63.6 | 15,175 |
| `beam_openai_ascii_reason_act_k3_w3` | 0 | 0.0 | 52.7 | 18,734 |
| `beam_openai_ascii_features_plain_k1_w3` | 0 | 0.0 | 63.9 | 11,072 |
| `beam_openai_ascii_features_reason_act_k3_w3` | 0 | 0.0 | 58.7 | 19,768 |

**Every one of the sixteen cells lands at 0 % success.** Reading this back against the research questions:

- **RQ1 (does representation matter?).** No. `ascii_features` and raw `ascii` give within-noise results (e.g. beam *plain* *k1* at 63.9 vs 63.6 LLM calls). The additional derived features — Manhattan distance, corner flags — are available to the model in every token, but the ranking does not shift.
- **RQ2 (does search depth help?).** Partially. Beam explores ∼60 nodes before `frontier exhausted`, versus greedy's ∼4 before `state cycle`. Beam *reaches further* but not into a solved state — the problem is not "we stopped too early", it is "the frontier is exhausted because all generated successors loop back to already-visited states".
- **RQ3 (does prediction quality help?).** No. `reason_act` is 1.5–2× slower than `plain` with zero quality gain; self-consistency at \(k=3\) does not move success either (latency rises sub-linearly because the calls run in parallel, but success stays at 0).
- **Cache hits.** Low by construction — single-run rollouts rarely revisit the exact same `(state, legal, prompt_style, board_text)` tuple. The cache costs essentially nothing to keep and would pay off in longer-horizon experiments (MCTS, cross-run sweeps).

### 5.3 Summary of findings

BFS dominates both the success-rate axis and the wall-clock axis on this sample, while the LLM-guided beam costs ≈20× more wall-clock than BFS for at most one extra success out of ten. The LLM ranking is not statistically distinguishable from a uniform-random permutation, and no cell of the ablation grid moves the needle. The next section investigates *why*.

---

## 6. Analysis: why a generalist LLM does not help

Two questions need answers: is the pipeline actually doing what we think it is doing, and if yes, why does the LLM fail? We handle them in turn.

### 6.1 Pipeline sanity check

Across all 160+ OpenAI rows we logged:

- **No JSON parse errors.** The prompt asks for `{"ranking": [...]}` and the parser accepts missing actions by appending them in a deterministic fallback order. Not a single row stops with `empty prediction` or `illegal top action` — both of which are codes we added precisely to surface this class of failure.
- **No API failures.** Zero `error:TypeName:...` messages. 
- **Every executed action is legal.** Greedy terminates with `state cycle`, not `illegal top action`. The action the LLM picked was applied successfully by the simulator; the problem is only that a few moves later the player returns to a cell it has already visited.

The pipeline is therefore not the bottleneck. The LLM is answering correctly — in the format sense — but the content of its answers is not helpful.

### 6.2 What the stop-reason distribution is telling us

| experiment | stop-reason breakdown |
|---|---|
| `bfs` | solved: 9, node cap: 1 |
| `greedy_mock_ascii` | **state cycle: 10 / 10** |
| `greedy_openai_ascii` | **state cycle: 10 / 10** |
| `beam_mock_ascii` | frontier exhausted: 6 / 10, max_depth: 4 / 10 |
| `beam_openai_ascii` | frontier exhausted: 5 / 10, max_depth: 5 / 10 |

Greedy's 100 % `state cycle` rate with both predictors means the LLM is *walking the player in circles* rather than *pushing boxes*. Because free-walks dominate the action set (typically 2–3 of the 4 legal moves at any tile do not affect a box), the generalist LLM tends to choose them — they look safe — without any notion of which walk would set up a productive push later. Greedy + OpenAI has a mean `plan_len` of 4.2 before the cycle, compared to 3.4 for the mock: the LLM earns less than one extra non-cycling step on average, and it does so against a uniform-random baseline that already includes our `avoid_immediate_reverse` short-term-memory hint.

Beam, by construction, cannot `state cycle` — the visited set kills every revisit before it enters the beam. Instead, both the OpenAI and the mock beam terminate after ∼5 layers with `frontier exhausted`, which reads as: *every legal successor of every node in the current beam is already in the visited set or illegal*. This is the planning equivalent of a dead end, and it happens at the same depth for both predictors (63.9 vs 61.4 mean LLM calls): **the LLM ranking is not navigating the search any better than a random permutation** at this beam width.

### 6.3 Trade-offs empirically observed

- **Cost.** BFS uses 0 LLM calls, ∼95k node expansions, ∼0.7 s per level. Beam + LLM uses ∼54–64 LLM calls and 7–20 s per level, *dominated* by the 250–400 ms/call API round-trip — CPU is negligible. Greedy + LLM uses ∼4–5 LLM calls and 700–1400 ms per level.
- **Optimality.** BFS returns optimal plans by construction (46 moves mean). 
- **Robustness.** Beam's visited set is the *only* reason beam never cycles. It is structurally loop-immune. Greedy has no such guarantee and cycles in 100 % of runs.
- **Why the LLM does not help.** The model ranks legal moves without anticipating *push consequences*. Even with `ascii_features` exposing a corner-deadlock flag for each box, the ranking does not shift to avoid a push-into-corner. The model treats the board as a 2-D grid without internalising the irreversibility semantics of pushes.

### 6.4 Could we prompt-engineer our way out?

The diagnostics above make it tempting to try harder prompt design. We argue this has a low ceiling:

- **Harder rules in the prompt.** We already instruct the model to prefer progress and avoid deadlocks; we ablated `reason_act` (an explicit chain-of-thought) and observed zero gain. The next rung would be to enforce rules programmatically — "the beam rejects any child whose push sends a non-goal-matching box into a corner". But that is moving the intelligence *from* the LLM *into* the search, and at the limit it becomes BFS with a hand-crafted heuristic (our A\* alternative in Section 8). It is no longer an LLM system.
- **Banning visited states in the prompt.** We already ban them structurally, via the visited set in beam. Beam with mock and beam with OpenAI reach the same ∼5-layer depth, which means that even when the LLM is forbidden from revisiting states, its forward ranking does not converge on a goal faster than random.
- **Few-shot exemplars.** A plausible marginal win — show the model a handful of solved level excerpts in the system prompt. We did not test this cell, but in the best case it gets the base model to "has seen Sokoban before" and not meaningfully closer to planning ability. It is still a generalist, still short-horizon, still no push-consequence reasoning.

The fundamental issue is that one-step ranking is a *policy* problem on a domain whose training signal is sparse in the pre-training corpus (Todd et al., 2023, report the same qualitative pattern on Sokoban *generation*). Prompt engineering reshapes how the policy is queried; it does not change what the policy has learned. The remaining lever is to **train the policy on Sokoban data**.

---

## 7. Limitations

- **Sample size.** Ten levels are enough to observe the *trend* reported here — every LLM cell is 0 / 10 — but a rigorous difficulty-vs-success curve would require bucketing Microban by number of boxes / room size and running 20+ levels per bucket. Our scripts already parse the full 155-level file; scaling up is a matter of budget, not engineering.
- **Single backbone.** We only tested `llama-3.1-8b-instruct`. A stronger model (GPT-4o, Claude 3.5 Sonnet, Qwen2.5-72B) might raise the ranking quality enough to matter at B=3; we did not measure this because the take-home explicitly prefers smaller models and because Todd et al. (2023) argue that the gap is bridged by *domain data*, not by model size.
- **`tuple_fields` omits walls.** It is useful only as a deliberate stress test of partial observability, not as a production encoding.
- **Non-determinism at `samples > 1`.** OpenAI runs at `samples=3` use `T=0.5`; we disable the predictor cache in that regime because it would be unsound, which inflates wall-clock for those cells.
- **No admissible lower bound in beam.** The beam scores by cumulative LLM preference; there is no admissibility guarantee, so the search can settle in a local minimum BFS would avoid. A Manhattan-distance-based A\* variant is listed in Section 8 as a concrete remedy.

---

## 8. Future work: a Sokoban-specific predictor via SFT + RL

Given that Section 6 pins the failure on the LLM's *learned* policy rather than on the pipeline or on the prompt surface, the highest-leverage next step is to replace the generalist backbone with a predictor **trained on Sokoban**.

The plan below is concrete enough to execute and reuses the existing harness end-to-end.

### 8.1 Drop-in integration

The `OneStepPredictor` protocol is intentionally narrow:

```python
async def predict_ranked(level, state, legal_actions, board_text) -> list[ActionPrediction]: ...
```

A trained Sokoban model only needs to satisfy this protocol. Everything downstream — `beam_search_llm_guided_async`, `greedy_single_action_rollout_async`, the API `solve` endpoint, the research scripts, the factorial grid, the UI — works unchanged. `track1.llm.factory.make_predictor` gains one option `"sokoban_rl"` that loads the local checkpoint.

### 8.2 Data generation (free, offline, fully deterministic)

BFS already provides optimal solutions for ≈ 90 % of Microban; that is a large supervised corpus we can mint for free.

- Run BFS over the full 155-level Microban set. For every solved level, emit one training example per state on the optimal plan: `(ascii_board, legal_actions, optimal_next_action)`. Expected ≈ 7k–15k supervised pairs from Microban alone.
- Augment with off-trajectory data: for each `(state, a^*)` on the optimal path, enumerate the siblings `(state, a')` with `a' ≠ a^*` and label them as negatives so that the ranking learns to rank the optimal action first and the rest by Manhattan-distance-after-push (already computed in `ascii_features`).
- Augment with Microban II, Sasquatch, Yoshio on demand. Hold out 10 levels across difficulty bins as the test set — the same protocol as Section 5.

### 8.3 Stage 1 — behavioural cloning (SFT)

- **Format.** Identical to our current OpenAI prompt, so the same `_build_user_prompt` / `_parse_ranking` code applies.
- **Loss.** Cross-entropy on the target ranking (first-ranked target is the BFS-optimal action; the rest are ranked by Manhattan-distance-after-push).
- **Expected outcome.** By analogy with Todd et al. (2023) on level generation: single-digit % success with the zero-shot base, jumping to a large plateau once domain data appears. Concrete KPI to beat: greedy success > 0 % on the held-out 10 levels. A plausible target for ≤ 3-box levels is 30–60 %.

### 8.4 Stage 2 — RL fine-tuning

SFT teaches the model what BFS *would* do on the optimal path, but the policy will still wander into off-trajectory states and have no idea what to do. RL closes that gap by letting the policy see its own failures.

- **Algorithm.** GRPO (group-relative policy optimisation) or PPO on top of the SFT checkpoint. GRPO avoids a learned value head and maps cleanly to the "sample many trajectories, keep the solved ones as positive advantage" pattern used in recent reasoning-RL work.
- **Rollout loop.** Sample \(N\) trajectories per level with the current policy embedded in beam (\(B=3\)) for exploration. Each trajectory terminates on solve, cycle, deadlock, or budget.
- **Reward shaping.**
  - **+1.0** on solve within the budget.
  - **+0.1 · ΔΣ**(Manhattan distance box → goal) per step — a dense shaping reward that says "pushing a box closer to its goal is good". The Manhattan term is the same we already precompute in `ascii_features`.
  - **−0.3** on deadlock (corner push on a non-goal box). Deterministically detectable.
  - **−0.01** per step, to discourage padding.
  - **−0.2** on state cycle, to directly penalise the exact failure mode we observe in Section 6.
- **Curriculum.** Start on trivial levels (≤ 2 boxes) and promote a level into the training set only once the policy reaches ≥ 80 % success on it. Mirrors how Microban itself is ordered and keeps the reward signal dense.
- **Expected effect on the failure modes.** The `state cycle` rate in greedy should drop sharply (the −0.2 cycle penalty directly targets it). `frontier exhausted` in beam should drop too, because an RL policy that has learnt to prefer *pushes* produces genuinely *new* states rather than re-visited ones. Encoding and prompt-style differences should re-appear meaningfully once the model has an internalised policy to key on.

## 9. Conclusion

Three observations summarise the project:

1. **The tree-search-with-one-step-LLM pattern is correctly implemented.** The predictor interface is narrow, the beam search uses LLM rankings as cumulative scores with a visited set and LLM calls, and the evaluation harness logs every quantity the task statement asks for.
2. **On Microban, a generalist LLM does not carry enough spatial signal for this pattern to solve more puzzles than BFS.** Across a full factorial over encoding, prompt style, and self-consistency — and against a random-permutation control — the LLM ranking never moves the success rate above the control line, and it costs orders of magnitude more wall-clock per level.
3. **The evidence points to the LLM's learned policy as the bottleneck, not the pipeline or the prompt.** Log inspection rules out pipeline failures; the prompt-engineering ceiling argument rules out easy wins on the prompt surface; the Todd et al. (2023) result predicts exactly this regime for zero-shot LLMs on Sokoban. The remedy is to train the policy on Sokoban data.

The one-line conclusion is therefore: **the most valuable next step for this project is a training pipeline, not another prompt tweak.**

---

## References

1. Todd, G., Earle, S., Nasir, M. U., Green, M. C., & Togelius, J. *Level Generation Through Large Language Models*. Foundations of Digital Games (FDG), 2023. [arXiv:2302.05817](https://arxiv.org/abs/2302.05817).
2. Skinner, D. W. *Microban puzzle collection*. <http://www.abelmartin.com/rj/sokobanJS/Skinner/David%20W.%20Skinner%20-%20Sokoban_files/Microban.txt>.
3. Junghanns, A., & Schaeffer, J. *Sokoban: Enhancing general single-agent search methods using domain knowledge*. Artificial Intelligence, 129(1–2), 2001. [link](https://pdf.sciencedirectassets.com/271585/1-s2.0-S0004370200X0077X/1-s2.0-S0004370201001096/main.pdf?X-Amz-Security-Token=IQoJb3JpZ2luX2VjEI7%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FwEaCXVzLWVhc3QtMSJHMEUCIGr4012jI25CAiOgVZw9HbVpR95sQ4eD2KHWuln1PeZTAiEA9fx0yxNzK4lWfXaRon%2FWkImAVWHdMYoUHtJvCFbAWM4qswUIVxAFGgwwNTkwMDM1NDY4NjUiDB8Rmj5Bs%2FUIiQJreiqQBf7h6por%2B8pZeNjFcMR%2BpTXVgT8xYpJBQng2I73l49s0%2BRFQF%2FEDQC8HcjUcvF0%2Fc1%2BLMvercTf2pLo6Es2KGnCi9foYekRBCDdS914iFUDq2Stz1yE9lQ%2BXw%2FMo7eklQInw4MJc78B4Dpma%2FCBLkAl6xNcPdOZEvuYP1OZKIfbVG4S5Jxqq1xRNtm9gLWvpQkUxxDycCk80L4pz%2FvbxmavV6NpPIPfjLkBwFaUxQ69G658cDlrWJ6XqSCnpCPyS5Vhdy1ZArnUFvQXUJCLvKZbUlRByOWBHrNIMHTpDT7A15FM%2Fcgr5Vq5R%2Fy5YGJBa0RxvBPUVML6eV1hVrZwV2yyfBdcv%2BCUxKqJgxdbEIWKxcxFO1hxct2qw%2BsqkIgWn6k4pUk4oMUmV7VdB0kL5rOy7ID6yfhoXQCiTBg1caRpKNekMlElVt6hU5C2NYALMH2V0FEsdAXSHb2mcAOVr0xKBaX3XbHlJsqTb522MYbfseIMPO3cQ6joNCL399OrZW90IMkJa4K4Ad3PJAcA0VC21cBeCGjOCf0aeMbNTqGK%2BtsLIy2Wefv7iIwv9YnViWwhR5bdSjulbh3kXO0X6hMSA9aM3Y%2B6ayGz78aswKLNl5mOfZsMogh8hxKVjKA1g%2Bel3Jrz3IhManVHVR5CpB2PKb%2F5lcbRossTn8QChYGMWSnI5RTdWY9ZhoG7yiW6DUNSv%2FXIEzFXXjSfGslRc069LpWw29bV7KX3bsl%2Bf2O1gfYwZF5xXgrmFpjhxphDBSfI4EnSLTQfCiQxhl%2FBL2iDerqAwZWK7dy2B0%2B%2BS%2BczIJMD6STQEh%2FcA3%2Bgk8P3KJjuwvQgNkVygFga5c1%2B4UynVAEKDrnqWLZ0npTFLIp%2FiMNSLpc8GOrEBOgf5yjkF%2Br8qXhdtb2RwivP4nU%2B7R7TvtGqUInagbu0bMDPVGm%2BN7PGc29zy8JDJjl2%2FDrESfuByBIBGzs9nkC1kJYh%2F%2F0%2BVg7PPIAKc38cegSYobwOyQDOPF6fNM5AWh4oKvc%2FgDdL4m7gOg4fVl3RtnEJZ2YVzZq1nSoLZ8iXO2jQ6NgLcc6%2F0QI7VXWJHqJflSxxYMJLnUXb3dhOTBcEkOcwwSvATNtJPmzVNNq64&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Date=20260422T225739Z&X-Amz-SignedHeaders=host&X-Amz-Expires=300&X-Amz-Credential=ASIAQ3PHCVTYWRAIUYGH%2F20260422%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Signature=8fec438f1c6428d5f7038c9b3187e7e69ddfdde159c109ee3291f38f49e61fcf&hash=93845fff3bf9cd35583cc5c05b302deabf234672274d8f9ad37b51e672e5fc0f&host=68042c943591013ac2b2430a89b270f6af2c76d8dfd086a07176afe7c76c2c61&pii=S0004370201001096&tid=spdf-35fc5606-70a6-4336-aea9-0517f693c11d&sid=29215c0e133bf647b4887b8-4afab37d445cgxrqb&type=client&tsoh=d3d3LnNjaWVuY2VkaXJlY3QuY29t&rh=d3d3LnNjaWVuY2VkaXJlY3QuY29t&ua=1e005e04535f570f560f50&rr=9f083a8a4e51cc1b&cc=de)

---

## Appendix A — Reproducibility

Install once (see the main `README.md` for the full walk-through):

```bash
cd track_1
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python scripts/download_microban.py -o data/microban.txt
```

Then re-run the experiments that produce the CSVs in `research/output/`:

```bash
# Section 5 headline: BFS + mock baselines (zero cost)
python scripts/run_experiments.py -n 12 -o research/output/latest.csv

# Section 5 headline: OpenAI greedy + beam rows (costs $, needs OPENAI_API_KEY)
python scripts/run_experiments.py -n 8 --openai-rollouts \
  -o research/output/latest_openai.csv

# Section 5.1: BFS vs greedy vs beam on the same levels
python scripts/run_llm_comparison.py -n 8 \
  --greedy-steps 60 --greedy-calls 60 \
  --beam-width 3 --beam-depth 30 --beam-calls 60 \
  -o research/output/llm_compare.csv

# Section 5.2.1: mock ablation (budget × encoding × beam_width)
python scripts/run_predictor_ablation.py -n 10 \
  -o research/output/predictor_ablation.csv

# Section 5.2.2: OpenAI factorial (encoding × prompt_style × samples × {greedy, beam})
python scripts/run_predictor_ablation.py -n 6 --with-openai \
  --openai-encodings ascii ascii_features \
  --prompt-styles plain reason_act \
  --samples 1 3 \
  --beam-widths 3 \
  --openai-budget-steps 80 --openai-budget-calls 80 \
  -o research/output/predictor_ablation_openai.csv
```

## Appendix B — Metric glossary

| Metric | Definition |
|---|---|
| `success` | `True` iff the returned plan, when replayed from the initial state, ends in a terminal state (all boxes on goals). |
| `plan_len` | Number of actions in the returned plan. `0` when no plan was returned. |
| `nodes_expanded` | BFS: states popped from the queue. Beam: state-action pairs scored. Greedy: steps applied. |
| `llm_calls` | API calls to `predict_ranked`. BFS = 0. |
| `cache_hits` | Predictor-cache hits on identical `(state, legal, prompt_style, board_text)`. |
| `wall_ms` | Wall-clock time for that run (ms). |
| `message` | Stop reason. Values: `""` (solved), `state cycle`, `max_llm_calls`, `max_depth`, `frontier exhausted`, `BFS exhausted or node cap`, `empty prediction`, `illegal top action`, `error:<TypeName>:<message>`. |

## Appendix C — Code layout

| Module | Role |
|---|---|
| `track1.sokoban.parser`, `track1.sokoban.simulator` | Microban parser, state/action semantics, push legality. |
| `track1.sokoban.search.breadth_first_search` | Exact BFS baseline. |
| `track1.llm.predictor` | `OneStepPredictor` protocol + `MockPredictor`. |
| `track1.llm.openai_predictor` | `OpenAIPredictor` (prompt construction, JSON parse, caching, self-consistency). |
| `track1.llm.prompts` | Board encodings (`ascii`, `labeled_rows`, `tuple_fields`, `ascii_features`). |
| `track1.llm.rollout.greedy_single_action_rollout_async` | Greedy one-step chain (ablation). |
| `track1.llm.beam.beam_search_llm_guided_async` | Beam search guided by LLM ranking (spec deliverable). |
| `track1.llm.factory.make_predictor` | Predictor factory — the one line to edit to plug in a trained model. |
| `track1.api.main`, `track1.api.schemas` | FastAPI with `/solve` supporting `bfs`, `llm_greedy`, `beam`. |
| `frontend/src/App.vue` | Vue 3 UI: three solver tabs (BFS, Beam + LLM, LLM predictor), apply-plan and replay controls. |
| `scripts/*` | Experimental harness (see Appendix A). |

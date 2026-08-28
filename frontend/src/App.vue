<script setup>
import { computed, onBeforeUnmount, onMounted, ref, watch } from "vue";
import BoardGrid from "./components/BoardGrid.vue";
import SolveMetrics from "./components/SolveMetrics.vue";
import * as api from "./api.js";

const levels = ref([]);
const levelIndex = ref(0);
const sessionId = ref(null);
const board = ref({ rows: [], solved: false, legal_actions: [] });
const moveCount = ref(0);
const canUndo = ref(false);
const status = ref("");
const modelName = ref("llama-3.1-8b-instruct");
const solveLog = ref("");
const openaiReady = ref(false);
const stateEncoding = ref("ascii");
const replayDelayMs = ref(280);
const replaying = ref(false);
const lastPlan = ref([]);
const lastMetrics = ref(null);
const previewText = ref("");
const previewLoading = ref(false);

const maxBfsNodes = ref(500000);
const beamWidth = ref(3);
const beamMaxDepth = ref(30);
const beamMaxLlmCalls = ref(150);
const solverPanel = ref("rollout");
let previewTimer = null;

const legalSet = computed(() => new Set(board.value.legal_actions || []));
const canReplay = computed(
  () => lastPlan.value.length > 0 && sessionId.value && !replaying.value
);
const showLlmPreview = computed(() => solverPanel.value !== "bfs");
const beamPredictorNote = computed(() =>
  openaiReady.value
    ? "OpenAI ranks legal actions at each node."
    : "OpenAI key missing → mock ranking (random, shown as baseline)."
);

async function refreshHealth() {
  try {
    const h = await api.health();
    openaiReady.value = !!h.openai_configured;
    modelName.value = h.openai_model || "llama-3.1-8b-instruct";
    status.value = `${h.levels_loaded} levels loaded`;
  } catch (e) {
    openaiReady.value = false;
    modelName.value = "unknown";
    status.value = `Cannot reach API: ${e.message}`;
  }
}

async function loadLevels() {
  levels.value = await api.listLevels();
}

async function newGame() {
  solveLog.value = "";
  lastPlan.value = [];
  lastMetrics.value = null;
  previewText.value = "";
  const s = await api.createSession(levelIndex.value);
  sessionId.value = s.session_id;
  board.value = s.board;
  moveCount.value = s.move_count;
  canUndo.value = !!s.can_undo;
}

async function reset() {
  if (!sessionId.value) return;
  const s = await api.resetSession(sessionId.value);
  board.value = s.board;
  moveCount.value = s.move_count;
  canUndo.value = !!s.can_undo;
  solveLog.value = "";
}

async function doMove(action) {
  if (!sessionId.value) return;
  if (!legalSet.value.has(action)) return;
  const r = await api.move(sessionId.value, action);
  board.value = r.board;
  moveCount.value = r.move_count;
  canUndo.value = !!r.can_undo;
}

async function undoLast() {
  if (!sessionId.value || !canUndo.value) return;
  try {
    const r = await api.undo(sessionId.value);
    board.value = r.board;
    moveCount.value = r.move_count;
    canUndo.value = !!r.can_undo;
  } catch (e) {
    solveLog.value = String(e.message || e);
  }
}

function sleep(ms) {
  return new Promise((r) => setTimeout(r, ms));
}

async function runSolve(method) {
  if (!sessionId.value) return;
  if (method === "bfs") solverPanel.value = "bfs";
  else if (method === "beam" || method === "beam_mock") solverPanel.value = "beam";
  else solverPanel.value = "rollout";
  solveLog.value = "Running…";
  const common = {
    max_nodes: Number(maxBfsNodes.value) || 500000,
  };
  let body;
  if (method === "bfs") {
    body = { method: "bfs", ...common };
  } else if (method === "llm_greedy") {
    body = {
      method: "llm_greedy",
      ...common,
      predictor: openaiReady.value ? "openai" : "auto",
      state_encoding: stateEncoding.value,
    };
  } else if (method === "beam" || method === "beam_mock") {
    body = {
      method: "beam",
      ...common,
      predictor: method === "beam_mock" ? "mock" : openaiReady.value ? "openai" : "auto",
      state_encoding: stateEncoding.value,
      beam_width: Number(beamWidth.value) || 3,
      beam_max_depth: Number(beamMaxDepth.value) || 30,
      beam_max_llm_calls: Number(beamMaxLlmCalls.value) || 150,
    };
  } else {
    solveLog.value = `Unknown solver: ${method}`;
    return;
  }
  try {
    const r = await api.solve(sessionId.value, body);
    const mode = r.search_mode || method;
    const enc = method !== "bfs" ? ` · encoding=${r.state_encoding}` : "";
    solveLog.value = `${String(mode).toUpperCase()}${enc} · ${
      r.success ? "solved" : "not solved"
    } · ${r.actions.length} steps · ${r.nodes_expanded} nodes · ${r.llm_calls ?? 0} LLM calls${
      r.message ? " · " + r.message : ""
    }`;
    lastPlan.value = r.actions || [];
    lastMetrics.value = {
      method: r.search_mode || method,
      encoding: r.state_encoding || stateEncoding.value,
      success: r.success,
      planLen: r.actions.length,
      nodesExpanded: r.nodes_expanded,
      llmCalls: r.llm_calls ?? 0,
    };
  } catch (e) {
    solveLog.value = String(e.message || e);
    lastPlan.value = [];
    lastMetrics.value = null;
  }
}

async function applySolutionInstant() {
  if (!sessionId.value || !lastPlan.value.length) return;
  const r = await api.applyPlan(sessionId.value, lastPlan.value, true);
  board.value = r.board;
  moveCount.value = r.move_count;
  canUndo.value = !!r.can_undo;
  solveLog.value = `Replay: applied ${r.applied_steps} move(s) · puzzle complete=${r.completed}${
    r.illegal_at_step != null ? " · stopped at step " + r.illegal_at_step : ""
  }`;
}

async function animateReplay() {
  if (!sessionId.value || !lastPlan.value.length || replaying.value) return;
  replaying.value = true;
  try {
    await api.resetSession(sessionId.value);
    const s = await api.getSession(sessionId.value);
    board.value = s.board;
    moveCount.value = s.move_count;
    canUndo.value = !!s.can_undo;
    for (const a of lastPlan.value) {
      await sleep(Number(replayDelayMs.value) || 200);
      const r = await api.move(sessionId.value, a);
      board.value = r.board;
      moveCount.value = r.move_count;
      canUndo.value = !!r.can_undo;
      if (r.illegal) break;
      if (board.value.solved) break;
    }
  } finally {
    replaying.value = false;
  }
}

async function refreshLlmPreview() {
  if (!sessionId.value) return;
  previewLoading.value = true;
  try {
    const r = await api.llmPreview(sessionId.value, stateEncoding.value);
    previewText.value = r.text;
  } catch (e) {
    previewText.value = String(e.message || e);
  } finally {
    previewLoading.value = false;
  }
}

function schedulePreviewRefresh() {
  if (!showLlmPreview.value) {
    previewText.value = "";
    previewLoading.value = false;
    return;
  }
  if (previewTimer) clearTimeout(previewTimer);
  if (!sessionId.value) {
    previewText.value = "";
    return;
  }
  previewTimer = setTimeout(() => {
    refreshLlmPreview();
  }, 150);
}

function onKey(e) {
  const t = e.target;
  if (t && (t.tagName === "INPUT" || t.tagName === "TEXTAREA" || t.tagName === "SELECT")) return;
  if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === "z") {
    e.preventDefault();
    undoLast();
    return;
  }
  const map = {
    ArrowUp: "up",
    ArrowDown: "down",
    ArrowLeft: "left",
    ArrowRight: "right",
    w: "up",
    W: "up",
    s: "down",
    S: "down",
    a: "left",
    A: "left",
    d: "right",
    D: "right",
  };
  const a = map[e.key];
  if (!a) return;
  e.preventDefault();
  doMove(a);
}

onMounted(async () => {
  window.addEventListener("keydown", onKey);
  await refreshHealth();
  try {
    await loadLevels();
    await newGame();
  } catch (e) {
    status.value = `Load error: ${e.message}`;
  }
});

onBeforeUnmount(() => window.removeEventListener("keydown", onKey));

watch(
  [sessionId, stateEncoding, () => (board.value.rows || []).join("\n"), showLlmPreview],
  () => {
    schedulePreviewRefresh();
  }
);

onBeforeUnmount(() => {
  if (previewTimer) clearTimeout(previewTimer);
});
</script>

<template>
  <div class="page">
    <header class="head">
      <h1>Sokoban lab</h1>
      <p class="model-line">Model: <strong>{{ modelName }}</strong></p>
      <p class="status">{{ status }}</p>
    </header>

    <section class="card session-top" aria-labelledby="level-heading">
      <h2 id="level-heading" class="card-title">Level & session</h2>
      <div class="row">
        <label class="field">
          <span class="label">Puzzle</span>
          <select v-model.number="levelIndex" class="control" @change="newGame">
            <option v-for="l in levels" :key="l.index" :value="l.index">
              #{{ l.index }} {{ l.title || "untitled" }} — {{ l.width }}×{{ l.height }},
              {{ l.num_boxes }} box(es)
            </option>
          </select>
        </label>
      </div>
      <div class="btn-row">
        <button type="button" class="btn primary" @click="newGame">New session</button>
        <button type="button" class="btn" @click="reset" :disabled="!sessionId">Reset board</button>
        <button type="button" class="btn" :disabled="!sessionId || !canUndo" @click="undoLast">
          Undo move
        </button>
      </div>
      <p class="kbd-hint"><kbd>Ctrl</kbd>+<kbd>Z</kbd> / <kbd>Cmd</kbd>+<kbd>Z</kbd> also undoes.</p>
    </section>

    <div class="grid-main">

      <section class="card board-card center-stack" aria-labelledby="board-heading">
        <h2 id="board-heading" class="card-title">Board</h2>
        <BoardGrid :rows="board.rows" />
        <div class="legal-row" v-if="board.legal_actions?.length">
          <span class="legal-label">Legal:</span>
          <span v-for="a in board.legal_actions" :key="a" class="chip">{{ a }}</span>
        </div>
        <p class="board-meta">
          Moves played: <strong>{{ moveCount }}</strong>
          <span v-if="board.solved" class="badge ok">All goals covered</span>
        </p>
        <SolveMetrics
          v-if="lastMetrics"
          :nodes-expanded="lastMetrics.nodesExpanded"
          :llm-calls="lastMetrics.llmCalls"
          :plan-len="lastMetrics.planLen"
          :success="lastMetrics.success"
          :method="lastMetrics.method"
          :encoding="lastMetrics.encoding"
        />
        <p v-if="solveLog" class="log">{{ solveLog }}</p>
      </section>

      <div class="stack">
        <section class="card solver-main" aria-labelledby="solve-heading">
          <h2 id="solve-heading" class="card-title">Solver lab</h2>
          <div class="solver-tabs">
            <button type="button" class="solver-tab-btn" :class="{ active: solverPanel === 'bfs' }" @click="solverPanel = 'bfs'">
              BFS
            </button>
            <button type="button" class="solver-tab-btn" :class="{ active: solverPanel === 'beam' }" @click="solverPanel = 'beam'">
              Beam + LLM
            </button>
            <button type="button" class="solver-tab-btn" :class="{ active: solverPanel === 'rollout' }" @click="solverPanel = 'rollout'">
              LLM predictor
            </button>
          </div>

          <div v-show="solverPanel === 'bfs'" class="solver-pane">
            <label class="field">
              <span class="label">BFS max nodes</span>
              <input v-model.number="maxBfsNodes" class="control" type="number" min="1000" max="5000000" step="1000" />
            </label>
            <button type="button" class="btn primary" :disabled="!sessionId" @click="runSolve('bfs')">
              Run BFS
            </button>
          </div>

          <div v-show="solverPanel === 'beam'" class="solver-pane">
            <p class="solver-note">
              Tree search guided by the one-step LLM predictor. 1 API call per node (ranks all legal actions).
              {{ beamPredictorNote }}
            </p>
            <label class="field">
              <span class="label">State for the LLM</span>
              <select v-model="stateEncoding" class="control">
                <option value="ascii">ASCII map</option>
                <option value="labeled_rows">ASCII + row index prefix</option>
                <option value="tuple_fields">Numeric fields only</option>
                <option value="ascii_features">ASCII + derived features</option>
              </select>
            </label>
            <div class="param-grid">
              <label class="field">
                <span class="label">Beam width</span>
                <input v-model.number="beamWidth" class="control" type="number" min="1" max="16" step="1" />
              </label>
              <label class="field">
                <span class="label">Max depth</span>
                <input v-model.number="beamMaxDepth" class="control" type="number" min="2" max="200" step="1" />
              </label>
              <label class="field">
                <span class="label">LLM call budget</span>
                <input v-model.number="beamMaxLlmCalls" class="control" type="number" min="5" max="2000" step="5" />
              </label>
            </div>
            <div class="btn-row">
              <button type="button" class="btn primary" :disabled="!sessionId" @click="runSolve('beam')">
                Run beam ({{ openaiReady ? "OpenAI" : "auto" }})
              </button>
              <button type="button" class="btn" :disabled="!sessionId" @click="runSolve('beam_mock')">
                Run beam (mock)
              </button>
            </div>
          </div>

          <div v-show="solverPanel === 'rollout'" class="solver-pane">
            <label class="field">
              <span class="label">State for the LLM</span>
              <select v-model="stateEncoding" class="control">
                <option value="ascii">ASCII map (default)</option>
                <option value="labeled_rows">ASCII + row index prefix</option>
                <option value="tuple_fields">Numeric fields only (ablation)</option>
                <option value="ascii_features">ASCII + derived features</option>
              </select>
            </label>
            <div class="btn-row">
              <button type="button" class="btn" :disabled="!sessionId" @click="runSolve('llm_greedy')">
                Greedy rollout
              </button>
            </div>
            <p class="solver-note">
              One LLM call per step: ask for a ranking of legal moves and apply the top pick.
            </p>
          </div>

          <details class="details" v-if="showLlmPreview">
            <summary>Live LLM preview (auto-updating)</summary>
            <p class="card-desc tiny">
              Updates automatically when the board or encoding changes.
            </p>
            <p class="card-desc tiny" v-if="previewLoading">Refreshing…</p>
            <pre v-if="previewText" class="preview">{{ previewText }}</pre>
          </details>
        </section>

        <section class="card" aria-labelledby="replay-heading">
          <h2 id="replay-heading" class="card-title">Replay plan</h2>
          <p class="card-desc">
            The last solver run stores a move list. Use this tab to apply it instantly or animate it.
          </p>
          <label class="field inline">
            <span class="label">Animation delay (ms)</span>
            <input
              v-model.number="replayDelayMs"
              class="control narrow"
              type="number"
              min="50"
              max="2000"
              step="50"
            />
          </label>
          <div class="btn-row">
            <button type="button" class="btn" :disabled="!canReplay" @click="applySolutionInstant">
              Apply full plan
            </button>
            <button type="button" class="btn" :disabled="!canReplay || replaying" @click="animateReplay">
              {{ replaying ? "Animating…" : "Animate replay" }}
            </button>
          </div>
        </section>
      </div>
    </div>
  </div>
</template>

<style scoped>
.page {
  max-width: 1100px;
  margin: 0 auto;
  padding: 1.5rem 1.25rem 3rem;
}

.head {
  margin-bottom: 1.5rem;
}

.head h1 {
  margin: 0 0 0.5rem;
  font-size: 1.5rem;
  font-weight: 650;
  letter-spacing: -0.02em;
}

.model-line {
  margin: 0 0 0.25rem;
  color: #b4bcc9;
  font-size: 0.85rem;
}

.status {
  margin: 0 0 0.75rem;
  color: #8fd6ff;
  font-size: 0.88rem;
}

.grid-main {
  display: grid;
  grid-template-columns: minmax(0, auto) minmax(320px, 1fr);
  gap: 1.5rem;
  align-items: start;
}

@media (max-width: 820px) {
  .grid-main {
    grid-template-columns: 1fr;
  }
}

.card {
  background: #181b22;
  border: 1px solid #2e3542;
  border-radius: 12px;
  padding: 1rem 1.1rem;
  margin-bottom: 1rem;
}

.board-card {
  position: sticky;
  top: 1rem;
}

.center-stack {
  min-width: 0;
}

.session-top {
  margin-bottom: 1rem;
}

.card-title {
  margin: 0 0 0.35rem;
  font-size: 0.95rem;
  font-weight: 600;
  color: #e4e8ef;
}

.card-desc {
  margin: 0 0 0.85rem;
  font-size: 0.8rem;
  color: #8f9aad;
  line-height: 1.45;
}

.card-desc.tiny {
  margin-top: 0.5rem;
  margin-bottom: 0.5rem;
  font-size: 0.75rem;
}

.board-meta {
  margin: 0.85rem 0 0;
  font-size: 0.85rem;
  color: #9aa4b2;
}

.legal-row {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 0.35rem;
  margin-top: 0.65rem;
}
.legal-label {
  font-size: 0.72rem;
  color: #8f9aad;
  text-transform: uppercase;
  letter-spacing: 0.04em;
}
.chip {
  font-size: 0.72rem;
  padding: 0.2rem 0.45rem;
  border-radius: 6px;
  background: #1e2f24;
  color: #9fe8b5;
  border: 1px solid #2d5a3d;
}

.badge {
  margin-left: 0.5rem;
  padding: 0.15rem 0.45rem;
  border-radius: 6px;
  font-size: 0.75rem;
  font-weight: 600;
}
.badge.ok {
  background: #1e2f24;
  color: #9fe8b5;
  border: 1px solid #2d5a3d;
}

.stack {
  min-width: 0;
}

@media (max-width: 980px) {
  .grid-main {
    grid-template-columns: 1fr;
  }
  .solver-tabs {
    grid-template-columns: 1fr;
  }
}

.row {
  margin-bottom: 0.75rem;
}

.field {
  display: flex;
  flex-direction: column;
  gap: 0.35rem;
  width: 100%;
}

.field.inline {
  flex-direction: row;
  align-items: center;
  flex-wrap: wrap;
  gap: 0.5rem;
}

.label {
  font-size: 0.78rem;
  font-weight: 500;
  color: #a8b0bf;
  text-transform: uppercase;
  letter-spacing: 0.04em;
}

.control {
  width: 100%;
  max-width: 100%;
  background: #10131a;
  color: #e8e6e3;
  border: 1px solid #3a4250;
  border-radius: 8px;
  padding: 0.45rem 0.55rem;
  font: inherit;
}

.control.narrow {
  width: 5.5rem;
  max-width: 100%;
}

.param-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(9.5rem, 1fr));
  gap: 0.65rem;
  margin-bottom: 1rem;
}

.btn-row {
  display: flex;
  flex-wrap: wrap;
  gap: 0.5rem;
}

.btn {
  background: #243042;
  color: #e8e6e3;
  border: 1px solid #3d4b63;
  border-radius: 8px;
  padding: 0.45rem 0.85rem;
  font: inherit;
  font-size: 0.88rem;
}

.btn:hover:not(:disabled) {
  background: #2c3a52;
}

.btn:disabled {
  opacity: 0.42;
  cursor: not-allowed;
}

.btn.primary {
  background: #2a4a6e;
  border-color: #4a7ab0;
}

.btn.primary:hover:not(:disabled) {
  background: #32587f;
}

.solver-block {
  margin-bottom: 0.75rem;
}

.solver-main {
  border-color: #355072;
}

.solver-tabs {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 0.4rem;
  margin-bottom: 0.75rem;
}

.solver-tab-btn {
  background: #131822;
  color: #b7c0cf;
  border: 1px solid #313b4e;
  border-radius: 10px;
  padding: 0.38rem 0.42rem;
  font: inherit;
  font-size: 0.76rem;
}

.solver-tab-btn.active {
  background: #2a4a6e;
  border-color: #4a7ab0;
  color: #e8f3ff;
}

.solver-pane {
  padding: 0.8rem 0.85rem;
  border: 1px solid #2f3948;
  border-radius: 10px;
  background: #141a24;
  display: flex;
  flex-direction: column;
  gap: 0.65rem;
}

.solver-note {
  margin: 0.55rem 0 0;
  font-size: 0.74rem;
  color: #9ba7bb;
}

.solver-note code {
  font-size: 0.9em;
  padding: 0.05em 0.35em;
  background: #10131a;
  border-radius: 4px;
}

.subhead {
  margin: 0 0 0.35rem;
  font-size: 0.82rem;
  font-weight: 600;
  color: #c5cdd9;
}

.dpad {
  display: grid;
  grid-template-columns: repeat(3, 3.1rem);
  grid-template-rows: repeat(2, 2.75rem);
  gap: 0.4rem;
  width: fit-content;
}

.dpad-btn {
  font-size: 1.15rem;
  background: #243042;
  color: #e8e6e3;
  border: 1px solid #3d4b63;
  border-radius: 8px;
  cursor: pointer;
}

.dpad-btn:hover:not(:disabled) {
  background: #2c3a52;
}

.details {
  margin-top: 0.85rem;
  padding-top: 0.75rem;
  border-top: 1px solid #2a303a;
  font-size: 0.85rem;
  color: #b4bcc9;
}

.details summary {
  cursor: pointer;
  font-weight: 600;
  color: #c5cdd9;
}

.preview {
  margin: 0.5rem 0 0;
  padding: 0.65rem 0.75rem;
  background: #0d1016;
  border: 1px solid #2a303a;
  border-radius: 8px;
  font-size: 0.72rem;
  line-height: 1.35;
  overflow: auto;
  max-height: 14rem;
  white-space: pre-wrap;
}

.kbd-hint {
  margin: 0.5rem 0 0;
  font-size: 0.75rem;
  color: #6f7a8c;
}

.kbd-hint kbd {
  padding: 0.1em 0.35em;
  border-radius: 4px;
  border: 1px solid #3a4250;
  background: #10131a;
  font-size: 0.7rem;
}

.log {
  margin-top: 0.75rem;
  padding: 0.65rem 0.85rem;
  background: #10131a;
  border-radius: 8px;
  border: 1px solid #2a303a;
  color: #c7e6ff;
  font-size: 0.8rem;
  line-height: 1.45;
  white-space: pre-wrap;
}
</style>

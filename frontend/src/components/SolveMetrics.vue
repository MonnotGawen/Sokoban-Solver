<script setup>
import { computed } from "vue";

const props = defineProps({
  nodesExpanded: { type: Number, default: 0 },
  llmCalls: { type: Number, default: 0 },
  planLen: { type: Number, default: 0 },
  success: { type: Boolean, default: false },
  method: { type: String, default: "" },
  encoding: { type: String, default: "" },
});

const maxBar = computed(() => Math.max(props.nodesExpanded, props.llmCalls, 1));

const methodLabel = computed(() => {
  const m = props.method;
  if (m === "bfs") return "BFS";
  if (m === "llm_greedy") return "Greedy · one LLM step / ply";
  if (m === "beam") return "Beam · tree search + LLM ranking";
  return m;
});

function pct(v) {
  return `${Math.min(100, Math.round((100 * v) / maxBar.value))}%`;
}
</script>

<template>
  <div class="metrics" v-if="method">
    <h3 class="h">Last solver run</h3>
    <p class="row">
      <span class="tag">{{ methodLabel }}</span>
      <span v-if="encoding" class="tag dim">{{ encoding }}</span>
      <span class="tag" :class="success ? 'ok' : 'bad'">{{ success ? "success" : "fail" }}</span>
    </p>
    <p class="nums">Plan length: {{ planLen }}</p>
    <div class="bars">
      <div class="lab">Nodes expanded</div>
      <div class="track"><div class="fill n" :style="{ width: pct(nodesExpanded) }" /></div>
      <div class="val">{{ nodesExpanded }}</div>
    </div>
    <div class="bars">
      <div class="lab">LLM calls</div>
      <div class="track"><div class="fill l" :style="{ width: pct(llmCalls) }" /></div>
      <div class="val">{{ llmCalls }}</div>
    </div>
  </div>
</template>

<style scoped>
.metrics {
  margin-top: 1rem;
  padding: 0.75rem 0.85rem;
  background: #141821;
  border: 1px solid #2a303a;
  border-radius: 10px;
  max-width: 22rem;
}
.h {
  margin: 0 0 0.5rem;
  font-size: 0.8rem;
  font-weight: 600;
  color: #b8c0d0;
}
.row {
  margin: 0 0 0.5rem;
  display: flex;
  flex-wrap: wrap;
  gap: 0.35rem;
}
.tag {
  font-size: 0.7rem;
  padding: 0.15rem 0.4rem;
  border-radius: 6px;
  background: #243042;
  border: 1px solid #3d4b63;
}
.tag.dim {
  opacity: 0.85;
}
.tag.ok {
  border-color: #3a6b4a;
  color: #9fe8b5;
}
.tag.bad {
  border-color: #6b3a3a;
  color: #ffb4b4;
}
.nums {
  margin: 0 0 0.5rem;
  font-size: 0.75rem;
  color: #9aa4b2;
}
.bars {
  display: grid;
  grid-template-columns: 7.5rem 1fr 2.5rem;
  align-items: center;
  gap: 0.35rem 0.5rem;
  margin-bottom: 0.35rem;
}
.lab {
  font-size: 0.7rem;
  color: #9aa4b2;
}
.track {
  height: 8px;
  background: #0d1016;
  border-radius: 4px;
  overflow: hidden;
}
.fill {
  height: 100%;
  border-radius: 4px;
}
.fill.n {
  background: linear-gradient(90deg, #4a90d9, #7ec8e3);
}
.fill.l {
  background: linear-gradient(90deg, #a06be8, #e6a8ff);
}
.val {
  font-size: 0.7rem;
  color: #c7cedd;
  text-align: right;
}
</style>

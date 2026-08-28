<script setup>
import { computed } from "vue";

const props = defineProps({
  rows: { type: Array, default: () => [] },
});

function tileKind(ch) {
  switch (ch) {
    case "#":
      return "wall";
    case "@":
      return "player";
    case "+":
      return "player_goal";
    case "$":
      return "box";
    case "*":
      return "box_goal";
    case ".":
      return "goal";
    default:
      return "floor";
  }
}

function cellLabel(ch) {
  const m = {
    "#": "Wall",
    "@": "Player",
    "+": "Player on goal",
    $: "Box",
    "*": "Box on goal",
    ".": "Goal",
    " ": "Floor",
  };
  return m[ch] || "Floor";
}

const tiles = computed(() =>
  (props.rows || []).map((row) =>
    String(row)
      .split("")
      .map((ch) => ({ ch, kind: tileKind(ch) }))
  )
);
</script>

<template>
  <div class="wrap">
    <div class="grid" role="img" aria-label="Sokoban board with icon tiles">
      <div v-for="(row, ri) in tiles" :key="ri" class="row">
        <span
          v-for="(cell, ci) in row"
          :key="ci"
          class="cell"
          :class="cell.kind"
          :title="cellLabel(cell.ch)"
        >
          <svg class="svg" viewBox="0 0 24 24" aria-hidden="true" focusable="false">
            <g v-if="cell.kind === 'wall'">
              <rect width="24" height="24" rx="2" fill="#3d4758" />
              <path d="M0 8h24M0 16h24" stroke="#2a303a" stroke-width="1.5" />
              <path d="M8 0v8M16 0v8M4 8v8M12 8v8M20 8v8M8 16v8M16 16v8" stroke="#323946" stroke-width="1" />
            </g>
            <g v-else-if="cell.kind === 'floor'">
              <rect width="24" height="24" fill="#151821" />
            </g>
            <g v-else-if="cell.kind === 'goal'">
              <rect width="24" height="24" fill="#1a2520" />
              <circle cx="12" cy="12" r="7" fill="none" stroke="#4caf6a" stroke-width="2" opacity="0.85" />
              <circle cx="12" cy="12" r="2.8" fill="#2d6b3d" opacity="0.45" />
            </g>
            <g v-else-if="cell.kind === 'player'">
              <rect width="24" height="24" fill="#1f2430" />
              <circle cx="12" cy="12" r="7.5" fill="#f0c14d" stroke="#b8892a" stroke-width="1.2" />
              <ellipse cx="12" cy="11" rx="2.2" ry="2.6" fill="#4a3b12" />
            </g>
            <g v-else-if="cell.kind === 'player_goal'">
              <rect width="24" height="24" fill="#1f2e24" />
              <circle cx="12" cy="12" r="9" fill="none" stroke="#4caf6a" stroke-width="1.8" />
              <circle cx="12" cy="12" r="6" fill="#f0c14d" stroke="#b8892a" stroke-width="1" />
              <ellipse cx="12" cy="11" rx="1.8" ry="2.2" fill="#4a3b12" />
            </g>
            <g v-else-if="cell.kind === 'box'">
              <rect width="24" height="24" fill="#221c2e" />
              <rect x="5.5" y="7" width="13" height="11" rx="2.2" fill="#9b6fd8" stroke="#6b4aa6" stroke-width="1.3" />
              <path d="M9 10h6M9 13h6" stroke="#c9b6e8" stroke-width="1" opacity="0.5" />
            </g>
            <g v-else-if="cell.kind === 'box_goal'">
              <rect width="24" height="24" fill="#1a2822" />
              <circle cx="12" cy="12" r="9" fill="none" stroke="#4caf6a" stroke-width="1.6" />
              <rect x="6" y="8" width="12" height="9" rx="1.8" fill="#6ec99a" stroke="#3d8f62" stroke-width="1.2" />
              <path d="M9 12h6M12 9v6" stroke="#2a5c40" stroke-width="1.2" opacity="0.55" />
            </g>
          </svg>
        </span>
      </div>
    </div>
  </div>
</template>

<style scoped>
.wrap {
  display: inline-block;
}

.grid {
  display: inline-block;
  padding: 0.65rem;
  border-radius: 12px;
  background: #12151c;
  border: 1px solid #2a303a;
  box-shadow: 0 8px 28px rgba(0, 0, 0, 0.4);
}

.row {
  display: flex;
  line-height: 0;
}

.cell {
  width: 2rem;
  height: 2rem;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  user-select: none;
  flex-shrink: 0;
}

.svg {
  width: 100%;
  height: 100%;
  display: block;
}

.cell:focus-visible {
  outline: 2px solid #7ec8e3;
  outline-offset: 1px;
  border-radius: 4px;
}
</style>

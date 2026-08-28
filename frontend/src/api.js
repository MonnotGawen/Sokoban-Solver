const base = "";

async function jfetch(url, opts = {}) {
  const r = await fetch(`${base}${url}`, {
    headers: { "Content-Type": "application/json", ...(opts.headers || {}) },
    ...opts,
  });
  if (!r.ok) {
    const t = await r.text();
    throw new Error(`${r.status} ${r.statusText}: ${t}`);
  }
  return r.json();
}

export async function health() {
  return jfetch("/api/health");
}

export async function listLevels() {
  return jfetch("/api/levels");
}

export async function createSession(levelIndex) {
  return jfetch("/api/session", {
    method: "POST",
    body: JSON.stringify({ level_index: levelIndex }),
  });
}

export async function getSession(sessionId) {
  return jfetch(`/api/session/${sessionId}`);
}

export async function resetSession(sessionId) {
  return jfetch(`/api/session/${sessionId}/reset`, { method: "POST" });
}

export async function move(sessionId, action) {
  return jfetch(`/api/session/${sessionId}/move`, {
    method: "POST",
    body: JSON.stringify({ action }),
  });
}

export async function solve(sessionId, body) {
  return jfetch(`/api/session/${sessionId}/solve`, {
    method: "POST",
    body: JSON.stringify(body),
  });
}

export async function applyPlan(sessionId, actions, stopOnIllegal = true) {
  return jfetch(`/api/session/${sessionId}/apply_plan`, {
    method: "POST",
    body: JSON.stringify({ actions, stop_on_illegal: stopOnIllegal }),
  });
}

export async function undo(sessionId) {
  return jfetch(`/api/session/${sessionId}/undo`, { method: "POST" });
}

export async function llmPreview(sessionId, encoding = "ascii") {
  const q = new URLSearchParams({ encoding });
  return jfetch(`/api/session/${sessionId}/llm_preview?${q}`);
}

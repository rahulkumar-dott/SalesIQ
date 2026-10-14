// Dev: Vite proxies /api -> FastAPI. Production build: FastAPI serves the UI and the API on the same origin.
const BASE = import.meta.env.DEV ? "/api" : "";

async function req(path, opts) {
  const r = await fetch(BASE + path, opts);
  if (!r.ok) {
    let msg = r.statusText;
    try {
      msg = (await r.json()).detail || msg;
    } catch {
      /* not json */
    }
    throw new Error(msg);
  }
  return r.json();
}

export const api = {
  calls: () => req("/calls"),
  call: (name) => req(`/calls/${encodeURIComponent(name)}`),
  aggregate: () => req("/aggregate"),
  budget: () => req("/budget"),
  dataset: () => req("/dataset"),
  jobs: () => req("/jobs"),
  job: (id) => req(`/jobs/${id}`),
  cancelJob: (id) => req(`/jobs/${id}/cancel`, { method: "POST" }),
  startSample: (sample) =>
    req("/jobs/dataset", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ sample }),
    }),
  upload: (file) => {
    const fd = new FormData();
    fd.append("file", file);
    return req("/jobs/upload", { method: "POST", body: fd });
  },
};

(() => {
  const M = window.ATTENTION_MODEL;
  if (!M) return;
  const { embed_dim: D, num_heads: H, max_seq: MAX_SEQ } = M.config;
  const d_h = D / H;

  const vocab = M.vocab;
  const word_to_id = new Map(vocab.map((w, i) => [w, i]));

  const E = Float32Array.from(M.E.data);    // (V, D) row-major
  const PE = Float32Array.from(M.PE.data);  // (MAX_SEQ, D)
  const W_q = Float32Array.from(M.W_q.data);
  const W_k = Float32Array.from(M.W_k.data);
  const W_v = Float32Array.from(M.W_v.data);
  const W_o = Float32Array.from(M.W_o.data);

  const V = vocab.length;

  // ===== tensor helpers =====
  function matmul(A, B, m, k, n) {
    // A: m x k, B: k x n, out: m x n
    const out = new Float32Array(m * n);
    for (let i = 0; i < m; i++) {
      for (let j = 0; j < n; j++) {
        let s = 0;
        for (let t = 0; t < k; t++) s += A[i * k + t] * B[t * n + j];
        out[i * n + j] = s;
      }
    }
    return out;
  }

  function softmaxRow(row, offset, len) {
    let max = -Infinity;
    for (let i = 0; i < len; i++) if (row[offset + i] > max) max = row[offset + i];
    let sum = 0;
    for (let i = 0; i < len; i++) {
      row[offset + i] = Math.exp(row[offset + i] - max);
      sum += row[offset + i];
    }
    for (let i = 0; i < len; i++) row[offset + i] /= sum;
  }

  // ===== forward pass =====
  function forward(tokenIds) {
    const T = tokenIds.length;
    // x = E[ids] + PE[:T]  → (T, D)
    const x = new Float32Array(T * D);
    for (let t = 0; t < T; t++) {
      const eRow = tokenIds[t] * D;
      const pRow = t * D;
      for (let d = 0; d < D; d++) {
        x[t * D + d] = E[eRow + d] + PE[pRow + d];
      }
    }
    const q = matmul(x, W_q, T, D, D);
    const k = matmul(x, W_k, T, D, D);
    const v = matmul(x, W_v, T, D, D);

    // scores per head (H, T, T)
    const attn = new Float32Array(H * T * T);
    const scale = 1.0 / Math.sqrt(d_h);
    for (let h = 0; h < H; h++) {
      const baseHead = h * d_h;
      // for each query position i
      for (let i = 0; i < T; i++) {
        const qBase = i * D + baseHead;
        // dot with each key position
        for (let j = 0; j < T; j++) {
          const kBase = j * D + baseHead;
          let s = 0;
          for (let t = 0; t < d_h; t++) s += q[qBase + t] * k[kBase + t];
          attn[h * T * T + i * T + j] = s * scale;
        }
        // softmax over j
        softmaxRow(attn, h * T * T + i * T, T);
      }
    }

    // output: for each head, attn @ v_head. Merge heads into (T, D).
    const out = new Float32Array(T * D);
    for (let h = 0; h < H; h++) {
      const baseHead = h * d_h;
      for (let i = 0; i < T; i++) {
        for (let t = 0; t < d_h; t++) {
          let s = 0;
          for (let j = 0; j < T; j++) {
            s += attn[h * T * T + i * T + j] * v[j * D + baseHead + t];
          }
          out[i * D + baseHead + t] = s;
        }
      }
    }
    const final = matmul(out, W_o, T, D, D);
    return { x, attn, out: final, T };
  }

  // ===== tokenization =====
  function tokenize(text) {
    const tokens = text.trim().toLowerCase().split(/\s+/).filter(Boolean);
    const ids = [];
    const flags = [];
    for (const t of tokens) {
      if (ids.length >= MAX_SEQ) break;
      if (word_to_id.has(t)) {
        ids.push(word_to_id.get(t));
        flags.push(false);
      } else {
        ids.push(word_to_id.get("<unk>"));
        flags.push(true);
      }
    }
    return { tokens: tokens.slice(0, MAX_SEQ), ids, flags };
  }

  // ===== rendering =====
  const CELL = 24;

  function renderTokens(tokens, flags) {
    const el = document.getElementById("attn-tokens");
    el.innerHTML = "";
    if (tokens.length === 0) {
      el.innerHTML = '<span class="hint" style="margin:0;opacity:.5">type a sentence above</span>';
      return;
    }
    tokens.forEach((t, i) => {
      const span = document.createElement("span");
      span.className = "attn-token" + (flags[i] ? " unk" : "");
      span.textContent = flags[i] ? `${t}(?)` : t;
      el.appendChild(span);
    });
  }

  function renderHeads(attn, T, tokens) {
    const container = document.getElementById("attn-heads");
    container.innerHTML = "";
    for (let h = 0; h < H; h++) {
      const head = document.createElement("div");
      head.className = "attn-head";
      head.innerHTML = `<h3>Head ${h + 1}</h3>`;

      const axisX = document.createElement("div");
      axisX.className = "attn-axis-x";
      tokens.forEach((t) => {
        const sp = document.createElement("span");
        sp.textContent = t.slice(0, 3);
        axisX.appendChild(sp);
      });
      head.appendChild(axisX);

      const row = document.createElement("div");
      row.className = "attn-head-row";
      const axisY = document.createElement("div");
      axisY.className = "attn-axis-y";
      tokens.forEach((t) => {
        const sp = document.createElement("span");
        sp.textContent = t.slice(0, 4);
        axisY.appendChild(sp);
      });
      row.appendChild(axisY);

      const grid = document.createElement("div");
      grid.className = "attn-grid";
      grid.style.gridTemplateColumns = `repeat(${T}, 18px)`;
      grid.style.gridTemplateRows = `repeat(${T}, 18px)`;

      // find max for normalization
      let max = 0;
      for (let i = 0; i < T * T; i++) {
        const v = attn[h * T * T + i];
        if (v > max) max = v;
      }
      max = max || 1;

      for (let i = 0; i < T; i++) {
        for (let j = 0; j < T; j++) {
          const v = attn[h * T * T + i * T + j];
          const norm = v / max;
          const red = Math.floor(norm * 255);
          const green = Math.floor(norm * 180);
          const cell = document.createElement("div");
          cell.className = "attn-cell";
          cell.dataset.head = h;
          cell.dataset.i = i;
          cell.dataset.j = j;
          cell.style.background = `rgb(${red},${green},0)`;
          cell.title = `${tokens[i]} → ${tokens[j]} : ${(v * 100).toFixed(1)}%`;
          grid.appendChild(cell);
        }
      }
      row.appendChild(grid);
      head.appendChild(row);
      container.appendChild(head);
    }
  }

  function renderBars(attn, T, tokens, head, i) {
    const el = document.getElementById("attn-bars");
    const label = document.getElementById("attn-selected-label");
    el.innerHTML = "";
    label.textContent = `Head ${head + 1} — query: "${tokens[i]}"`;
    for (let j = 0; j < T; j++) {
      const v = attn[head * T * T + i * T + j];
      const row = document.createElement("div");
      row.className = "bar-row";
      row.innerHTML =
        `<span class="tok">${tokens[j]}</span>` +
        `<div class="bar-wrap"><div class="bar" style="width:${v * 100}%"></div></div>` +
        `<span class="pct">${(v * 100).toFixed(1)}%</span>`;
      el.appendChild(row);
    }
  }

  function renderEmbedStrip(canvas, data, T) {
    const ctx = canvas.getContext("2d");
    const cw = canvas.width;
    const ch = canvas.height;
    const cellW = cw / D;
    const cellH = Math.min(ch / Math.max(T, 1), 24);
    ctx.fillStyle = "#000";
    ctx.fillRect(0, 0, cw, ch);
    // find max abs for normalization
    let max = 0;
    for (let i = 0; i < T * D; i++) {
      const a = Math.abs(data[i]);
      if (a > max) max = a;
    }
    max = max || 1;
    for (let i = 0; i < T; i++) {
      for (let j = 0; j < D; j++) {
        const v = data[i * D + j] / max; // -1..1
        if (v > 0) {
          const r = Math.floor(v * 255);
          ctx.fillStyle = `rgb(${r},${Math.floor(v * 180)},0)`;
        } else {
          const b = Math.floor(-v * 255);
          ctx.fillStyle = `rgb(0,${Math.floor(-v * 140)},${b})`;
        }
        ctx.fillRect(j * cellW, i * cellH, Math.ceil(cellW), Math.ceil(cellH));
      }
    }
  }

  function renderEmbeddings(x, tokens) {
    // x = E[ids] + PE; to show E alone, re-look up embeddings
    const T = tokens.length;
    const embedOnly = new Float32Array(T * D);
    const peOnly = new Float32Array(T * D);
    for (let i = 0; i < T; i++) {
      const id = word_to_id.has(tokens[i]) ? word_to_id.get(tokens[i]) : word_to_id.get("<unk>");
      for (let j = 0; j < D; j++) {
        embedOnly[i * D + j] = E[id * D + j];
        peOnly[i * D + j] = PE[i * D + j];
      }
    }
    renderEmbedStrip(document.getElementById("attn-embed"), embedOnly, T);
    renderEmbedStrip(document.getElementById("attn-pe"), peOnly, T);
  }

  let lastRun = null;

  function run(text) {
    const { tokens, ids, flags } = tokenize(text);
    renderTokens(tokens, flags);
    if (tokens.length === 0) {
      document.getElementById("attn-heads").innerHTML = "";
      document.getElementById("attn-bars").innerHTML = "";
      return;
    }
    const { attn, out, x, T } = forward(ids);
    lastRun = { attn, tokens, T };
    renderHeads(attn, T, tokens);
    renderBars(attn, T, tokens, 0, 0);
    renderEmbeddings(x, tokens);
  }

  // delegate clicks on heads to select a query row
  document.getElementById("attn-heads").addEventListener("click", (e) => {
    const cell = e.target.closest(".attn-cell");
    if (!cell || !lastRun) return;
    const head = parseInt(cell.dataset.head, 10);
    const i = parseInt(cell.dataset.i, 10);
    document.querySelectorAll(".attn-cell.row-active").forEach((c) => c.classList.remove("row-active"));
    document
      .querySelectorAll(`.attn-cell[data-head="${head}"][data-i="${i}"]`)
      .forEach((c) => c.classList.add("row-active"));
    renderBars(lastRun.attn, lastRun.T, lastRun.tokens, head, i);
  });

  // ===== controls =====
  const sampleSel = document.getElementById("attn-sample");
  const input = document.getElementById("attn-input");
  const runBtn = document.getElementById("attn-run");

  (M.sample_sentences || []).forEach((s) => {
    const opt = document.createElement("option");
    opt.value = s;
    opt.textContent = s;
    sampleSel.appendChild(opt);
  });
  sampleSel.addEventListener("change", () => {
    input.value = sampleSel.value;
    run(input.value);
  });
  runBtn.addEventListener("click", () => run(input.value));
  input.addEventListener("keydown", (e) => {
    if (e.key === "Enter") {
      e.preventDefault();
      run(input.value);
    }
  });

  // ===== similarity probe =====
  const simA = document.getElementById("sim-a");
  const simB = document.getElementById("sim-b");
  vocab.forEach((w, i) => {
    if (w === "<pad>" || w === "<unk>") return;
    const oa = document.createElement("option");
    oa.value = w;
    oa.textContent = w;
    const ob = oa.cloneNode(true);
    simA.appendChild(oa);
    simB.appendChild(ob);
  });
  // pick interesting defaults
  simA.value = "cat";
  simB.value = "mouse";

  function cosine(a, b) {
    const ia = word_to_id.get(a), ib = word_to_id.get(b);
    let dot = 0, na = 0, nb = 0;
    for (let i = 0; i < D; i++) {
      const x = E[ia * D + i], y = E[ib * D + i];
      dot += x * y;
      na += x * x;
      nb += y * y;
    }
    return dot / (Math.sqrt(na) * Math.sqrt(nb) + 1e-9);
  }
  function updateSim() {
    const a = simA.value, b = simB.value;
    const c = cosine(a, b);
    document.getElementById("sim-result").textContent =
      `cos(${a}, ${b}) = ${c.toFixed(3)}`;
  }
  simA.addEventListener("change", updateSim);
  simB.addEventListener("change", updateSim);
  updateSim();

  // initial render
  run(input.value);
})();

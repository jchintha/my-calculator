(() => {
  const DRAW = document.getElementById("draw");
  const DCTX = DRAW.getContext("2d");

  function resetCanvas() {
    DCTX.fillStyle = "#000";
    DCTX.fillRect(0, 0, DRAW.width, DRAW.height);
  }
  resetCanvas();
  DCTX.strokeStyle = "#fff";
  DCTX.lineWidth = 20;
  DCTX.lineCap = "round";
  DCTX.lineJoin = "round";

  let drawing = false;
  let last = null;

  function pointerPos(e) {
    const r = DRAW.getBoundingClientRect();
    const src = e.touches ? e.touches[0] : e;
    return [
      (src.clientX - r.left) * (DRAW.width / r.width),
      (src.clientY - r.top) * (DRAW.height / r.height),
    ];
  }
  function start(e) {
    e.preventDefault();
    drawing = true;
    last = pointerPos(e);
    DCTX.beginPath();
    DCTX.moveTo(last[0], last[1]);
    DCTX.lineTo(last[0] + 0.01, last[1] + 0.01);
    DCTX.stroke();
  }
  function move(e) {
    if (!drawing) return;
    e.preventDefault();
    const [x, y] = pointerPos(e);
    DCTX.beginPath();
    DCTX.moveTo(last[0], last[1]);
    DCTX.lineTo(x, y);
    DCTX.stroke();
    last = [x, y];
    scheduleInference();
  }
  function end() {
    if (!drawing) return;
    drawing = false;
    scheduleInference();
  }

  DRAW.addEventListener("mousedown", start);
  DRAW.addEventListener("mousemove", move);
  DRAW.addEventListener("mouseup", end);
  DRAW.addEventListener("mouseleave", end);
  DRAW.addEventListener("touchstart", start);
  DRAW.addEventListener("touchmove", move);
  DRAW.addEventListener("touchend", end);

  document.getElementById("clear").addEventListener("click", () => {
    resetCanvas();
    runInference();
  });

  const INPUT_PREVIEW = document.getElementById("input-preview");
  const ICTX = INPUT_PREVIEW.getContext("2d");
  ICTX.imageSmoothingEnabled = false;

  const TMP = document.createElement("canvas");
  TMP.width = 28;
  TMP.height = 28;
  const TCTX = TMP.getContext("2d");

  function preprocess() {
    TCTX.fillStyle = "#000";
    TCTX.fillRect(0, 0, 28, 28);
    TCTX.drawImage(DRAW, 0, 0, 28, 28);
    const img = TCTX.getImageData(0, 0, 28, 28);
    const x = new Float32Array(784);
    for (let i = 0; i < 784; i++) {
      x[i] = img.data[i * 4] / 255.0;
    }
    ICTX.clearRect(0, 0, INPUT_PREVIEW.width, INPUT_PREVIEW.height);
    ICTX.drawImage(TMP, 0, 0, INPUT_PREVIEW.width, INPUT_PREVIEW.height);
    return x;
  }

  const M = window.MODEL;
  const W1 = Float32Array.from(M.W1.data);
  const b1 = Float32Array.from(M.b1.data);
  const W2 = Float32Array.from(M.W2.data);
  const b2 = Float32Array.from(M.b2.data);
  const W3 = Float32Array.from(M.W3.data);
  const b3 = Float32Array.from(M.b3.data);

  function matvec(x, W, b, inDim, outDim) {
    const out = new Float32Array(outDim);
    for (let j = 0; j < outDim; j++) {
      let s = b[j];
      for (let i = 0; i < inDim; i++) {
        s += x[i] * W[i * outDim + j];
      }
      out[j] = s;
    }
    return out;
  }
  function relu(x) {
    const out = new Float32Array(x.length);
    for (let i = 0; i < x.length; i++) out[i] = x[i] > 0 ? x[i] : 0;
    return out;
  }
  function softmax(x) {
    let max = -Infinity;
    for (let i = 0; i < x.length; i++) if (x[i] > max) max = x[i];
    const out = new Float32Array(x.length);
    let sum = 0;
    for (let i = 0; i < x.length; i++) {
      out[i] = Math.exp(x[i] - max);
      sum += out[i];
    }
    for (let i = 0; i < x.length; i++) out[i] /= sum;
    return out;
  }

  function forward(x) {
    const a1 = relu(matvec(x, W1, b1, 784, 128));
    const a2 = relu(matvec(a1, W2, b2, 128, 64));
    const y = softmax(matvec(a2, W3, b3, 64, 10));
    return { a1, a2, y };
  }

  function renderGrid(canvas, values, cols, rows) {
    const ctx = canvas.getContext("2d");
    const cw = canvas.width / cols;
    const ch = canvas.height / rows;
    let max = 0;
    for (let i = 0; i < values.length; i++) if (values[i] > max) max = values[i];
    max = max || 1;
    ctx.fillStyle = "#000";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    for (let r = 0; r < rows; r++) {
      for (let c = 0; c < cols; c++) {
        const v = Math.min(1, values[r * cols + c] / max);
        const red = Math.floor(v * 255);
        const green = Math.floor(v * 160);
        ctx.fillStyle = `rgb(${red},${green},0)`;
        ctx.fillRect(c * cw, r * ch, Math.ceil(cw), Math.ceil(ch));
      }
    }
  }

  const PROBS = document.getElementById("probs");
  for (let i = 0; i < 10; i++) {
    const row = document.createElement("div");
    row.className = "prob-row";
    row.innerHTML =
      `<span class="digit">${i}</span>` +
      `<div class="bar-wrap"><div class="bar" id="bar-${i}"></div></div>` +
      `<span class="pct" id="pct-${i}">0%</span>`;
    PROBS.appendChild(row);
  }
  const BARS = [];
  const PCTS = [];
  for (let i = 0; i < 10; i++) {
    BARS.push(document.getElementById(`bar-${i}`));
    PCTS.push(document.getElementById(`pct-${i}`));
  }
  const PRED = document.getElementById("prediction");

  function clearUI() {
    PRED.textContent = "—";
    for (let i = 0; i < 10; i++) {
      BARS[i].style.width = "0%";
      BARS[i].style.background = "#3a3a4e";
      PCTS[i].textContent = "0%";
    }
    renderGrid(document.getElementById("h1"), new Float32Array(128), 16, 8);
    renderGrid(document.getElementById("h2"), new Float32Array(64), 8, 8);
  }

  function runInference() {
    const x = preprocess();
    let sum = 0;
    for (let i = 0; i < x.length; i++) sum += x[i];
    if (sum < 1.0) {
      clearUI();
      return;
    }
    const { a1, a2, y } = forward(x);
    renderGrid(document.getElementById("h1"), a1, 16, 8);
    renderGrid(document.getElementById("h2"), a2, 8, 8);
    let best = 0;
    for (let i = 1; i < 10; i++) if (y[i] > y[best]) best = i;
    PRED.textContent = String(best);
    for (let i = 0; i < 10; i++) {
      const p = y[i] * 100;
      BARS[i].style.width = `${p}%`;
      BARS[i].style.background = i === best ? "#27ae60" : "#3a3a4e";
      PCTS[i].textContent = `${p.toFixed(1)}%`;
    }
  }

  let rafPending = false;
  function scheduleInference() {
    if (rafPending) return;
    rafPending = true;
    requestAnimationFrame(() => {
      rafPending = false;
      runInference();
    });
  }

  clearUI();
})();

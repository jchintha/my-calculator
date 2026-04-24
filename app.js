const display = document.getElementById("display");
let current = "0";
let previous = null;
let operator = null;
let justEvaluated = false;

function render() {
  display.value = current;
}

function inputDigit(d) {
  if (justEvaluated) { current = "0"; justEvaluated = false; }
  current = current === "0" ? d : current + d;
  render();
}

function inputDot() {
  if (justEvaluated) { current = "0"; justEvaluated = false; }
  if (!current.includes(".")) { current += "."; render(); }
}

function clearAll() {
  current = "0"; previous = null; operator = null; justEvaluated = false;
  render();
}

function backspace() {
  if (justEvaluated) return;
  current = current.length > 1 ? current.slice(0, -1) : "0";
  render();
}

function compute(a, b, op) {
  switch (op) {
    case "+": return a + b;
    case "-": return a - b;
    case "*": return a * b;
    case "/": return b === 0 ? NaN : a / b;
  }
}

function setOperator(op) {
  if (operator && previous !== null && !justEvaluated) {
    previous = compute(previous, parseFloat(current), operator);
    current = String(previous);
  } else {
    previous = parseFloat(current);
  }
  operator = op;
  justEvaluated = true;
  render();
}

function equals() {
  if (operator === null || previous === null) return;
  const result = compute(previous, parseFloat(current), operator);
  current = Number.isFinite(result) ? String(result) : "Error";
  previous = null;
  operator = null;
  justEvaluated = true;
  render();
}

document.querySelector(".keys").addEventListener("click", (e) => {
  const btn = e.target.closest("button");
  if (!btn) return;
  if (btn.dataset.digit !== undefined) inputDigit(btn.dataset.digit);
  else if (btn.dataset.op) setOperator(btn.dataset.op);
  else if (btn.dataset.action === "dot") inputDot();
  else if (btn.dataset.action === "clear") clearAll();
  else if (btn.dataset.action === "back") backspace();
  else if (btn.dataset.action === "equals") equals();
});

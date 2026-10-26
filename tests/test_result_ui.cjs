const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");
const test = require("node:test");

function render(payload) {
  const result = { hidden: true, innerHTML: "", scrollIntoView() {} };
  const source = fs.readFileSync(path.join(__dirname, "../static/app.js"), "utf8");
  const context = vm.createContext({
    window: { DEFAULT_BANK_ID: "gpt", BANK_SUMMARIES: { gpt: {} } },
    document: { getElementById: () => result },
    payload,
  });
  vm.runInContext(source.slice(0, source.indexOf("async function analyzeManual")), context);
  vm.runInContext("renderResult(payload)", context);
  assert.equal(result.hidden, false);
  return result.innerHTML;
}

const payload = {
  prediction_name: "model-a", probability: 0.8, family_prediction_name: "GPT",
  family_probability: 1, used_outputs: 3, diagnostics: [],
  results: [
    { model: "model-a", display_name: "model-a", family_name: "GPT", probability: 0.8, profile_similarity: 0.8 },
    { model: "model-b", display_name: "model-b", family_name: "GPT", probability: 0.2, profile_similarity: 0.8 },
  ],
};

test("ambiguous results show candidates without highlighting a winner", () => {
  const html = render({ ...payload, identification: {
    status: "ambiguous", candidates: ["model-a", "model-b"], message: "无法可靠区分 <script>",
    validation: { samples: 12, correct: 9 },
  } });
  assert.match(html, /无法可靠区分/);
  assert.match(html, /model-a、model-b/);
  assert.match(html, /9\/12/);
  assert.match(html, /&lt;script&gt;/);
  assert.doesNotMatch(html, /class="winner"/);
});

test("older responses do not imply validated identity", () => {
  assert.match(render(payload), /验证不足/);
});

// Four-panel graphical abstract: (1) geothermal doublet simulated in COMSOL, (2) dataset and
// surrogate, (3) test-set accuracy and Model A vs B, (4) ensemble screening over all scenarios.
// Images come from prepare_panels.py (results/figures/graphical_abstract/); the numbers are taken
// from results/metrics/reviewer_metrics.json, the evaluation results and the manuscript.
// Usage: node build_graphical_abstract.js [output.pptx]   (needs the pptxgenjs npm package)
const pptxgen = require("pptxgenjs");
const path = require("path");

const DIR = path.join(__dirname, "..", "..", "results", "figures", "graphical_abstract");
const OUT = process.argv[2] || path.join(DIR, "Graphical_Abstract.pptx");
const TITLE = "Neural operator surrogate for 3D geothermal doublet simulation";

const NAVY = "1F3B57", HEAT = "D9822B", GREEN = "2E7D4F", PURPLE = "6A4C93", BLUE = "2E86AB";
const LINE = "8FA3B5", PALE = "F6F8FA", GREY = "4A5560";

const pres = new pptxgen();
pres.defineLayout({ name: "GA", width: 13.33, height: 7.5 });
pres.layout = "GA";
pres.theme = { headFontFace: "Calibri", bodyFontFace: "Calibri" };
const s = pres.addSlide();
s.background = { color: "FFFFFF" };

const txt = (t, o) => s.addText(t, Object.assign({ isTextBox: true, margin: 0, fontFace: "Calibri", color: NAVY }, o));
const box = (x, y, w, h, fill, line) =>
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h, fill: { color: fill }, line: { color: line || LINE, width: 1.25 }, rectRadius: 0.06 });

// table helper: header row in `col`, body rows white, first column left-aligned
function table(x, y, w, colW, head, rows, col, fs = 14) {
  const cell = (t, o = {}) => ({ text: t, options: Object.assign({ fontFace: "Calibri", fontSize: fs, color: NAVY, valign: "middle", margin: [1, 4, 1, 4] }, o) });
  const data = [head.map((h, i) => cell(h, { bold: true, color: "FFFFFF", fill: { color: col }, align: i ? "center" : "left" }))];
  for (const r of rows) data.push(r.map((c, i) => cell(c, { align: i ? "center" : "left" })));
  s.addTable(data, { x, y, w, colW, rowH: 0.33, border: { type: "solid", pt: 0.75, color: "B8C4CF" } });
}
function callout(x, y, w, h, icon, text, fs = 14) {
  box(x, y, w, h, "FFFFFF", LINE);
  if (icon === "star") s.addShape(pres.shapes.STAR_5_POINT, { x: x + 0.1, y: y + h / 2 - 0.22, w: 0.44, h: 0.44, fill: { color: "F2B632" }, line: { color: "C98F12", width: 0.75 } });
  else {
    s.addShape(pres.shapes.OVAL, { x: x + 0.1, y: y + h / 2 - 0.22, w: 0.44, h: 0.44, fill: { color: GREEN }, line: { color: GREEN } });
    txt("✓", { x: x + 0.1, y: y + h / 2 - 0.22, w: 0.44, h: 0.44, fontSize: 20, bold: true, color: "FFFFFF", align: "center", valign: "middle" });
  }
  txt(text, { x: x + 0.62, y, w: w - 0.7, h, fontSize: fs, valign: "middle" });
}

// title
txt(TITLE, { x: 0.15, y: 0.12, w: 13.03, h: 0.55, fontSize: 28, fontFace: "Cambria", color: "222222" });

// panels with numbered headings and arrows between them
const PY = 0.78, PH = 5.6;
const P = [
  { x: 0.15, w: 3.0, head: "Geothermal doublet (COMSOL)" },
  { x: 3.43, w: 3.05, head: "Dataset and surrogate" },
  { x: 6.76, w: 3.05, head: "Accuracy on 232 test scenarios" },
  { x: 10.09, w: 3.09, head: "Screening of 1,535 scenarios" },
];
P.forEach((p, i) => {
  box(p.x, PY, p.w, PH, "FFFFFF", "7F93A6");
  s.addShape(pres.shapes.OVAL, { x: p.x + 0.1, y: PY + 0.1, w: 0.46, h: 0.46, fill: { color: NAVY }, line: { color: NAVY } });
  txt(String(i + 1), { x: p.x + 0.1, y: PY + 0.1, w: 0.46, h: 0.46, fontSize: 20, bold: true, color: "FFFFFF", align: "center", valign: "middle" });
  txt(p.head, { x: p.x + 0.64, y: PY + 0.06, w: p.w - 0.72, h: 0.56, fontSize: 18, fontFace: "Cambria", color: "222222", valign: "middle" });
  if (i < 3) s.addShape(pres.shapes.RIGHT_ARROW, { x: p.x + p.w + 0.02, y: PY + PH / 2 - 0.14, w: 0.24, h: 0.28, fill: { color: NAVY }, line: { color: NAVY } });
});

// ---- 1: doublet cross-section
s.addImage({ path: path.join(DIR, "geology.png"), x: 0.25, y: 1.5, w: 2.8, h: 4.67 });

// ---- 2: dataset table and architecture flow
{
  const x = 3.55, w = 2.81;
  table(x, 1.5, w, [1.05, 1.76], ["Dataset", "COMSOL"], [
    ["Scenarios", "1,535 (LHS design)"],
    ["Split", "1,074 / 229 / 232"],
    ["Inputs", "8 maps + 14 params"],
    ["Outputs", "3D T, p; 3 well curves"],
    ["Horizon", "40 years"],
  ], GREEN, 13);
  const steps = ["Parameters → FiLM", "3D U-Net", "4D Fourier neural operator", "Well-output head"];
  let y = 3.65;
  steps.forEach((t, i) => {
    box(x, y, w, 0.4, PALE, NAVY);
    txt(t, { x, y, w, h: 0.4, fontSize: 14, bold: true, align: "center", valign: "middle" });
    if (i < steps.length - 1) s.addShape(pres.shapes.DOWN_ARROW, { x: x + w / 2 - 0.09, y: y + 0.41, w: 0.18, h: 0.13, fill: { color: NAVY }, line: { color: NAVY } });
    y += 0.54;
  });
  box(x, 5.72, 1.36, 0.55, "E9F3EC", GREEN);
  txt([{ text: "Model A", options: { bold: true, breakLine: true } }, { text: "data loss" }], { x, y: 5.72, w: 1.36, h: 0.55, fontSize: 13, align: "center", valign: "middle" });
  box(x + 1.45, 5.72, 1.36, 0.55, "F1ECF6", PURPLE);
  txt([{ text: "Model B", options: { bold: true, breakLine: true } }, { text: "+ mass, energy" }], { x: x + 1.45, y: 5.72, w: 1.36, h: 0.55, fontSize: 13, align: "center", valign: "middle" });
}

// ---- 3: accuracy tables and callouts
{
  const x = 6.88, w = 2.81;
  table(x, 1.5, w, [1.86, 0.95], ["Model A", "MAE"], [
    ["Temperature field", "0.92 K"],
    ["Pressure field", "0.019 MPa"],
    ["Cooled cells (> 1 K)", "3.0 K"],
    ["Production temp.", "0.95 K"],
    ["40-year cooling", "1.26 K"],
    ["Thermal power", "0.29 MW"],
  ], BLUE, 13);
  table(x, 3.93, w, [1.17, 0.82, 0.82], ["Rel. error", "A", "B"], [
    ["Temperature", "0.24%", "0.24%"],
    ["Pressure", "0.08%", "0.19%"],
    ["Wells", "4.90%", "4.74%"],
  ], PURPLE, 13);
  callout(x, 5.37, w, 0.9, "star", "Physics loss: no gain. COMSOL's own fields do not satisfy its residuals.", 13);
}

// ---- 4: ensemble scatter, cluster table, callout
{
  const x = 10.2, w = 2.87;
  s.addImage({ path: path.join(DIR, "decline_scatter.png"), x: x + 0.05, y: 1.45, w: 2.77, h: 2.31 });
  txt("40-year production-temperature decline", { x, y: 3.76, w, h: 0.3, fontSize: 12.5, italic: true, color: GREY, align: "center" });
  table(x, 4.1, w, [1.32, 1.55], ["Capacity, yr 40", "MWth"], [
    ["Cluster A / C / B", "24.8 / 22.6 / 20.0"],
    ["All: surrogate", "22.3"],
    ["All: COMSOL", "22.4"],
  ], HEAT, 13);
  callout(x, 5.5, w, 0.77, "check", "No scenario exceeds 10% decline (median 5.0 K).", 13);
}

// ---- conclusion banner
box(0.15, 6.53, 13.03, 0.82, "F3F7F1", "7FA37F");
for (const [d, c] of [[0.62, "B03A2E"], [0.42, "FFFFFF"], [0.22, "B03A2E"]])
  s.addShape(pres.shapes.OVAL, { x: 0.55 - d / 2, y: 6.94 - d / 2, w: d, h: d, fill: { color: c }, line: { color: "B03A2E", width: 1 } });
txt("A compact neural operator trained on COMSOL data reproduces 3D fields, well outputs and ensemble statistics at ≈ 0.05 s per scenario, "
  + "supporting preliminary geothermal screening; a physics loss should be validated against the simulator before use.",
  { x: 1.0, y: 6.55, w: 12.05, h: 0.78, fontSize: 15, valign: "middle" });

pres.writeFile({ fileName: OUT }).then(f => console.log("wrote", f));

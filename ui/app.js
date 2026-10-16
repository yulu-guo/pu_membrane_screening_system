const PROPERTIES = [
  ["tg_c", "玻璃化转变温度 Tg", "°C"],
  ["youngs_modulus_mpa", "杨氏模量", "MPa"],
  ["tensile_strength_mpa", "拉伸强度", "MPa"],
  ["elongation_at_break_percent", "断裂伸长率", "%"],
  ["td5_c", "初始分解温度 Td5%", "°C"],
  ["max_decomposition_c", "最大分解温度", "°C"],
];

const state = {
  catalog: [],
  roles: [],
  prediction: null,
  screening: null,
  componentRow: 0,
};

const $ = (selector) => document.querySelector(selector);
const ROLE_LABELS = {
  isocyanate: "二异氰酸酯",
  polyol: "多元醇",
  chain_extender: "扩链剂",
  crosslinker: "交联剂",
};

function setStatus(kind, text, chip) {
  const dot = $("#status-dot");
  dot.className = "dot";
  if (kind) dot.classList.add(kind);
  $("#status-text").textContent = text;
  if (chip !== undefined) $("#model-chip").textContent = chip;
}

function showError(message) {
  const box = $("#error");
  box.textContent = message;
  box.classList.remove("hidden");
}

function clearError() {
  $("#error").classList.add("hidden");
}

function setLoading(loading) {
  $("#loading").classList.toggle("hidden", !loading);
  document.querySelectorAll(".primary").forEach((button) => {
    button.disabled = loading;
  });
}

async function api(path, options = {}) {
  const response = await fetch(path, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  const payload = await response.json();
  if (!response.ok) throw new Error(payload.error || `HTTP ${response.status}`);
  return payload;
}

function format(value, digits = 3) {
  if (!Number.isFinite(Number(value))) return "--";
  return Number(value).toLocaleString("en-US", {
    minimumFractionDigits: digits,
    maximumFractionDigits: digits,
  });
}

function switchTab(name) {
  document.querySelectorAll(".tab").forEach((button) => {
    button.classList.toggle("active", button.dataset.tab === name);
  });
  document.querySelectorAll(".panel").forEach((panel) => {
    panel.classList.remove("active");
  });
  $(`#tab-${name}`).classList.add("active");
  if (name === "data") {
    $("#empty").classList.add("hidden");
    $("#prediction").classList.add("hidden");
    $("#screening").classList.add("hidden");
  }
}

function monomerOptions(role, selectedId, query = "") {
  const normalized = query.trim().toLowerCase();
  const relevant = state.catalog.filter((item) => {
    if (item.role !== role) return false;
    if (!normalized) return true;
    const haystack = [
      item.id,
      item.name,
      item.family,
      item.smiles,
      item.molecular_formula,
    ].join(" ").toLowerCase();
    return haystack.includes(normalized);
  });
  return relevant.map((item) => (
    `<option value="${item.id}" ${item.id === selectedId ? "selected" : ""}>`
    + `${item.id} | ${item.name} | ${item.molecular_formula || ""}</option>`
  )).join("");
}

function roleOptions(selectedRole) {
  const allowed = ["isocyanate", "polyol", "chain_extender", "crosslinker"];
  const roles = allowed.filter((role) => state.roles.includes(role));
  return roles.map((role) => (
    `<option value="${role}" ${role === selectedRole ? "selected" : ""}>${ROLE_LABELS[role] || role}</option>`
  )).join("");
}

function updateMonomerSelect(row) {
  const role = row.querySelector(".component-role").value;
  const select = row.querySelector(".component-monomer");
  const search = row.querySelector(".component-search").value;
  const previous = select.value;
  select.innerHTML = monomerOptions(role, previous, search);
  if (select.options.length === 0) {
    select.innerHTML = `<option value="">没有匹配的单体</option>`;
  }
  updateComponentInfo(row);
}

function updateComponentInfo(row) {
  const monomerId = row.querySelector(".component-monomer").value;
  const info = row.querySelector(".component-info");
  const monomer = state.catalog.find((item) => item.id === monomerId);
  if (!monomer) {
    info.textContent = "请选择单体";
    return;
  }
  info.innerHTML = `
    <strong>${monomer.name}</strong>
    <span>分子式：${monomer.molecular_formula || "--"}</span>
    <span>分子量：${format(monomer.molecular_weight, 2)}</span>
    <span>类别：${monomer.family || monomer.role}</span>
    <span>反应位点：NCO ${monomer.nco_site_count ?? 0} / OH ${monomer.oh_site_count ?? 0} / NH2 ${monomer.nh2_site_count ?? 0}</span>
    <span>SMILES：${monomer.smiles}</span>
    <span>来源：${monomer.source || "--"}</span>
  `;
}

function addComponentRow(role = "isocyanate", monomerId = "", count = 1) {
  state.componentRow += 1;
  const row = document.createElement("div");
  row.className = "component-row";
  row.dataset.row = String(state.componentRow);
  row.innerHTML = `
    <div class="component-main">
      <select class="component-role">${roleOptions(role)}</select>
      <input class="component-search" type="text" placeholder="搜索编号/名称/分子式/SMILES">
      <select class="component-monomer">${monomerOptions(role, monomerId)}</select>
      <input class="component-count" type="number" min="1" max="12" value="${count}">
      <button class="remove-component" type="button">×</button>
    </div>
    <div class="component-info"></div>
  `;
  row.querySelector(".component-role").addEventListener("change", () => updateMonomerSelect(row));
  row.querySelector(".component-search").addEventListener("input", () => updateMonomerSelect(row));
  row.querySelector(".component-monomer").addEventListener("change", () => updateComponentInfo(row));
  row.querySelector(".remove-component").addEventListener("click", () => {
    if (document.querySelectorAll(".component-row").length > 1) row.remove();
  });
  $("#component-list").appendChild(row);
  updateComponentInfo(row);
}

function defaultFormulation() {
  addComponentRow("isocyanate", "ISO-MDI", 2);
  addComponentRow("polyol", "POL-PEG400", 1);
  addComponentRow("chain_extender", "CE-BDO", 1);
}

function switchReactionFamily() {
  $("#component-list").replaceChildren();
  $("#add-component").disabled = false;
  $("#arrangement").disabled = false;
  $("#nco-index").disabled = false;
  $("#pla-ptmc-ratio").disabled = false;
  defaultFormulation();
}

function applySoftSegmentRatio() {
  const [plaCount, ptmcCount] = $("#pla-ptmc-ratio").value
    .split(":")
    .map(Number);
  document.querySelectorAll(".component-row").forEach((row) => {
    const monomerId = row.querySelector(".component-monomer").value;
    if (monomerId.startsWith("POL-PLA")) {
      row.querySelector(".component-count").value = String(plaCount);
    } else if (monomerId.startsWith("POL-PTMC")) {
      row.querySelector(".component-count").value = String(ptmcCount);
    }
  });
}

function renderTargets() {
  $("#target-fields").innerHTML = PROPERTIES.map(([key, label]) => `
    <div class="target-row">
      <span>${label}</span>
      <input type="number" step="any" data-target="${key}" placeholder="--">
      <input type="number" min="0" step="0.5" value="0" data-weight="${key}">
    </div>
  `).join("");
}

function processingPayload() {
  return {
    nco_oh_index: Number($("#nco-index").value),
    synthesis_step: Number($("#synthesis-step").value),
    forming_method: $("#forming-method").value,
    reaction_temperature_1_c: Number($("#temp1").value),
    reaction_temperature_2_c: Number($("#temp2").value),
    cross_section_area_mm2: Number($("#cross-area").value),
    strain_rate_min_inv: Number($("#strain-rate").value),
    soft_segment_mn_g_mol: $("#soft-mn").value
      ? Number($("#soft-mn").value)
      : null,
  };
}

function componentPayload() {
  return [...document.querySelectorAll(".component-row")].map((row) => ({
    monomer_id: row.querySelector(".component-monomer").value,
    count: Number(row.querySelector(".component-count").value || 1),
  })).filter((item) => item.monomer_id);
}

function renderPrediction(payload) {
  state.prediction = payload;
  state.screening = null;
  $("#empty").classList.add("hidden");
  $("#screening").classList.add("hidden");
  $("#prediction").classList.remove("hidden");
  $("#sequence-line").textContent = payload.sequence
    ? `排列顺序：${payload.sequence.join(" → ")}`
    : `结构：${payload.structure_name || "已知聚合物"}`;
  $("#polymer-smiles").textContent = payload.polymer_smiles;

  const structure = payload.structure;
  $("#structure-meta").innerHTML = `
    <span> MW ${format(structure.molecular_weight, 1)}</span>
    <span> 芳环 ${structure.aromatic_rings}</span>
    <span> 可旋转键 ${structure.rotatable_bonds}</span>
    <span> 氨基甲酸酯基 ${structure.urethane_groups}</span>
    <span> 脲基 ${structure.urea_groups}</span>
    <span> 连接点 ${structure.attachment_points}</span>
  `;

  $("#property-grid").innerHTML = PROPERTIES.map(([key, label, unit]) => `
    <div class="property">
      <strong>${format(payload.properties[key], key === "tg_c" ? 1 : 3)}</strong>
      <span>${label} | ${unit}</span>
    </div>
  `).join("");

  const warnings = [...payload.warnings];
  warnings.push(...(payload.reference_notes || []));
  if (payload.feature_compiler_approximate) {
    warnings.push("组成特征由单体描述符近似编译，机械性能应按筛选结果使用。");
  }
  if (payload.family_details) {
    warnings.push(
      `PLA/PTMC 家族模型数据来源：${payload.family_details.source_dois.join("；")}`,
    );
  }
  $("#warnings").innerHTML = warnings.map((text) => `<div class="warning">${text}</div>`).join("");

  const featureEntries = Object.entries(payload.compiled_features);
  if (payload.family_details) {
    const family = payload.family_details;
    featureEntries.push(
      ["PLA/PTMC 软段 Tg（组成模型）", family.soft_segment_tg_c],
      ["PTMC 质量分数", family.ptmc_mass_fraction],
      ["软段多元醇 Mn g/mol", family.soft_segment_mn_g_mol],
      ["PLA 段 Tg（链端修正后）", family.lactide_tg_c],
      ["PTMC 段 Tg（链端修正后）", family.carbonate_tg_c],
      ["Gordon-Taylor k（文献拟合）", family.gordon_taylor_k],
      ["Flory-Fox K（文献拟合）", family.flory_fox_k],
      ["链端修正位移 °C", family.chain_end_shift_c],
      ["家族 Td5%", family.td5_c],
      ["家族 Tdmax", family.tdmax_c],
      ["通用结构模型 Tg（对照）", family.structure_model_tg_c],
    );
  }
  $("#feature-table").innerHTML = featureEntries
    .map(([key, value]) => `
      <div class="data-cell"><span>${key}</span><span>${format(value, 4)}</span></div>
    `).join("");
}

function renderScreening(payload) {
  state.screening = payload;
  state.prediction = null;
  $("#empty").classList.add("hidden");
  $("#prediction").classList.add("hidden");
  $("#screening").classList.remove("hidden");
  $("#screen-summary").textContent = `检查 ${payload.examined_candidates} 个配方，返回 ${payload.returned_candidates} 个候选。评分越低越接近目标。`;

  const columns = [
    "#", "单体数量", "排列", "评分", "Tg °C", "E MPa", "TS MPa", "EB %",
    "Td5% °C", "Tdmax °C", "单体序列", "SMILES",
  ];
  $("#screen-table thead").innerHTML = `<tr>${columns.map((label) => `<th>${label}</th>`).join("")}</tr>`;
  $("#screen-table tbody").innerHTML = payload.results.map((item, index) => `
    <tr>
      <td>${index + 1}</td>
      <td>${item.monomers.length}</td>
      <td>${item.sequence.join(" → ")}</td>
      <td>${format(item.score, 4)}</td>
      <td>${format(item.properties.tg_c, 1)}</td>
      <td>${format(item.properties.youngs_modulus_mpa, 3)}</td>
      <td>${format(item.properties.tensile_strength_mpa, 3)}</td>
      <td>${format(item.properties.elongation_at_break_percent, 3)}</td>
      <td>${format(item.properties.td5_c, 1)}</td>
      <td>${format(item.properties.max_decomposition_c, 1)}</td>
      <td>${item.monomers.map((monomer) => monomer.name).join(" → ")}</td>
      <td class="smiles" title="${item.polymer_smiles}">${item.polymer_smiles}</td>
    </tr>
  `).join("");
}

async function runPrediction() {
  clearError();
  setLoading(true);
  try {
    const result = await api("/api/predict", {
      method: "POST",
      body: JSON.stringify({
        reaction_family: $("#reaction-family").value,
        components: componentPayload(),
        arrangement: $("#arrangement").value,
        processing: processingPayload(),
        random_seed: 42,
      }),
    });
    renderPrediction(result);
  } catch (error) {
    showError(error.message);
  } finally {
    setLoading(false);
  }
}

async function runScreening() {
  clearError();
  const targets = {};
  const weights = {};
  document.querySelectorAll("[data-target]").forEach((input) => {
    if (input.value !== "") targets[input.dataset.target] = Number(input.value);
  });
  document.querySelectorAll("[data-weight]").forEach((input) => {
    weights[input.dataset.weight] = Number(input.value || 0);
  });
  if (!Object.keys(targets).length) {
    showError("请至少输入一个目标值并设置大于 0 的权重。");
    return;
  }
  setLoading(true);
  try {
    const result = await api("/api/screen", {
      method: "POST",
      body: JSON.stringify({
        targets,
        weights,
        roles: ["isocyanate", "polyol", "chain_extender"],
        number_of_monomers: Number($("#screen-monomers").value),
        max_candidates: Number($("#screen-count").value),
        processing: processingPayload(),
        arrangement: $("#screen-arrangement").value,
        random_seed: 42,
      }),
    });
    renderScreening(result);
  } catch (error) {
    showError(error.message);
  } finally {
    setLoading(false);
  }
}

function exportCsv() {
  if (!state.screening) return;
  const rows = state.screening.results.map((item, index) => ({
    rank: index + 1,
    score: item.score,
    sequence: item.sequence.join(" | "),
    monomer_names: item.monomers.map((monomer) => monomer.name).join(" | "),
    polymer_smiles: item.polymer_smiles,
    Tg_C: item.properties.tg_c,
    Youngs_modulus_MPa: item.properties.youngs_modulus_mpa,
    Tensile_strength_MPa: item.properties.tensile_strength_mpa,
    Elongation_at_break_percent: item.properties.elongation_at_break_percent,
  }));
  const headers = Object.keys(rows[0] || {});
  const csv = [
    headers.join(","),
    ...rows.map((row) => headers.map((header) => (
      `"${String(row[header] ?? "").replaceAll('"', '""')}"`
    )).join(",")),
  ].join("\n");
  const blob = new Blob([csv], { type: "text/csv;charset=utf-8" });
  const link = document.createElement("a");
  link.href = URL.createObjectURL(blob);
  link.download = "pu_membrane_candidates.csv";
  link.click();
  URL.revokeObjectURL(link.href);
}

async function initialize() {
  renderTargets();
  try {
    const [catalog, health] = await Promise.all([
      api("/api/catalog"),
      api("/api/health"),
    ]);
    state.catalog = catalog.monomers;
    state.roles = catalog.roles;
    defaultFormulation();
    const tgR2 = health.metrics?.tg?.holdout?.r2;
    setStatus(
      "ready",
      `${health.monomers} 个单体 | Tg 模型已加载`,
      tgR2 ? `Tg R² ${format(tgR2, 2)}` : "ready"
    );
    $("#data-status").innerHTML = `
      <div class="data-cell"><span>Tg 训练数据</span><span>${health.metrics?.tg?.rows_total ?? "--"} 条</span></div>
      <div class="data-cell"><span>含聚氨酯/PU脲结构</span><span>${health.metrics?.tg?.rows_polyurethane ?? "--"} 条</span></div>
      <div class="data-cell"><span>力学训练数据</span><span>643 条</span></div>
      <div class="data-cell"><span>Tg 测试 R²</span><span>${format(tgR2, 3)}</span></div>
      <div class="data-cell"><span>杨氏模量测试 R²</span><span>${format(health.metrics?.mechanical?.logYM?.r2, 3)}</span></div>
      <div class="data-cell"><span>拉伸强度测试 R²</span><span>${format(health.metrics?.mechanical?.logTS?.r2, 3)}</span></div>
      <div class="data-cell"><span>断裂伸长率测试 R²</span><span>${format(health.metrics?.mechanical?.logEB?.r2, 3)}</span></div>
      <div class="data-cell"><span>网络 Td 模型 R²</span><span>${format(health.metrics?.public_thermal?.general_td?.holdout?.r2, 3)}</span></div>
      <div class="data-cell"><span>网络 Td5 模型 R²</span><span>${format(health.metrics?.public_thermal?.general_td5?.holdout?.r2, 3)}</span></div>
    `;
  } catch (error) {
    setStatus("error", "加载失败", "offline");
    showError(error.message);
  }
}

document.querySelectorAll(".tab").forEach((button) => {
  button.addEventListener("click", () => switchTab(button.dataset.tab));
});
$("#add-component").addEventListener("click", () => addComponentRow("polyol", "", 1));
$("#reaction-family").addEventListener("change", switchReactionFamily);
$("#pla-ptmc-ratio").addEventListener("change", applySoftSegmentRatio);
$("#predict-button").addEventListener("click", runPrediction);
$("#screen-button").addEventListener("click", runScreening);
$("#export-csv").addEventListener("click", exportCsv);
$("#copy-smiles").addEventListener("click", async () => {
  if (!state.prediction) return;
  await navigator.clipboard.writeText(state.prediction.polymer_smiles);
  $("#copy-smiles").textContent = "已复制";
  setTimeout(() => { $("#copy-smiles").textContent = "复制 SMILES"; }, 1200);
});

initialize();

/* Qwen-Image2.1 三模式工作台（t2i / i2i / edit）
 * ⚠️ 页面 URL 为 /ui/，同源接口调用必须用【根绝对路径】（'/capabilities' 而非 'capabilities'），
 *    否则会被解析为 /ui/capabilities → 404（平台规则 §六.12 实测教训）。
 */
(() => {
  "use strict";
  const $ = (id) => document.getElementById(id);

  const MODE_TIPS = {
    t2i: "纯文本生成图像；宽高可自选（含 2K 档位）",
    i2i: "多张参考图融合 / 参考构图（建议 2–4 张，官方上限 10）",
    edit: "源图 + 指令定向修改；可选圈选 / 独立蒙版做局部编辑",
  };
  const COLORS = ["#3b82f6", "#ef4444", "#22c55e", "#eab308"];
  const MAX_REF = 10;
  const SOFT_REF = 4;
  // 尺寸对齐（任务 56）：管线 multiple_of = vae_scale_factor(16) × 2 = 32，
  // 非 32 对齐会被**静默向下取整**（1360→1344）→ 前端统一吸附到 32，避免"显示≠实际"
  const SIZE_ALIGN = 32;
  const SIZE_MIN = 256;
  const SIZE_MAX = 2048;

  let mode = "t2i";
  let schemes = [];
  let variants = [];
  let images = [];       // [{name, dataUrl, w, h, el}]
  let circles = [];      // [{cx, cy, r, color}] —— 归一化坐标（相对图宽）
  let colorIdx = 0;
  let drag = null;
  let lastResultUrl = "";
  let lastBeforeUrl = "";

  // ===== 基础 =====
  function setState(state) {
    const dot = $("state-dot");
    dot.className = "dot " + (state === "ready" ? "ready" : state === "error" ? "error" : "loading");
    $("state-text").textContent = state;
  }
  // ===== 反馈（v2.1 显眼化）=====
  // 普通提示：中性灰字
  function setHint(msg) {
    const el = $("hint");
    el.className = "hint";
    el.textContent = msg || "";
  }
  function setImageHint(msg, isErr) {
    const el = $("image-hint");
    el.className = "hint" + (isErr ? " hint-error" : "");
    el.textContent = msg || "";
  }
  // 错误提示：顶部 toast + 红字 + 出错控件闪红框（用户之前误以为"没反应"）
  function toastWrap() {
    let w = document.getElementById("toast-wrap");
    if (!w) {
      w = document.createElement("div");
      w.id = "toast-wrap";
      w.className = "toast-wrap";
      document.body.appendChild(w);
    }
    return w;
  }
  function showToast(msg) {
    const el = document.createElement("div");
    el.className = "toast";
    el.setAttribute("role", "alert");
    el.textContent = msg;
    toastWrap().appendChild(el);
    setTimeout(() => el.remove(), 5000);
  }
  function setErr(msg, anchor) {
    const el = $("hint");
    const text = humanize(msg);          // 内部 id → UI 标签（R10）
    el.className = "hint hint-error";
    el.textContent = text;
    showToast(text);
    if (anchor) {
      anchor.classList.add("flash-error");
      setTimeout(() => anchor.classList.remove("flash-error"), 1700);
      try { anchor.scrollIntoView({ block: "center", behavior: "smooth" }); } catch (_) { /* 忽略 */ }
    }
  }
  function setStage(msg) { $("stage").textContent = msg || ""; }

  // ===== 单一档位入口（任务 51 · 方案 A）=====
  //   模型档位下拉 = 唯一选择入口（不可用项灰显且 disabled）
  //   精度档       = 只读标签，由所选模型**自动带出**（variant→scheme 近乎 1:1）
  //   发请求时必带成对的 (variant, scheme)，从根上杜绝"界面显示 × 实际发送"不一致
  function schemeForVariant(variant) {
    const s = schemes.find((x) => (x.applicable_variants || []).includes(variant));
    return s ? s.id : "";
  }
  function schemeLabel(sid) {
    const s = schemes.find((x) => x.id === sid);
    return s ? s.name : (sid || "—");
  }
  function resolvePair() {
    const variant = $("variant-select").value;
    return { variant, scheme: schemeForVariant(variant) };
  }
  function applyPair() {
    const pair = resolvePair();
    $("scheme-badge").textContent = pair.scheme ? schemeLabel(pair.scheme) : "—";
    return pair;
  }

  // ===== 档位联动参数默认值（任务 54 / 任务 56-V2 修订）=====
  // 权威依据：
  //   ① 引擎 engine.py::_effective_steps —— `turbo` 变体走 8 步蒸馏契约，其余 40 步；
  //   ② 任务 52 全档位回归报告 —— T1(bf16，fp16 compute) 在 8 步**严重欠收敛**（须 ≥20–30 步）；
  //   ③ **任务 56-V2 步数 A/B 实证（覆盖 ② 中对 T2/T3 的结论）**：同 prompt + 同 seed 只变步数，
  //      场景类（杯子+木桌+暖光窗影）8 步 lapVar=84.5 且**稳定复现半透明"玻璃板"伪影**，
  //      20 步 139.7 伪影消失、40 步 223.1；彩色产品（红鞋+黄底）8 步 187 发虚拖影、40 步 599 锐利。
  //      → 结论：**"8 步清晰稳定"只适用于纯色背景的简单产品，不适用于场景/多材质/细节丰富内容**。
  //      故 T2/T3 默认步数由 8 上调到 **20**（清晰基线），复杂内容由用户/提示调至 30–40。
  //      turbo 仍为 8（8 步蒸馏契约，模型本身即按 8 步训练，不在此结论范围内）。
  const STEPS_BY_VARIANT = {
    "qwen-image-21-official-bf16": 40,       // 全精度画质基线；8 步欠收敛
    "qwen-image-21-official-gguf-q8": 20,    // ★ 主力档；20 步清晰基线（原 8 步 → 场景伪影）
    "qwen-image-21-official-gguf-q4": 20,    // 省显存档；同为 fp32 compute，8 步同样欠收敛
    "qwen-image-21-turbo-gguf-q8": 8,        // 8 步蒸馏契约
    "qwen-image-21-uncensored-gguf-q8": 40,  // 社区重打包、非 Turbo → 保守
    "qwen-image-21-official-int8": 40,       // 二期档 → 保守
  };
  function stepsForVariant(variant) {
    if (/turbo/i.test(variant || "")) return 8;      // 蒸馏契约优先
    const n = STEPS_BY_VARIANT[variant];
    return (typeof n === "number") ? n : 40;         // 未知档位兜底 40（保守）
  }
  function stepsHint(variant) {
    if (/turbo/i.test(variant || "")) return "8 步（8 步蒸馏契约）";
    return stepsForVariant(variant) === 20
      ? "20 步（产品图清晰基线；场景/多材质建议 30–40 步）"
      : "40 步（该档 8 步欠收敛，需 ≥20–30 步）";
  }
  function variantAlias(variant) {
    const v = variants.find((x) => x.name === variant);
    return (v && (v.alias || v.name)) || variant || "—";
  }
  // 只填充**与档位强相关**的参数：步数 + CFG
  //   （宽/高/种子/透明 保持用户设置 —— 换档不应清掉用户有意设定的比例或复现种子）
  function applyTierDefaults(pair) {
    const steps = stepsForVariant(pair.variant);
    $("steps").value = steps;
    $("cfg").value = "1.0";                          // Qwen-Image-2.1 发布路径恒为 1.0
    $("tier-hint").textContent =
      `档位默认：${variantAlias(pair.variant)} → ${stepsHint(pair.variant)} · CFG 1.0`;
  }
  function renderVariantOptions() {
    const prev = $("variant-select").value;
    const usable = variants.filter((v) => v.available !== false);
    $("variant-select").innerHTML = variants.map((v) => {
      const dis = v.available === false ? " disabled" : "";
      const tail = v.available === false ? "（不可用）" : "";
      return `<option value="${v.name}"${dis}>${v.alias || v.name}${tail}</option>`;
    }).join("") || `<option value="">（无）</option>`;
    // 显式回填选择（不依赖浏览器"默认选中第一项"的隐式行为）
    // 任务 56-V2 实测修正：默认不应取 /capabilities 列表首项（本机首项是 bf16/T1，
    //   会导致"一开页就是 FP 档"，首次生成时静默触发整档切换 → 30GB 重载 + 质量档不符）。
    //   应以 **默认精度档（scheme.default=true）** 对应的可用模型为准（本引擎 = T2 Q8）。
    const defScheme = schemes.find((s) => s.default) || null;
    const defVariant = defScheme
      ? (defScheme.applicable_variants || []).find((n) => usable.some((u) => u.name === n))
      : "";
    const want = (prev && variants.some((v) => v.name === prev && v.available !== false))
      ? prev : (defVariant || (usable[0] ? usable[0].name : ""));
    if (want) $("variant-select").value = want;
    applyPair();
  }

  // 内部 id → UI 标签（任务 51 · R10）：错误信息里不再出现裸 `t2_gguf_q8`
  function humanize(msg) {
    let out = String(msg == null ? "" : msg);
    schemes.forEach((s) => { out = out.split(s.id).join(`「${s.name}」`); });
    variants.forEach((v) => { out = out.split(v.name).join(`「${v.alias || v.name}」`); });
    return out;
  }

  // 选择/模式变更即清除上次的错误与状态行（避免"上次失败"被误读成"本次选择矛盾"）
  function clearFeedback() {
    const h = $("hint");
    if ((h.className || "").includes("hint-error")) { h.className = "hint"; h.textContent = ""; }
    setStage("");
  }

  async function jget(url) {
    const r = await fetch(url, { cache: "no-store" });
    if (!r.ok) throw new Error(url + " → " + r.status);
    return r.json();
  }

  // ===== 元数据 =====
  async function loadMeta() {
    try {
      const [cap, sch, st] = await Promise.all([
        jget("/capabilities"), jget("/schemes"), jget("/status"),
      ]);
      variants = cap.variants || [];
      schemes = sch.schemes || [];

      // 单一档位入口（任务 51）：模型下拉（不可用灰显）+ 只读精度标签自动配对
      renderVariantOptions();
      if (!window.__variantBound) {
        window.__variantBound = true;
        // 档位切换 → 参数自动填充（任务 54：模型 / 精度档位决定采样步数）
        $("variant-select").addEventListener("change", () => {
          applyTierDefaults(applyPair());
          clearFeedback();
        });
      }
      // 首次渲染对齐一次；**不在「加载模型 / 生成」时重置**（避免清掉用户手改的步数/宽高）
      if (!window.__tierInit) {
        window.__tierInit = true;
        applyTierDefaults(resolvePair());
      }

      const hw = sch.hardware || {};
      const cap2 = hw.capability ? hw.capability.join(".") : "无 GPU";
      $("hw-badge").textContent =
        `${hw.device_name || "CPU"} · sm_${cap2} · ${hw.vram_gb || 0}GB` +
        (hw.bf16_tensor_core ? " · bf16✓" : " · 无 bf16 张量核→运行时降 fp16/fp32") +
        (hw.has_int_mm ? " · int8✓" : "");

      renderRatios(cap.supported_sizes || []);
      setState(st.state || "idle");
    } catch (e) {
      setErr("元数据加载失败：" + e.message);
    }
  }

  function renderRatios(sizes) {
    const names = {
      "1024x1024": "1:1", "1024x768": "4:3", "768x1024": "3:4",
      "1664x928": "16:9", "928x1664": "9:16",
      "1664x704": "21:9", "704x1664": "9:21",
      "2048x2048": "1:1 2K", "1152x896": "9:7", "896x1152": "7:9",
    };
    $("ratio-row").innerHTML = sizes.slice(0, 7).map((s) => {
      const [w, h] = s.split("x");
      return `<button class="ghost ratio" data-w="${w}" data-h="${h}">${names[s] || s}</button>`;
    }).join("");
  }

  // ===== 加载模型 =====
  async function doLoad() {
    const pair = applyPair();
    const variant = pair.variant;
    if (!variant) return setErr("无可用模型档位", $("variant-select"));
    setHint(`加载中…（${variant} · ${schemeLabel(pair.scheme)}，首次可能较慢）`);
    setState("loading");
    try {
      const r = await fetch("/load", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ variant }),
      });
      const data = await r.json();
      if (!r.ok) { setErr("加载失败：" + (data.detail || r.status), $("variant-select")); setState("error"); return; }
      setState("ready");
      setHint(`已加载 ${variant}（${schemeLabel(pair.scheme)}，${data.load_time}s，显存 ${data.vram_used || "—"}）`);
    } catch (e) {
      setState("error"); setErr("加载异常：" + e.message, $("variant-select"));
    }
  }

  // ===== 模式切换 =====
  function setMode(m) {
    mode = m;
    document.querySelectorAll(".mode-tab").forEach((b) =>
      b.classList.toggle("active", b.dataset.mode === m));
    $("mode-tip").textContent = MODE_TIPS[m] || "";

    const needImg = m !== "t2i";
    // 任务 51 · T4.1：三模式为主轴 —— 只显示该模式需要的控件
    //   t2i  : 提示词 + 尺寸/比例 + 透明；无参考图、无局部编辑、无扩图
    //   i2i  : 参考图(2–10) + 提示词 + 尺寸/比例（默认跟随末张参考图）；无局部编辑、无透明、无扩图
    //   edit : 源图(1) + 局部编辑 + 扩图 + 提示词 + 尺寸/比例（默认跟随源图）；无透明
    // 任务 56：尺寸/比例不再对 i2i/edit 整行隐藏 —— 改为「跟随条件图」复选（默认勾选，
    //          勾选时宽高禁用；取消勾选即可显式指定，后端一直支持）
    $("image-block").hidden = !needImg;
    $("image-title").textContent = m === "i2i" ? "参考图（可多张）" : "源图（1 张）";
    $("mask-block").hidden = m !== "edit";
    $("expand-block").hidden = m !== "edit";
    $("size-row").hidden = false;
    $("ratio-row").hidden = false;
    $("follow-lbl").hidden = !needImg;
    $("follow-text").textContent = m === "i2i" ? "跟随参考图尺寸（末张）" : "跟随源图尺寸";
    $("transparent-lbl").hidden = needImg;
    applyFollowSize();
    applyExpand();
    renderThumbs();
    renderCanvas();
    clearFeedback();                          // 模式变更即清除上次提示（T3.2）
  }

  // ===== 尺寸：跟随 / 显式（任务 56）=====
  function applyFollowSize() {
    const needImg = mode !== "t2i";
    const follow = $("follow-size").checked;
    const dis = (needImg && follow) || expandOn();
    $("width").disabled = dis;
    $("height").disabled = dis;
    updateSizeHint();
  }

  function updateSizeHint() {
    const el = $("size-hint");
    if (mode === "t2i") { el.className = "hint"; el.textContent = ""; return; }
    el.className = "hint";
    if (expandOn()) {
      const s = expandPreview();
      el.textContent = s
        ? `扩图生效：输出画布 = ${s}（忽略宽高输入；与局部编辑互斥，已切到「整图编辑」）`
        : "扩图生效：请先选择 1 张源图，才能计算扩展后画布尺寸";
    } else if ($("follow-size").checked) {
      el.textContent = "跟随条件图：输出尺寸 ≈ 条件图比例（按 1024² 等比推算，不改构图）";
    } else {
      el.textContent = `显式尺寸生效：改比例 = 模型按新画布重新构图（非裁剪/补边）；宽高须 ${SIZE_ALIGN} 对齐、${SIZE_MIN}–${SIZE_MAX}`;
    }
  }

  // 尺寸吸附（任务 56 实测修正）：必须**向上取整** —— 与后端 image_staging.align_up() 同口径。
  //   原实现用 Math.round：输入 1000 → 992（把用户要的尺寸**变小**），与后端语义相反；
  //   改为 ceil：1000 → 1024、1360 → 1376（永不小于用户请求；2048 本身是 32 倍数，不会越界）。
  function snapAlign(v) { return Math.ceil(Number(v) / SIZE_ALIGN) * SIZE_ALIGN; }

  // ===== 扩图（任务 56）=====
  function expandOn() { return !$("expand-block").hidden && $("expand-on").checked; }

  function expandPreview() {
    if (!images.length) return "";
    const im = images[0];
    const aw = Math.ceil(im.w / SIZE_ALIGN) * SIZE_ALIGN;
    const ah = Math.ceil(im.h / SIZE_ALIGN) * SIZE_ALIGN;
    const a = Number($("expand-amt").value) || 0;
    const d = $("expand-dir").value;
    if (d === "all") return `${aw + 2 * a}×${ah + 2 * a}`;
    if (d === "left" || d === "right") return `${aw + a}×${ah}`;
    return `${aw}×${ah + a}`;
  }

  function applyExpand() {
    const on = expandOn();
    $("expand-dir").disabled = !on;
    $("expand-amt").disabled = !on;
    if (mode === "edit") {
      // 扩图与「局部编辑（圈选 / 独立蒙版）」互斥：圈选坐标以原图为基准，
      // 与"扩展后的画布"语义冲突 → 启用扩图时强制"整图编辑"
      if (on) {
        $("local-mode").value = "none";
        $("local-mode").disabled = true;
      } else {
        $("local-mode").disabled = false;
      }
      renderCanvas();
    }
    applyFollowSize();
  }

  // ===== 输入图管理 =====
  function renderThumbs() {
    const box = $("thumbs");
    box.innerHTML = images.map((im, i) => `
      <div class="thumb">
        <span class="idx">图${i + 1}</span>
        <img src="${im.dataUrl}" title="${im.name}（${im.w}×${im.h}）" />
        <button class="del" data-i="${i}">✕</button>
      </div>`).join("");
    if (mode === "i2i") {
      const n = images.length;
      if (n === 0) setImageHint("需要 2–10 张参考图", true);
      else if (n < 2) setImageHint(`还需 ${2 - n} 张（图生图需 ≥2 张）`, true);
      else if (n > SOFT_REF) setImageHint(`已 ${n} 张（>${SOFT_REF} 张指令遵循度可能下降）`, false);
      else setImageHint(`已 ${n} 张，指代用「图1 / 图2…」`, false);
    } else if (mode === "edit") {
      const n = images.length;
      if (n === 0) setImageHint("需要 1 张源图", true);
      else if (n > 1) setImageHint(`已选 ${n} 张；图像编辑只用第 1 张（切到「图生图」可用全部）`, false);
      else setImageHint("已选源图；输出尺寸默认跟随源图", false);
    }
  }

  function addFiles(files) {
    const list = Array.from(files || []).filter((f) => f.type.startsWith("image/"));
    if (!list.length) return;
    list.forEach((file) => {
      const reader = new FileReader();
      reader.onload = () => {
        const el = new Image();
        el.onload = () => {
          // 任务 51 · T4.2：不再在上传时裁剪（原 `if (mode === "edit") images = []`
          // 会在模式切换时把已选图**静默丢掉**）；改由 buildImagesPayload 提交时取第 1 张
          images.push({ name: file.name, dataUrl: reader.result, w: el.naturalWidth, h: el.naturalHeight, el });
          if (images.length > MAX_REF) { images = images.slice(0, MAX_REF); setErr(`最多 ${MAX_REF} 张`, $("image-block")); }
          renderThumbs(); renderCanvas(); applyExpand();   // 源图变化 → 刷新扩图尺寸预告
          resetCircles();
        };
        el.src = reader.result;
      };
      reader.readAsDataURL(file);
    });
  }

  function resetCircles() { circles = []; }

  // ===== 局部编辑画布 =====
  function renderCanvas() {
    const cv = $("mask-canvas");
    const need = mode === "edit" && images.length === 1 && $("local-mode").value !== "none";
    if (!need) { cv.hidden = true; return; }
    const im = images[0];
    const maxW = Math.min(cv.parentElement.clientWidth - 40, 560);
    const scale = Math.min(1, maxW / im.w);
    cv.hidden = false;
    cv.width = Math.round(im.w * scale);
    cv.height = Math.round(im.h * scale);
    drawCanvas();
  }

  function drawCanvas() {
    const cv = $("mask-canvas");
    if (cv.hidden || !images.length) return;
    const ctx = cv.getContext("2d");
    const im = images[0];
    ctx.clearRect(0, 0, cv.width, cv.height);
    ctx.drawImage(im.el, 0, 0, cv.width, cv.height);
    ctx.lineWidth = Math.max(2, cv.width * 0.005);
    circles.forEach((c) => {
      ctx.beginPath();
      ctx.arc(c.cx * cv.width, c.cy * cv.height, c.r * cv.width, 0, Math.PI * 2);
      ctx.strokeStyle = c.color;
      ctx.stroke();
    });
  }

  function canvasPos(e) {
    const cv = $("mask-canvas");
    const r = cv.getBoundingClientRect();
    return { x: (e.clientX - r.left) / r.width, y: (e.clientY - r.top) / r.height };
  }

  // ===== 生成 =====
  function b64(dataUrl) { return dataUrl.split(",", 1)[1] || dataUrl; }

  function composeMarked() {
    const im = images[0];
    const c = document.createElement("canvas");
    c.width = im.w; c.height = im.h;
    const ctx = c.getContext("2d");
    ctx.drawImage(im.el, 0, 0, im.w, im.h);
    ctx.lineWidth = Math.max(3, im.w * 0.006);
    circles.forEach((cir) => {
      ctx.beginPath();
      ctx.arc(cir.cx * im.w, cir.cy * im.h, cir.r * im.w, 0, Math.PI * 2);
      ctx.strokeStyle = cir.color;
      ctx.stroke();
    });
    return c.toDataURL("image/png");
  }

  function composeMask() {
    const im = images[0];
    const c = document.createElement("canvas");
    c.width = im.w; c.height = im.h;
    const ctx = c.getContext("2d");
    ctx.fillStyle = "#000000";
    ctx.fillRect(0, 0, c.width, c.height);
    ctx.fillStyle = "#ffffff";
    circles.forEach((cir) => {
      ctx.beginPath();
      ctx.arc(cir.cx * im.w, cir.cy * im.h, cir.r * im.w, 0, Math.PI * 2);
      ctx.fill();
    });
    return c.toDataURL("image/png");
  }

  function buildImagesPayload() {
    const out = [];
    if (mode === "i2i") {
      images.forEach((im) => out.push({ b64_json: b64(im.dataUrl), role: "reference" }));
    } else if (mode === "edit") {
      if (!images.length) throw new Error("图像编辑需要 1 张源图");
      const lm = $("local-mode").value;
      if (lm === "mask" && circles.length) {
        out.push({ b64_json: b64(images[0].dataUrl), role: "source" });
        out.push({ b64_json: b64(composeMask()), role: "mask" });
      } else if (lm === "paint" && circles.length) {
        out.push({ b64_json: b64(composeMarked()), role: "source" });
      } else {
        out.push({ b64_json: b64(images[0].dataUrl), role: "source" });
      }
    }
    return out;
  }

  async function generate() {
    const prompt = $("prompt").value.trim();
    if (!prompt) return setErr("请输入提示词", $("prompt"));
    if (mode === "i2i" && (images.length < 2 || images.length > MAX_REF)) {
      const msg = `图生图需要 2–${MAX_REF} 张参考图（当前 ${images.length} 张）`;
      setImageHint(msg, true);
      return setErr(msg, $("image-block"));
    }
    if (mode === "edit" && images.length < 1) {
      const msg = "图像编辑需要 1 张源图（当前 0 张）";
      setImageHint(msg, true);
      return setErr(msg, $("image-block"));
    }

    let payloadImages = [];
    try { payloadImages = buildImagesPayload(); }
    catch (e) { return setErr(e.message, $("image-block")); }

    const pair = applyPair();                 // 成对解析（任务 51：显示档 = 发送档）
    const inputs = {
      prompt,
      mode,
      negative_prompt: $("negative").value.trim() || undefined,
      num_inference_steps: Number($("steps").value) || undefined,
      true_cfg_scale: Number($("cfg").value),
      scheme: pair.scheme || undefined,
    };
    if (expandOn()) {
      // 扩图（任务 56）：输出尺寸 = 扩展画布（引擎侧按 32 对齐计算并覆盖），故不传 width/height
      const amt = Number($("expand-amt").value);
      if (!Number.isFinite(amt) || amt < SIZE_ALIGN || amt > 1024 || amt % SIZE_ALIGN !== 0) {
        const msg = `扩幅须为 ${SIZE_ALIGN}–1024 且 ${SIZE_ALIGN} 对齐（当前 ${amt}）`;
        return setErr(msg, $("expand-block"));
      }
      const pv = expandPreview();
      if (pv) {
        const [pw, ph] = pv.split("×").map(Number);
        if (pw > SIZE_MAX || ph > SIZE_MAX) {
          const msg = `扩图后画布 ${pw}×${ph} 超出上限 ${SIZE_MAX}×${SIZE_MAX} → 请减小扩幅`;
          return setErr(msg, $("expand-block"));
        }
      }
      inputs.expand = { direction: $("expand-dir").value, amount: amt };
    } else if (mode === "t2i" || !$("follow-size").checked) {
      let w = Number($("width").value);
      let h = Number($("height").value);
      if (!Number.isFinite(w) || !Number.isFinite(h)) {
        return setErr("宽高必须是数字", $("size-row"));
      }
      if (w < SIZE_MIN || w > SIZE_MAX || h < SIZE_MIN || h > SIZE_MAX) {
        const msg = `宽高须在 ${SIZE_MIN}–${SIZE_MAX} 之间（当前 ${w}×${h}）`;
        return setErr(msg, $("size-row"));
      }
      if (w % SIZE_ALIGN || h % SIZE_ALIGN) {
        // 吸附到 32 对齐（管线硬约束），并**显式告知**用户，不静默改值
        const sw = snapAlign(w), sh = snapAlign(h);
        $("width").value = sw; $("height").value = sh;
        showToast(`宽高须 ${SIZE_ALIGN} 对齐：${w}×${h} → 已吸附为 ${sw}×${sh}`);
        w = sw; h = sh;
      }
      inputs.width = w;
      inputs.height = h;
    }
    if ($("transparent").checked) inputs.transparent = true;
    const seed = $("seed").value.trim();
    if (seed !== "") inputs.seed = Number(seed);

    const body = { variant: pair.variant || undefined, inputs };
    if (payloadImages.length) body.images = payloadImages;

    lastBeforeUrl = (mode !== "t2i" && images.length) ? images[0].dataUrl : "";

    $("gen-btn").disabled = true;
    $("progress").style.width = "0%";
    setStage("提交任务…");
    setHint("");
    let errMsg = "";
    try {
      const r = await fetch("/infer", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      if (!r.ok) {
        let detail = "";
        try { detail = (await r.json()).detail || ""; } catch (_) { /* 非 JSON 响应 */ }
        errMsg = `HTTP ${r.status}${detail ? "：" + detail : ""}`;
      } else {
        const reader = r.body.getReader();
        const dec = new TextDecoder();
        let buf = "";
        for (;;) {
          const { done, value } = await reader.read();
          if (done) break;
          buf += dec.decode(value, { stream: true });
          const parts = buf.split("\n\n");
          buf = parts.pop();
          parts.forEach((chunk) => {
            const line = chunk.trim();
            if (!line.startsWith("data:")) return;
            let f; try { f = JSON.parse(line.slice(5).trim()); } catch (_) { return; }
            if (f.progress != null) $("progress").style.width = Math.round(f.progress * 100) + "%";
            if (f.stage && !f.error) setStage(`${f.stage} ${f.detail || ""}`);
            if (f.result || f.type === "final" || f.stage === "completed") {
              const url = (f.result && (f.result.image_url || (f.result.image_urls || [])[0])) || f.image_url;
              if (url) showResult(url);
            }
            if (f.error) errMsg = f.error.message || String(f.error);
          });
        }
      }
    } catch (e) {
      errMsg = e.message;
    } finally {
      $("gen-btn").disabled = false;
      if (errMsg) {
        // 失败信息**持久**展示（修复"闪一下就没反应"：原先流结束后无条件
        // setStage("完成") 会立刻覆盖错误提示，用户看不到失败原因）
        setStage("失败");
        setErr("❌ 生成失败：" + errMsg);
        loadMeta();          // 刷新顶部状态点，反映引擎真实状态
      } else {
        setStage("完成");
        loadHistory();
      }
    }
  }

  function showResult(url) {
    lastResultUrl = url;
    $("preview").innerHTML = `<img src="${url}" alt="result" title="点击查看大图" />`;
    $("download-btn").hidden = false;
    $("reuse-btn").hidden = false;
    if (lastBeforeUrl) {
      $("compare").hidden = false;
      $("compare-btn").hidden = false;
      $("compare-btn").textContent = "收起对比";
      $("cmp-before").src = lastBeforeUrl;
      $("cmp-after").src = url;
      updateCompare(Number($("cmp-range").value));
    } else {
      $("compare").hidden = true;
      $("compare-btn").hidden = true;
    }
  }

  // ===== 详情图查看器（点击结果图 / 历史图查看大图） =====
  function openLightbox(url) {
    if (!url) return;
    const dl = $("lightbox-download");
    $("lightbox-img").src = url;
    dl.href = url;
    dl.download = url.split("/").pop() || "result.png";
    $("lightbox").hidden = false;
  }

  function closeLightbox() {
    $("lightbox").hidden = true;
    $("lightbox-img").removeAttribute("src");
  }

  function updateCompare(v) {
    const pct = Math.max(0, Math.min(100, v));
    $("cmp-after").style.clipPath = `inset(0 0 0 ${pct}%)`;
    $("cmp-line").style.left = pct + "%";
  }

  async function reuseResult() {
    if (!lastResultUrl) return;
    try {
      const blob = await (await fetch(lastResultUrl)).blob();
      const dataUrl = await new Promise((res, rej) => {
        const fr = new FileReader();
        fr.onload = () => res(fr.result);
        fr.onerror = rej;
        fr.readAsDataURL(blob);
      });
      const el = new Image();
      el.onload = () => {
        images = [{ name: "result.png", dataUrl, w: el.naturalWidth, h: el.naturalHeight, el }];
        resetCircles();
        // 任务 56：统一走 setMode("edit")（原实现手工改 DOM，会漏掉
        // 尺寸/扩图控件联动 → "以此图继续编辑"后状态不一致）
        setMode("edit");
        setHint("已把结果设为源图，可直接继续编辑");
      };
      el.src = dataUrl;
    } catch (e) { setErr("回填失败：" + e.message); }
  }

  async function loadHistory() {
    try {
      const h = await jget("/history?page=1&page_size=20");
      $("history").innerHTML = (h.items || []).map((it) =>
        `<div class="hist-item"><img src="${it.image_url}" title="${(it.prompt || "").slice(0, 60)}" data-url="${it.image_url}" /><span class="badge">${it.mode || "t2i"}</span></div>`
      ).join("");
    } catch (_) { /* 降级忽略 */ }
  }

  // ===== 事件绑定 =====
  $("mode-tabs").addEventListener("click", (e) => {
    const b = e.target.closest(".mode-tab");
    if (b) setMode(b.dataset.mode);
  });
  $("local-mode").addEventListener("change", () => { renderCanvas(); });
  $("follow-size").addEventListener("change", applyFollowSize);
  $("load-btn").addEventListener("click", doLoad);
  $("gen-btn").addEventListener("click", generate);
  $("reuse-btn").addEventListener("click", reuseResult);
  $("download-btn").addEventListener("click", () => {
    if (lastResultUrl) {
      const a = document.createElement("a");
      a.href = lastResultUrl; a.download = lastResultUrl.split("/").pop() || "result.png";
      a.click();
    }
  });
  $("ratio-row").addEventListener("click", (e) => {
    const b = e.target.closest("button.ratio");
    if (!b) return;
    // 任务 56：点预设比例必须**取消「跟随」**（否则值被忽略 → 用户以为"点了没反应"）；
    // 同时关闭扩图（预设比例与扩图二选一：扩图的输出尺寸由画布决定）
    if (expandOn()) {
      $("expand-on").checked = false;
      applyExpand();
      showToast("已关闭扩图：预设比例与扩图二选一");
    }
    $("follow-size").checked = false;
    applyFollowSize();
    $("width").value = b.dataset.w;
    $("height").value = b.dataset.h;
  });
  $("expand-on").addEventListener("change", applyExpand);
  $("expand-dir").addEventListener("change", applyExpand);
  $("expand-amt").addEventListener("input", applyExpand);
  $("thumbs").addEventListener("click", (e) => {
    const b = e.target.closest(".del");
    if (!b) return;
    images.splice(Number(b.dataset.i), 1);
    renderThumbs(); renderCanvas(); applyExpand();
  });
  $("undo-btn").addEventListener("click", () => { circles.pop(); drawCanvas(); });
  $("clear-btn").addEventListener("click", () => { resetCircles(); drawCanvas(); });
  $("cmp-range").addEventListener("input", (e) => updateCompare(Number(e.target.value)));
  $("compare-btn").addEventListener("click", () => {
    if ($("compare").hidden) {
      $("compare").hidden = false;
      updateCompare(Number($("cmp-range").value));
      $("compare-btn").textContent = "收起对比";
    } else {
      $("compare").hidden = true;
      $("compare-btn").textContent = "对比原图";
    }
  });
  $("preview").addEventListener("click", (e) => {
    const img = e.target.closest("img");
    if (img) openLightbox(img.getAttribute("src"));
  });
  $("lightbox-close").addEventListener("click", closeLightbox);
  $("lightbox").addEventListener("click", (e) => {
    if (e.target.id === "lightbox") closeLightbox();
  });
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape" && !$("lightbox").hidden) closeLightbox();
  });
  $("history").addEventListener("click", (e) => {
    const img = e.target.closest("img");
    if (img) {
      const url = img.getAttribute("data-url");
      showResult(url);
      openLightbox(url);
    }
  });

  // 上传：点击 / 拖拽 / 粘贴
  $("dropzone").addEventListener("click", () => $("file-input").click());
  $("file-input").addEventListener("change", (e) => { addFiles(e.target.files); e.target.value = ""; });
  ["dragenter", "dragover"].forEach((ev) => $("dropzone").addEventListener(ev, (e) => {
    e.preventDefault(); $("dropzone").classList.add("over");
  }));
  ["dragleave", "drop"].forEach((ev) => $("dropzone").addEventListener(ev, (e) => {
    e.preventDefault(); $("dropzone").classList.remove("over");
    if (ev === "drop") addFiles(e.dataTransfer.files);
  }));
  document.addEventListener("paste", (e) => {
    if (mode === "t2i" || !e.clipboardData) return;
    const files = Array.from(e.clipboardData.files || []);
    if (files.length) addFiles(files);
  });

  // 画布画圆
  $("mask-canvas").addEventListener("pointerdown", (e) => {
    if ($("mask-canvas").hidden) return;
    const p = canvasPos(e);
    drag = { cx: p.x, cy: p.y, color: COLORS[colorIdx] };
    $("mask-canvas").setPointerCapture(e.pointerId);
  });
  $("mask-canvas").addEventListener("pointermove", (e) => {
    if (!drag) return;
    const p = canvasPos(e);
    const r = Math.hypot(p.x - drag.cx, p.y - drag.cy);
    const cv = $("mask-canvas");
    const ctx = cv.getContext("2d");
    drawCanvas();
    ctx.beginPath();
    ctx.arc(drag.cx * cv.width, drag.cy * cv.height, r * cv.width, 0, Math.PI * 2);
    ctx.strokeStyle = drag.color;
    ctx.stroke();
  });
  $("mask-canvas").addEventListener("pointerup", (e) => {
    if (!drag) return;
    const p = canvasPos(e);
    const r = Math.hypot(p.x - drag.cx, p.y - drag.cy);
    if (r > 0.02) circles.push({ cx: drag.cx, cy: drag.cy, r, color: drag.color });
    drag = null;
    drawCanvas();
  });

  // 颜色色板
  $("swatches").innerHTML = COLORS.map((c, i) =>
    `<span class="swatch${i === 0 ? " active" : ""}" data-i="${i}" style="background:${c}"></span>`).join("");
  $("swatches").addEventListener("click", (e) => {
    const s = e.target.closest(".swatch");
    if (!s) return;
    colorIdx = Number(s.dataset.i);
    document.querySelectorAll(".swatch").forEach((x) => x.classList.toggle("active", x === s));
  });

  setMode("t2i");
  loadMeta();
  loadHistory();
})();

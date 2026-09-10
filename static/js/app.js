/* AI Archaeologist — frontend logic (no dependencies). */
(function () {
  "use strict";

  var state = { file: null, sample: null, library: [], busy: false };

  var SAMPLES = [
    { file: "chola_nataraja_met.jpg", title: "Chola bronze" },
    { file: "indus_seal_unicorn.jpg", title: "Indus seal" },
    { file: "greek_amphora_blackfig.jpg", title: "Greek vase" },
    { file: "egypt_chephren.jpg", title: "Egyptian statue" },
    { file: "meso_tablet_cuneiform.jpg", title: "Cuneiform tablet" },
    { file: "roman_amphora_pompeii.jpg", title: "Roman amphora" }
  ];

  var LOADING_STEPS = [
    "Reading pixels…",
    "Mapping colour signature…",
    "Sensing material & texture…",
    "Comparing with reference artefacts…",
    "Fusing context & estimating date…"
  ];

  function $(id) { return document.getElementById(id); }
  function esc(s) {
    return String(s == null ? "" : s).replace(/[&<>"']/g, function (c) {
      return { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c];
    });
  }
  function confLabel(c) {
    if (c >= 0.8) return "Very high";
    if (c >= 0.65) return "High";
    if (c >= 0.5) return "Medium";
    return "Low";
  }
  function meter(conf) {
    var pct = Math.round(conf * 100);
    return '<div class="meter"><div style="width:' + pct + '%"></div></div>' +
      '<div class="meter-label"><span>' + pct + '%</span><span>' + confLabel(conf) + ' confidence</span></div>';
  }

  /* ---------------- library ---------------- */
  function loadLibrary() {
    fetch("/api/artifacts").then(function (r) { return r.json(); }).then(function (data) {
      state.library = data.artifacts || [];
      $("statRefs").textContent = data.count || state.library.length;
      renderLibrary();
      renderSamples();
    }).catch(function () {
      $("libGrid").innerHTML = '<p class="muted">Could not load the reference library.</p>';
    });
  }

  function renderLibrary() {
    var grid = $("libGrid");
    grid.innerHTML = state.library.map(function (a) {
      return '<div class="lib-card" data-sample="' + esc(a.image) + '" title="Click to analyze this artefact">' +
        '<img src="/static/ref/' + esc(a.image) + '" alt="' + esc(a.name) + '" loading="lazy">' +
        '<div class="lib-body">' +
        '<p class="lib-name">' + esc(a.name) + '</p>' +
        '<p class="lib-period">' + esc(a.period_label) + '</p>' +
        '<p class="lib-region">' + esc(a.region) + '</p>' +
        '<p class="lib-desc">' + esc(a.description) + '</p>' +
        '<div class="lib-tags"><span class="pill">' + esc(a.category) + '</span>' +
        '<span class="pill">' + esc(a.material) + '</span>' +
        (a.replica ? '<span class="pill">replica photo</span>' : '') + '</div>' +
        '</div></div>';
    }).join("");
    grid.querySelectorAll(".lib-card").forEach(function (card) {
      card.addEventListener("click", function () {
        setSample(card.getAttribute("data-sample"));
        document.getElementById("analyzer").scrollIntoView({ behavior: "smooth" });
        runAnalysis();
      });
    });
  }

  function renderSamples() {
    var row = $("sampleRow");
    row.innerHTML = "";
    SAMPLES.forEach(function (s) {
      var img = document.createElement("img");
      img.src = "/static/ref/" + s.file;
      img.alt = s.title;
      img.title = "Try: " + s.title;
      img.className = "sample-thumb";
      img.dataset.sample = s.file;
      img.addEventListener("click", function () { setSample(s.file); });
      row.appendChild(img);
    });
  }

  /* ---------------- input handling ---------------- */
  var dropzone = $("dropzone"), fileInput = $("fileInput"),
      dzPreview = $("dzPreview"), formError = $("formError");

  function showError(msg) {
    formError.textContent = msg;
    formError.hidden = !msg;
  }

  function setPreview(src) {
    dzPreview.src = src;
    dropzone.classList.add("has-image");
  }

  function clearInput() {
    state.file = null;
    state.sample = null;
    dzPreview.removeAttribute("src");
    dropzone.classList.remove("has-image");
    fileInput.value = "";
    document.querySelectorAll(".sample-thumb").forEach(function (t) { t.classList.remove("active"); });
  }

  function setFile(file) {
    if (!file || !file.type || file.type.indexOf("image/") !== 0) {
      showError("Please choose an image file (JPG/PNG).");
      return;
    }
    clearInput();
    state.file = file;
    showError("");
    var reader = new FileReader();
    reader.onload = function (e) { setPreview(e.target.result); };
    reader.readAsDataURL(file);
  }

  function setSample(name) {
    clearInput();
    state.sample = name;
    showError("");
    setPreview("/static/ref/" + name);
    document.querySelectorAll(".sample-thumb").forEach(function (t) {
      t.classList.toggle("active", t.dataset.sample === name);
    });
  }

  dropzone.addEventListener("click", function (e) {
    if (e.target.id === "dzRemove") return;
    fileInput.click();
  });
  dropzone.addEventListener("keydown", function (e) {
    if (e.key === "Enter" || e.key === " ") { e.preventDefault(); fileInput.click(); }
  });
  fileInput.addEventListener("change", function () {
    if (fileInput.files && fileInput.files[0]) setFile(fileInput.files[0]);
  });
  ["dragenter", "dragover"].forEach(function (ev) {
    dropzone.addEventListener(ev, function (e) { e.preventDefault(); dropzone.classList.add("dragover"); });
  });
  ["dragleave", "drop"].forEach(function (ev) {
    dropzone.addEventListener(ev, function (e) { e.preventDefault(); dropzone.classList.remove("dragover"); });
  });
  dropzone.addEventListener("drop", function (e) {
    var f = e.dataTransfer && e.dataTransfer.files && e.dataTransfer.files[0];
    if (f) setFile(f);
  });
  document.addEventListener("paste", function (e) {
    var items = (e.clipboardData || {}).items || [];
    for (var i = 0; i < items.length; i++) {
      if (items[i].type && items[i].type.indexOf("image/") === 0) {
        setFile(items[i].getAsFile());
        break;
      }
    }
  });
  $("dzRemove").addEventListener("click", function (e) { e.stopPropagation(); clearInput(); });

  /* ---------------- analysis ---------------- */
  var analyzeBtn = $("analyzeBtn"), loading = $("loading"),
      loadingStep = $("loadingStep"), loadingFill = $("loadingFill"),
      resultsEmpty = $("resultsEmpty"), resultsBox = $("results");
  var stepTimer = null;

  function setBusy(b) {
    state.busy = b;
    analyzeBtn.disabled = b;
    analyzeBtn.innerHTML = b ? "⏳ &nbsp;Analyzing…" : "🏺 &nbsp;Analyze artefact";
  }

  function startLoading() {
    resultsEmpty.hidden = true;
    resultsBox.hidden = true;
    loading.hidden = false;
    var i = 0;
    loadingStep.textContent = LOADING_STEPS[0];
    loadingFill.style.width = "8%";
    stepTimer = setInterval(function () {
      i = (i + 1) % LOADING_STEPS.length;
      loadingStep.textContent = LOADING_STEPS[i];
      loadingFill.style.width = Math.min(8 + (i + 1) * 20, 96) + "%";
    }, 650);
  }

  function stopLoading() {
    if (stepTimer) clearInterval(stepTimer);
    stepTimer = null;
    loading.hidden = true;
    loadingFill.style.width = "100%";
  }

  function context() {
    return {
      found_region: $("foundRegion").value,
      material_hint: $("materialHint").value,
      notes: $("notes").value
    };
  }

  function runAnalysis() {
    if (state.busy) return;
    if (!state.file && !state.sample) {
      showError("Please upload a photo or pick a sample first.");
      return;
    }
    showError("");
    setBusy(true);
    startLoading();

    var req;
    if (state.sample) {
      req = fetch("/api/analyze-sample/" + encodeURIComponent(state.sample), {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(context())
      });
    } else {
      var fd = new FormData();
      fd.append("image", state.file);
      var c = context();
      fd.append("found_region", c.found_region);
      fd.append("material_hint", c.material_hint);
      fd.append("notes", c.notes);
      req = fetch("/api/analyze", { method: "POST", body: fd });
    }

    req.then(function (r) { return r.json().then(function (j) { return { ok: r.ok, body: j }; }); })
      .then(function (res) {
        stopLoading();
        setBusy(false);
        if (!res.ok) {
          resultsEmpty.hidden = false;
          showError((res.body && res.body.error) || "Analysis failed.");
          return;
        }
        renderResults(res.body);
      })
      .catch(function () {
        stopLoading();
        setBusy(false);
        resultsEmpty.hidden = false;
        showError("Server unreachable. Is the app running?");
      });
  }

  analyzeBtn.addEventListener("click", runAnalysis);

  /* ---------------- results rendering ---------------- */
  function timelineHTML(t) {
    function pct(x) { return ((x - t.min) / (t.max - t.min) * 100).toFixed(2); }
    return '<div class="timeline"><div class="tl-track">' +
      '<div class="tl-range" style="left:' + pct(t.low) + '%;width:' + Math.max(pct(t.high) - pct(t.low), 2) + '%"></div>' +
      '<div class="tl-marker" style="left:calc(' + pct(t.predicted) + '% - 2px)" title="Predicted date"></div>' +
      '</div><div class="tl-labels"><span>3000 BCE</span><span>0</span><span>2000 CE</span></div></div>';
  }

  function renderResults(r) {
    var html = "";

    if (r.tentative) {
      html += '<div class="tentative-banner">⚠️ <strong>Tentative hypothesis</strong> — no close match in the ' +
        'reference library. Add find-spot / material context for a sharper read.</div>';
    }

    html += '<div class="pred-grid">' +
      '<div class="pred span"><div class="pred-kicker">⏳ Approximate period</div>' +
      '<div class="pred-value">' + esc(r.period.label) + '</div>' +
      '<p class="pred-sub">' + esc(r.period.date_range) + ' · midpoint ' + esc(r.period.year_mid_label) + '</p>' +
      timelineHTML(r.timeline) + meter(r.period.confidence) + '</div>' +

      '<div class="pred"><div class="pred-kicker">🏷️ Category</div>' +
      '<div class="pred-value" style="font-size:1.1rem">' + esc(r.category.label) + '</div>' +
      meter(r.category.confidence) +
      r.category.all.slice(1, 3).map(function (a) {
        return '<span class="pill">' + esc(a.label) + ' ' + Math.round(a.score * 100) + '%</span>';
      }).join("") + '</div>' +

      '<div class="pred"><div class="pred-kicker">🗺️ Possible region</div>' +
      '<div class="pred-value" style="font-size:1.1rem">' + esc(r.region.label) + '</div>' +
      '<p class="pred-sub">' + esc(r.region.group) + ' · ' + esc(r.region.culture) + '</p>' +
      meter(r.region.confidence) + '</div>' +

      '<div class="pred"><div class="pred-kicker">🧱 Likely material</div>' +
      '<div class="pred-value" style="font-size:1.1rem">' + esc(r.material.label) + '</div>' +
      meter(r.material.confidence) + '</div>' +

      '<div class="pred"><div class="pred-kicker">⚡ Verdict</div>' +
      '<div class="pred-value" style="font-size:1.1rem">' +
      (r.tentative ? "Needs more evidence" : "Confident hypothesis") + '</div>' +
      '<p class="pred-sub">' + esc(r.meta.library_size) + ' references · ' +
      esc(r.meta.latency_ms) + ' ms analysis</p></div>' +
      '</div>';

    html += '<div class="sub-card"><h4>🏛️ Similarity to known artefacts</h4><div class="sim-grid">' +
      r.similar.map(function (s, i) {
        return '<div class="sim-card' + (i === 0 ? ' sim-top' : '') + '">' +
          '<img src="/static/ref/' + esc(s.image) + '" alt="' + esc(s.name) + '" loading="lazy">' +
          '<div class="sim-body"><div class="sim-score">▲ ' + esc(s.similarity) + '%</div>' +
          '<div class="sim-name">' + esc(s.name) + '</div>' +
          '<div class="sim-meta">' + esc(s.period) + '<br>' + esc(s.region) +
          (s.replica ? '<br><span class="replica-tag">replica photo</span>' : '') + '</div></div></div>';
      }).join("") + '</div></div>';

    var cueNames = {
      terracotta: "Terracotta", pale_ceramic: "Pale ceramic", dark_stone: "Dark stone",
      bronze: "Bronze", gold: "Gold", black_figure_paint: "Black-figure paint"
    };
    html += '<div class="two-col"><div class="sub-card" style="margin-top:0"><h4>🎨 Visual fingerprint</h4>' +
      '<div class="swatches">' + r.visual.dominant_colors.map(function (c) {
        return '<span class="swatch"><i style="background:' + esc(c.hex) + '"></i>' +
          esc(c.hex) + ' · ' + Math.round(c.pct * 100) + '%</span>';
      }).join("") + '</div>' +
      '<div style="margin-top:10px">' +
      '<span class="pill">detail ' + esc(r.visual.texture.edge_density) + '</span>' +
      '<span class="pill">symmetry ' + esc(r.visual.texture.symmetry) + '</span>' +
      '<span class="pill">brightness ' + esc(r.visual.texture.brightness) + '</span></div>' +
      '</div><div class="sub-card" style="margin-top:0"><h4>🧪 Material cues</h4>' +
      Object.keys(cueNames).map(function (k) {
        var v = Math.round((r.visual.material_cues[k] || 0) * 100);
        return '<div class="cue-row"><span>' + cueNames[k] + '</span>' +
          '<div class="cue-bar"><div style="width:' + v + '%"></div></div><span>' + v + '%</span></div>';
      }).join("") + '</div></div>';

    html += '<div class="sub-card"><h4>💡 Why this prediction?</h4><ol class="reason-list">' +
      r.reasoning.map(function (line) { return "<li>" + esc(line) + "</li>"; }).join("") +
      '</ol></div>';

    html += '<p class="result-meta">' + esc(r.meta.model) + ' · ' + esc(r.meta.note) + '</p>';

    resultsBox.innerHTML = html;
    resultsBox.hidden = false;
    if (window.innerWidth < 960) {
      $("resultsCard").scrollIntoView({ behavior: "smooth", block: "start" });
    }
  }

  /* ---------------- boot ---------------- */
  loadLibrary();
})();

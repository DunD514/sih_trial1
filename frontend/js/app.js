/**
 * SatQuery AI Frontend Application Logic
 * Interactive Multimodal Remote-Sensing Assistant Client
 */

document.addEventListener("DOMContentLoaded", () => {
  // DOM Elements
  const sampleSelect = document.getElementById("sample-select");
  const dropZone1 = document.getElementById("drop-zone-1");
  const dropZone2 = document.getElementById("drop-zone-2");
  const fileInput1 = document.getElementById("file-input-1");
  const fileInput2 = document.getElementById("file-input-2");
  const previewArea1 = document.getElementById("preview-area-1");
  const previewArea2 = document.getElementById("preview-area-2");
  const imgPreview1 = document.getElementById("img-preview-1");
  const imgPreview2 = document.getElementById("img-preview-2");

  const taskSelect = document.getElementById("task-select");
  const userPrompt = document.getElementById("user-prompt");
  const btnAnalyze = document.getElementById("btn-analyze");
  const analyzeSpinner = document.getElementById("analyze-spinner");

  const tabBtns = document.querySelectorAll(".tab-btn");
  const tabContents = document.querySelectorAll(".tab-content");
  const opacitySlider = document.getElementById("overlay-opacity");

  const viewBaseImg = document.getElementById("view-base-img");
  const viewOverlayImg = document.getElementById("view-overlay-img");
  const viewSplitT1 = document.getElementById("view-split-t1");
  const viewSplitT2 = document.getElementById("view-split-t2");
  const viewMaskImg = document.getElementById("view-mask-img");
  const viewDiffImg = document.getElementById("view-diff-img");
  const bboxLayer = document.getElementById("bbox-overlay-layer");

  const statChangePct = document.getElementById("stat-change-pct");
  const statClusters = document.getElementById("stat-clusters");
  const statConfidence = document.getElementById("stat-confidence");
  const statModel = document.getElementById("stat-model");
  const geoBadge = document.getElementById("geo-badge");

  const aiAnswer = document.getElementById("ai-answer");
  const semanticTags = document.getElementById("semantic-tags");
  const traceList = document.getElementById("trace-list");
  const btnDownloadMd = document.getElementById("btn-download-md");
  const btnDownloadGeoJson = document.getElementById("btn-download-geojson");

  // State
  let image1Data = null;
  let image2Data = null;
  let image1Path = null;
  let image2Path = null;
  let currentResult = null;

  // Initialize: Load sample list from backend
  fetchSamples();

  // Tab switching
  window.switchView = function(viewName) {
    if (!viewName) return;
    let tabId = viewName;
    if (viewName === "difference-heatmap" || viewName === "heatmap" || viewName === "diff_map") tabId = "diff-map";
    if (viewName === "change-mask" || viewName === "change_mask" || viewName === "mask") tabId = "change-mask";
    if (viewName === "side-by-side" || viewName === "side_by_side") tabId = "side-by-side";
    if (viewName === "interactive-overlay" || viewName === "overlay" || viewName === "visual-overlay") tabId = "interactive-overlay";

    tabBtns.forEach(b => b.classList.remove("active"));
    tabContents.forEach(c => c.classList.remove("active"));
    const btn = document.querySelector(`[data-tab="${tabId}"]`);
    if (btn) btn.classList.add("active");
    const targetTab = document.getElementById(`tab-${tabId}`);
    if (targetTab) targetTab.classList.add("active");

    const legend = document.getElementById("canvas-legend");
    if (legend) {
      legend.style.display = (tabId === "diff-map" || tabId === "change-mask") ? "flex" : "none";
    }
  };

  window.handleJarvisUIAction = function(action) {
    if (!action) return;
    if (action.type === "SELECT_VIEW" && action.view) {
      window.switchView(action.view);
    }
  };

  tabBtns.forEach(btn => {
    btn.addEventListener("click", () => {
      window.switchView(btn.dataset.tab);
    });
  });

  // Right Context Panel Tabs Switching
  const panelTabs = document.querySelectorAll(".panel-tab-btn");
  const panelSections = document.querySelectorAll(".panel-section");
  panelTabs.forEach(btn => {
    btn.addEventListener("click", () => {
      panelTabs.forEach(b => b.classList.remove("active"));
      panelSections.forEach(s => s.style.display = "none");
      btn.classList.add("active");
      const target = document.getElementById(btn.dataset.target);
      if (target) target.style.display = "flex";
    });
  });

  // Top Minimal Navigation Links
  document.getElementById("nav-btn-explore")?.addEventListener("click", (e) => {
    document.querySelectorAll(".nav-item").forEach(n => n.classList.remove("active"));
    e.target.classList.add("active");
    window.switchView("interactive-overlay");
    userPrompt.value = "Describe this satellite scene.";
    userPrompt.focus();
  });

  document.getElementById("nav-btn-change")?.addEventListener("click", (e) => {
    document.querySelectorAll(".nav-item").forEach(n => n.classList.remove("active"));
    e.target.classList.add("active");
    window.switchView("change-mask");
    userPrompt.value = "What changed between these two images?";
    userPrompt.focus();
  });

  document.getElementById("nav-btn-grounding")?.addEventListener("click", (e) => {
    document.querySelectorAll(".nav-item").forEach(n => n.classList.remove("active"));
    e.target.classList.add("active");
    window.switchView("interactive-overlay");
    if (taskSelect) taskSelect.value = "grounding";
    userPrompt.value = "Where is the built-up region? Locate structures.";
    userPrompt.focus();
  });

  document.getElementById("nav-btn-reports")?.addEventListener("click", (e) => {
    document.querySelectorAll(".nav-item").forEach(n => n.classList.remove("active"));
    e.target.classList.add("active");
    document.getElementById("tab-btn-intel")?.click();
    btnDownloadMd?.scrollIntoView({ behavior: "smooth" });
  });

  // Unified Query Execution Handler
  function triggerQuery() {
    if (window._jarvis && window._jarvis.isOn) {
      const q = userPrompt.value.trim();
      if (!q) return;
      window._jarvis._interruptSpeech();
      window._jarvis._addMsg("user", q);
      window._jarvis.handleQuery(q);
      userPrompt.value = "";
    } else {
      executeAnalysis();
    }
  }

  userPrompt.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      triggerQuery();
    }
  });

  // Opacity Slider
  opacitySlider.addEventListener("input", (e) => {
    viewOverlayImg.style.opacity = (e.target.value / 100).toString();
  });

  // Quick Chips
  document.querySelectorAll(".chip-btn").forEach(chip => {
    chip.addEventListener("click", () => {
      userPrompt.value = chip.dataset.query;
      userPrompt.focus();
    });
  });

  // Upload Box 1 Click & Drag
  dropZone1.addEventListener("click", () => fileInput1.click());
  setupDragAndDrop(dropZone1, (file) => handleFileUpload(file, 1));
  fileInput1.addEventListener("change", (e) => {
    if (e.target.files.length) handleFileUpload(e.target.files[0], 1);
  });

  // Upload Box 2 Click & Drag
  dropZone2.addEventListener("click", () => fileInput2.click());
  setupDragAndDrop(dropZone2, (file) => handleFileUpload(file, 2));
  fileInput2.addEventListener("change", (e) => {
    if (e.target.files.length) handleFileUpload(e.target.files[0], 2);
  });

  // Sample Selection
  sampleSelect.addEventListener("change", (e) => {
    const selected = e.target.value;
    if (!selected) return;
    let sampleData = null;
    try {
      sampleData = JSON.parse(selected);
    } catch (err) {
      console.warn("Could not parse sample JSON:", selected, err);
      return;
    }
    const t1Path = sampleData.t1_path || sampleData.t1 || "";
    const t2Path = sampleData.t2_path || sampleData.t2 || "";
    const t1Url = sampleData.t1_url || (t1Path ? `/api/v1/sample_image?path=${encodeURIComponent(t1Path)}` : "");
    const t2Url = sampleData.t2_url || (t2Path ? `/api/v1/sample_image?path=${encodeURIComponent(t2Path)}` : "");

    if (!t1Url || !t2Url) return;

    image1Path = t1Path;
    image2Path = t2Path;
    image1Data = null;
    image2Data = null;

    // Load preview images
    imgPreview1.src = t1Url;
    imgPreview1.style.display = "block";
    const ph1 = previewArea1 ? previewArea1.querySelector(".upload-placeholder") : null;
    if (ph1) ph1.style.display = "none";

    imgPreview2.src = t2Url;
    imgPreview2.style.display = "block";
    const ph2 = previewArea2 ? previewArea2.querySelector(".upload-placeholder") : null;
    if (ph2) ph2.style.display = "none";

    // Set viewport images
    if (viewBaseImg) {
      viewBaseImg.src = t2Url;
      viewBaseImg.style.display = "block";
    }
    if (viewSplitT1) {
      viewSplitT1.src = t1Url;
      viewSplitT1.style.display = "block";
    }
    if (viewSplitT2) {
      viewSplitT2.src = t2Url;
      viewSplitT2.style.display = "block";
    }

    // Reset results view
    if (viewOverlayImg) viewOverlayImg.style.display = "none";
    if (viewMaskImg) viewMaskImg.src = "";
    if (viewDiffImg) viewDiffImg.src = "";
    if (bboxLayer) bboxLayer.innerHTML = "";

    // Notify JARVIS controller of image change
    window.dispatchEvent(new CustomEvent("satquery:image-changed", {
      detail: { image1Path, image2Path, image1Data: null, image2Data: null, t1Url, t2Url }
    }));
  });

  // Analyze Button
  btnAnalyze.addEventListener("click", triggerQuery);

  // Download Markdown Report
  btnDownloadMd.addEventListener("click", () => {
    if (!currentResult || !currentResult.markdown_report) return;
    const blob = new Blob([currentResult.markdown_report], { type: "text/markdown" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `SatQuery_Report_${Date.now()}.md`;
    a.click();
    URL.revokeObjectURL(url);
  });

  // Download GeoJSON Features
  if (btnDownloadGeoJson) {
    btnDownloadGeoJson.addEventListener("click", () => {
      if (!currentResult || !currentResult.geojson) return;
      const jsonStr = JSON.stringify(currentResult.geojson, null, 2);
      const blob = new Blob([jsonStr], { type: "application/geo+json" });
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `SatQuery_Changes_${Date.now()}.geojson`;
      a.click();
      URL.revokeObjectURL(url);
    });
  }

  function setupDragAndDrop(zone, onDropCallback) {
    zone.addEventListener("dragover", (e) => {
      e.preventDefault();
      zone.classList.add("dragover");
    });
    zone.addEventListener("dragleave", () => zone.classList.remove("dragover"));
    zone.addEventListener("drop", (e) => {
      e.preventDefault();
      zone.classList.remove("dragover");
      if (e.dataTransfer.files.length) {
        onDropCallback(e.dataTransfer.files[0]);
      }
    });
  }

  function handleFileUpload(file, targetIndex) {
    const reader = new FileReader();
    reader.onload = (e) => {
      const dataUrl = e.target.result;
      if (targetIndex === 1) {
        image1Data = dataUrl;
        image1Path = null;
        imgPreview1.src = dataUrl;
        imgPreview1.style.display = "block";
        previewArea1.querySelector(".upload-placeholder").style.display = "none";
        viewBaseImg.src = dataUrl;
        viewSplitT1.src = dataUrl;
      } else {
        image2Data = dataUrl;
        image2Path = null;
        imgPreview2.src = dataUrl;
        imgPreview2.style.display = "block";
        previewArea2.querySelector(".upload-placeholder").style.display = "none";
        viewSplitT2.src = dataUrl;
      }
      // Notify JARVIS controller of image change
      window.dispatchEvent(new CustomEvent("satquery:image-changed", {
        detail: { image1Path, image2Path, image1Data, image2Data }
      }));
    };
    reader.readAsDataURL(file);
  }

  async function fetchSamples() {
    try {
      const res = await fetch("/api/v1/samples");
      const data = await res.json();
      sampleSelect.innerHTML = '<option value="">Select pre-loaded sample pair...</option>';

      if (data.levir_samples && data.levir_samples.length > 0) {
        const optGroup = document.createElement("optgroup");
        optGroup.label = "LEVIR-CD Benchmark Samples";
        data.levir_samples.forEach(s => {
          const opt = document.createElement("option");
          opt.value = JSON.stringify(s);
          opt.textContent = `${s.name}`;
          optGroup.appendChild(opt);
        });
        sampleSelect.appendChild(optGroup);
      }

      if (data.dsifn_samples && data.dsifn_samples.length > 0) {
        const optGroup = document.createElement("optgroup");
        optGroup.label = "DSIFN Benchmark Samples";
        data.dsifn_samples.forEach(s => {
          const opt = document.createElement("option");
          opt.value = JSON.stringify(s);
          opt.textContent = `${s.name}`;
          optGroup.appendChild(opt);
        });
        sampleSelect.appendChild(optGroup);
      }

      if (data.custom_samples && data.custom_samples.length > 0) {
        const optGroup = document.createElement("optgroup");
        optGroup.label = "Custom Input Pairs";
        data.custom_samples.forEach(s => {
          const opt = document.createElement("option");
          opt.value = JSON.stringify(s);
          opt.textContent = `${s.name}`;
          optGroup.appendChild(opt);
        });
        sampleSelect.appendChild(optGroup);
      }

      // Auto-load first benchmark sample if no imagery loaded yet
      if (!image1Path && !image1Data) {
        setTimeout(() => {
          const firstOpt = sampleSelect.querySelector("optgroup option");
          if (firstOpt) {
            sampleSelect.value = firstOpt.value;
            sampleSelect.dispatchEvent(new Event("change"));
          }
        }, 120);
      }
    } catch (err) {
      console.warn("Could not fetch pre-loaded samples:", err);
    }
  }

  async function executeAnalysis() {
    if (!image1Data && !image1Path) {
      alert("Please upload or select at least one primary image.");
      return;
    }

    btnAnalyze.disabled = true;
    analyzeSpinner.style.display = "block";
    aiAnswer.textContent = "Agent controller analyzing inputs and orchestrating specialist models...";

    const payload = {
      image_1: image1Data,
      image_2: image2Data,
      image_1_path: image1Path,
      image_2_path: image2Path,
      question: userPrompt.value.trim() || undefined,
      task: taskSelect.value === "auto" ? undefined : taskSelect.value
    };

    try {
      const response = await fetch("/api/v1/analyze", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload)
      });

      if (!response.ok) {
        const errJson = await response.json();
        throw new Error(errJson.detail || errJson.error || "Analysis failed");
      }

      const result = await response.json();
      currentResult = result;
      updateUIWithResults(result);

    } catch (err) {
      aiAnswer.innerHTML = `<span style="color: var(--accent-red)">Error: ${err.message}</span>`;
      console.error(err);
    } finally {
      btnAnalyze.disabled = false;
      analyzeSpinner.style.display = "none";
    }
  }

  function updateUIWithResults(result) {
    // 1. AI Answer
    aiAnswer.textContent = result.answer || "Analysis complete.";

    // 2. Metrics Bar
    if (result.change_percentage !== undefined && result.task === "change_analysis") {
      statChangePct.textContent = `${result.change_percentage.toFixed(2)} %`;
      statClusters.textContent = result.bboxes ? result.bboxes.length : "0";
    } else if (result.count !== undefined) {
      statChangePct.textContent = "N/A";
      statClusters.textContent = `Count: ${result.count}`;
    } else if (result.num_regions !== undefined) {
      statChangePct.textContent = "N/A";
      statClusters.textContent = `Regions: ${result.num_regions}`;
    } else {
      statChangePct.textContent = "N/A";
      statClusters.textContent = "N/A";
    }

    // Confidence
    if (result.confidence !== null && result.confidence !== undefined) {
      statConfidence.textContent = `${(result.confidence * 100).toFixed(1)}%`;
      statConfidence.title = result.confidence_source || "Model Probability";
    } else {
      statConfidence.textContent = "N/A";
      statConfidence.title = "Uncertainty unavailable";
    }

    // Specialist
    if (result.task === "change_analysis") {
      statModel.textContent = "ChangeFormerV6";
    } else if (result.models_used && result.models_used.length > 0) {
      statModel.textContent = result.models_used[0].split(" ")[0];
    } else {
      statModel.textContent = result.task.toUpperCase();
    }

    // Geospatial Badge
    if (result.geospatial_metadata && result.geospatial_metadata.t1 && result.geospatial_metadata.t1.crs) {
      geoBadge.textContent = `CRS: ${result.geospatial_metadata.t1.crs.split(" ")[0]}`;
      geoBadge.style.display = "inline-block";
    } else {
      geoBadge.textContent = "Geo: Standard";
    }

    // 3. Overlays & Visualizations
    if (result.grounding_vis_url) {
      viewOverlayImg.src = result.grounding_vis_url;
      viewOverlayImg.style.display = "block";
      viewBaseImg.src = result.grounding_vis_url;
    } else if (result.fusion_vis_url) {
      viewOverlayImg.src = result.fusion_vis_url;
      viewOverlayImg.style.display = "block";
      viewBaseImg.src = result.fusion_vis_url;
    } else if (result.overlay_t2_url) {
      viewOverlayImg.src = result.overlay_t2_url;
      viewOverlayImg.style.display = "block";
    } else {
      viewOverlayImg.style.display = "none";
    }

    if (result.change_mask_url && viewMaskImg) {
      viewMaskImg.src = result.change_mask_url;
    }
    if (result.difference_map_url && viewDiffImg) {
      viewDiffImg.src = result.difference_map_url;
    }

    // 4. Specialist Status Tags
    semanticTags.innerHTML = "";
    if (result.top_predictions && result.top_predictions.length > 0) {
      result.top_predictions.forEach(p => {
        const tag = document.createElement("span");
        tag.className = "tag tag-executed";
        tag.textContent = `${p.class} (${(p.probability * 100).toFixed(0)}%)`;
        semanticTags.appendChild(tag);
      });
    } else if (result.specialist_statuses) {
      for (const [spec, status] of Object.entries(result.specialist_statuses)) {
        const tag = document.createElement("span");
        tag.className = status === "EXECUTED" ? "tag tag-executed" : "tag tag-unconfigured";
        tag.textContent = `${spec}: ${status}`;
        semanticTags.appendChild(tag);
      }
    }

    // 5. Execution Trace Timeline
    traceList.innerHTML = "";
    if (result.trace) {
      result.trace.forEach((step, idx) => {
        const li = document.createElement("li");
        li.className = "trace-item complete";
        li.innerHTML = `<span class="step-num">${idx + 1}</span> ${step}`;
        traceList.appendChild(li);
      });
    }

    // 6. Qwen2-VL Visual Context
    const vcCard = document.getElementById("visual-context-card");
    if (vcCard) {
      if (result.visual_context) {
        const vc = result.visual_context;
        const setTxt = (id, val) => {
          const el = document.getElementById(id);
          if (el) el.textContent = Array.isArray(val) ? (val.join(", ") || "None") : (val || "None");
        };
        setTxt("vc-image-type", vc.image_type);
        setTxt("vc-summary", vc.summary);
        setTxt("vc-land-cover", vc.land_cover);
        setTxt("vc-infrastructure", vc.infrastructure);
        setTxt("vc-terrain-water", [
          ...(Array.isArray(vc.terrain) ? vc.terrain : [vc.terrain]).filter(Boolean),
          ...(Array.isArray(vc.water_features) ? vc.water_features : [vc.water_features]).filter(x => x && x !== "none")
        ]);
        setTxt("vc-conditions", vc.notable_conditions);
        setTxt("vc-uncertainties", vc.uncertainties);
        vcCard.style.display = "block";
      } else {
        vcCard.style.display = "none";
      }
    }

    // 7. Report Export Actions
    btnDownloadMd.disabled = false;
    if (btnDownloadGeoJson) {
      btnDownloadGeoJson.disabled = !result.geojson;
    }

  }

  // ============================================================================
  // JARVIS 2.0 Controller — Autonomous Remote-Sensing Analyst Copilot
  // ============================================================================

  class JarvisController {
    constructor() {
      this.isOn = false;
      this.isListening = false;
      this.isSpeaking = false;
      this.audioPlayer = new Audio();
      this.speechQueue = [];
      this.recognition = null;

      // State context synchronized with workspace
      this.ctx = {
        image1Path: null,
        image2Path: null,
        image1Data: null,
        image2Data: null,
        hasImage1: false,
        hasImage2: false,
        currentTab: "interactive-overlay",
        lastResultId: null,
        lastResult: null,
        selectedCluster: 0,
        chatHistory: []
      };

      // Strict Action Registry
      this.ACTION_REGISTRY = new Set([
        "navigate_tab", "show_overlay", "show_side_by_side", "show_change_mask",
        "show_diff_map", "select_cluster", "highlight_cluster", "zoom_to_cluster",
        "show_grounding", "show_vqa", "show_optical_sar", "show_geospatial",
        "open_trace", "open_evidence", "open_impact", "open_summary",
        "generate_report", "show_why_this_result", "show_why_this_matters",
        "show_executive_summary", "show_agent_progress", "update_answer",
        "speak_response", "clear_selection", "reset_view"
      ]);

      this._initDOM();
      this._initVoice();
      this._bindEvents();
      this._initAudioPlayer();
    }

    _initDOM() {
      this._dom = {
        toggleBtn:     document.getElementById("jarvis-toggle"),
        toggleGroup:   document.getElementById("jarvis-toggle-group"),
        label:         document.getElementById("jarvis-label"),
        bar:           document.getElementById("jarvis-bar"),
        statusDot:     document.getElementById("jarvis-status-dot"),
        voiceIndicator:document.getElementById("jarvis-voice-indicator"),
        input:         document.getElementById("jarvis-input"),
        micBtn:        document.getElementById("jarvis-mic-btn"),
        submitBtn:     document.getElementById("jarvis-submit-btn"),
        conv:          document.getElementById("jarvis-conversation"),
        listeningTxt:  document.getElementById("jarvis-listening-text"),
        statusBadge:   document.getElementById("jarvis-status-badge"),
        voiceBadge:    document.getElementById("jarvis-voice-badge"),
        provenanceStrip: document.getElementById("jarvis-provenance-strip"),
        provenanceSource: document.getElementById("provenance-source"),
        provenanceId:  document.getElementById("provenance-id"),
        whyResultCard: document.getElementById("why-result-card"),
        whyResultModel: document.getElementById("why-result-model"),
        whyResultInputs: document.getElementById("why-result-inputs"),
        whyResultMeaning: document.getElementById("why-result-meaning"),
        whyResultConf: document.getElementById("why-result-confidence"),
        whyResultLimits: document.getElementById("why-result-limitations"),
        evidenceCard:  document.getElementById("evidence-panel-card"),
        evidenceItems: document.getElementById("evidence-items-list"),
        provenanceBadges: document.getElementById("provenance-badges-list"),
        whyMattersCard: document.getElementById("why-matters-card"),
        whyMattersText: document.getElementById("why-matters-text"),
        recommendedCard: document.getElementById("recommended-next-card"),
        recommendedText: document.getElementById("recommended-next-text"),
        summaryCard:   document.getElementById("exec-summary-card"),
        summaryBody:   document.getElementById("exec-summary-content"),
        traceCard:     document.getElementById("jarvis-trace-card"),
        traceList:     document.getElementById("jarvis-trace-list")
      };
    }

    _initAudioPlayer() {
      this.audioPlayer.addEventListener("ended", () => {
        this.isSpeaking = false;
        this._setVoiceState("IDLE");
        this._playNextInQueue();
      });
      this.audioPlayer.addEventListener("error", (e) => {
        console.warn("[JARVIS TTS] Audio playback failed, falling back to speech synthesis", e);
        this.isSpeaking = false;
        this._setVoiceState("IDLE");
      });
    }

    _bindEvents() {
      if (this._dom.toggleBtn) {
        this._dom.toggleBtn.addEventListener("click", () => this._toggle());
      }
      if (this._dom.submitBtn) {
        this._dom.submitBtn.addEventListener("click", () => this._submitQuery());
      }
      if (this._dom.input) {
        this._dom.input.addEventListener("keydown", (e) => {
          if (e.key === "Enter" && !e.shiftKey) {
            e.preventDefault();
            this._submitQuery();
          }
        });
      }
      if (this._dom.micBtn) {
        this._dom.micBtn.addEventListener("click", () => this._handleVoiceInput());
      }

      // Track active tabs
      document.querySelectorAll(".tab-btn").forEach(b => {
        b.addEventListener("click", () => {
          this.ctx.currentTab = b.dataset.tab || "interactive-overlay";
        });
      });

      // Listen for image state changes from main app
      window.addEventListener("satquery:image-changed", (e) => {
        const d = e.detail;
        this.ctx.image1Path = d.image1Path || null;
        this.ctx.image2Path = d.image2Path || null;
        this.ctx.image1Data = d.image1Data || null;
        this.ctx.image2Data = d.image2Data || null;
        this.ctx.hasImage1 = !!(d.image1Path || d.image1Data);
        this.ctx.hasImage2 = !!(d.image2Path || d.image2Data);
      });
    }

    _initVoice() {
      const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
      if (!SR) {
        if (this._dom && this._dom.micBtn) this._dom.micBtn.style.display = "none";
        return;
      }
      this.recognition = new SR();
      this.recognition.continuous = false;
      this.recognition.interimResults = true;
      this.recognition.lang = "en-US";

      this.recognition.onstart = () => {
        this.isListening = true;
        this._interruptSpeech();
        this._setVoiceState("LISTENING");
      };
      this.recognition.onend = () => {
        this.isListening = false;
        this._setVoiceState("IDLE");
      };
      this.recognition.onerror = () => {
        this.isListening = false;
        this._setVoiceState("IDLE");
      };

      this.recognition.onresult = (ev) => {
        const transcript = Array.from(ev.results).map(r => r[0].transcript).join("");
        this._dom.input.value = transcript;
        
        // Instant voice interruption check
        const lower = transcript.toLowerCase().trim();
        if (["stop", "pause", "be quiet", "shut up", "cancel"].includes(lower)) {
          this._interruptSpeech();
          this.recognition.stop();
          this._addMsg("user", transcript);
          this._addMsg("jarvis", "Speech stopped.");
          return;
        }

        if (ev.results[ev.results.length - 1].isFinal) {
          this.recognition.stop();
          setTimeout(() => this._submitQuery(), 400);
        }
      };
    }

    _setVoiceState(state) {
      const mic = this._dom.micBtn;
      const txt = this._dom.listeningTxt;
      const wave = this._dom.voiceIndicator;
      const dot = this._dom.statusDot;

      if (wave) wave.style.display = (state === "SPEAKING") ? "flex" : "none";

      if (state === "LISTENING") {
        if (mic) mic.classList.add("listening");
        if (txt) txt.textContent = "\uD83C\uDF99 LISTENING... Speak your request";
        if (dot) dot.style.background = "var(--accent-red)";
      } else if (state === "THINKING") {
        if (mic) mic.classList.remove("listening");
        if (txt) txt.textContent = "\u25C9 THINKING... Planning with qwen3:8b";
        if (dot) dot.style.background = "var(--accent-cyan)";
      } else if (state === "EXECUTING") {
        if (mic) mic.classList.remove("listening");
        if (txt) txt.textContent = "\u25C9 EXECUTING SPECIALIST... Running inference";
        if (dot) dot.style.background = "#6366f1";
      } else if (state === "SPEAKING") {
        if (mic) mic.classList.remove("listening");
        if (txt) txt.textContent = "\uD83D\uDD0A SPEAKING... Delivering analyst findings";
        if (dot) dot.style.background = "#10b981";
      } else {
        if (mic) mic.classList.remove("listening");
        if (txt) txt.textContent = "JARVIS ONLINE \u2014 Speak or type a command";
        if (dot) dot.style.background = "var(--accent-cyan)";
      }
    }

    _interruptSpeech() {
      this.audioPlayer.pause();
      this.audioPlayer.currentTime = 0;
      this.audioPlayer.src = "";
      this.speechQueue = [];
      if (window.speechSynthesis) {
        window.speechSynthesis.cancel();
      }
      this.isSpeaking = false;
      this._setVoiceState("IDLE");
    }

    _handleVoiceInput() {
      if (!this.recognition) return;
      if (this.isListening) {
        this.recognition.stop();
      } else {
        this._interruptSpeech();
        try {
          this.recognition.start();
        } catch (e) {
          console.warn("[JARVIS Voice]", e);
        }
      }
    }

    async _toggle() {
      if (this.isOn) {
        this._disable();
      } else {
        await this._enable();
      }
    }

    async _enable() {
      try {
        const resp = await fetch("/api/v1/jarvis/status");
        const status = await resp.json();
        if (!status.ollama_available) {
          alert(`JARVIS cannot start:\n${status.reason || "Ollama qwen3:8b is not available."}`);
          return;
        }
        if (this._dom.voiceBadge) {
          this._dom.voiceBadge.textContent = status.tts === "ready" ? "VOICE: Neural Ready" : "VOICE: Browser Fallback";
        }
      } catch {
        alert("JARVIS cannot start: SatQuery backend unreachable.");
        return;
      }

      this.isOn = true;
      this._dom.toggleBtn.classList.add("on");
      this._dom.toggleGroup.classList.add("online");
      this._dom.label.textContent = "\u25CF ONLINE";
      this._dom.bar.classList.add("visible");
      document.body.classList.add("jarvis-active");
      if (this._dom.conv) this._dom.conv.style.display = "flex";
      if (this._dom.statusBadge) {
        this._dom.statusBadge.textContent = "JARVIS \u25CF ONLINE";
        this._dom.statusBadge.classList.add("badge-agent-jarvis");
      }
      this._setVoiceState("IDLE");
      userPrompt.focus();
    }

    _disable() {
      this._interruptSpeech();
      this.isOn = false;
      this._dom.toggleBtn.classList.remove("on");
      this._dom.toggleGroup.classList.remove("online");
      this._dom.label.textContent = "OFF";
      this._dom.bar.classList.remove("visible");
      document.body.classList.remove("jarvis-active");
      if (this._dom.conv) this._dom.conv.style.display = "none";
      if (this._dom.statusBadge) {
        this._dom.statusBadge.textContent = "Agent Orchestrator Active";
        this._dom.statusBadge.classList.remove("badge-agent-jarvis");
      }
      // Hide JARVIS-only cards
      if (this._dom.provenanceStrip) this._dom.provenanceStrip.style.display = "none";
      if (this._dom.whyResultCard) this._dom.whyResultCard.style.display = "none";
      if (this._dom.recommendedCard) this._dom.recommendedCard.style.display = "none";
      this._dom.traceCard.classList.remove("visible");
      this._dom.whyMattersCard.classList.remove("visible");
      this._dom.summaryCard.classList.remove("visible");
    }

    _submitQuery() {
      const q = this._dom.input.value.trim() || userPrompt.value.trim();
      if (!q) return;
      this._interruptSpeech();
      this._dom.input.value = "";
      userPrompt.value = "";
      this._addMsg("user", q);
      this.handleQuery(q);
    }

    async handleQuery(query) {
      this.ctx.lastQuery = query;
      this._setVoiceState("THINKING");
      this._showThinking();
      this._dom.submitBtn.disabled = true;

      try {
        const payload = {
          query,
          has_image_1:     this.ctx.hasImage1,
          has_image_2:     this.ctx.hasImage2,
          image_1_path:    this.ctx.image1Path,
          image_2_path:    this.ctx.image2Path,
          image_1_data:    this.ctx.image1Data,
          image_2_data:    this.ctx.image2Data,
          current_tab:     this.ctx.currentTab,
          prior_result_id: this.ctx.lastResultId,
          selected_cluster:this.ctx.selectedCluster,
          voice_enabled:   true,
          chat_history:    this.ctx.chatHistory.slice(-6)
        };

        const resp = await fetch("/api/v1/jarvis", {
          method:  "POST",
          headers: { "Content-Type": "application/json" },
          body:    JSON.stringify(payload)
        });

        if (!resp.ok) {
          const err = await resp.json();
          throw new Error(err.detail || "JARVIS analysis failed");
        }

        const result = await resp.json();

        // Update context & history
        this.ctx.lastResultId = result.task_id;
        this.ctx.lastResult   = result;
        this.ctx.chatHistory.push({ role: "user", content: query });
        const ansText = result.response || result.answer || "Done.";
        this.ctx.chatHistory.push({ role: "assistant", content: ansText });

        // Show answer in conversation
        this._addMsg("jarvis", ansText);

        // Apply specialist result to main SatQuery UI
        if (result.evidence && Object.keys(result.evidence).length > 0) {
          currentResult = result.evidence;
          updateUIWithResults(result.evidence);
        }

        // Direct structured UI Action handler
        if (result.ui_action && window.handleJarvisUIAction) {
          window.handleJarvisUIAction(result.ui_action);
        }

        // Execute validated UI actions
        await this._executeActions(result.ui_actions || []);

        // Show JARVIS 2.0 Analyst Cards
        if (result.provenance) this._showProvenance(result.provenance);
        if (result.evidence && Object.keys(result.evidence).length > 0) {
          this._showEvidencePanel(result.evidence);
        }
        if (result.why_this_result && Object.keys(result.why_this_result).length > 0) {
          this._showWhyThisResult(result.why_this_result);
        }
        if (result.why_this_matters) this._showWhyThisMatters(result.why_this_matters);
        if (result.recommended_next_step) this._showRecommendedNext(result.recommended_next_step);
        if (result.executive_summary && Object.keys(result.executive_summary).length > 0) {
          this._showExecutiveSummary(result.executive_summary);
        }
        if (result.trace) this._showTrace(result.trace);

        // Voice Response
        if (result.audio_base64) {
          this._enqueueAudio(result.audio_base64, result.speak_text);
        } else if (result.speak_text) {
          this._enqueueSpeechSynthesis(result.speak_text);
        } else {
          this._setVoiceState("IDLE");
        }

      } catch (err) {
        this._addMsg("error", `Error: ${err.message}`);
        this._setVoiceState("IDLE");
      } finally {
        this._hideThinking();
        this._dom.submitBtn.disabled = false;
      }
    }

    // ---- Speech Queue & Playback ----

    _enqueueAudio(audioBase64, fallbackText) {
      this._interruptSpeech();
      this.speechQueue.push({ type: "audio", data: audioBase64, text: fallbackText });
      this._playNextInQueue();
    }

    _enqueueSpeechSynthesis(text) {
      this._interruptSpeech();
      this.speechQueue.push({ type: "tts", text });
      this._playNextInQueue();
    }

    _playNextInQueue() {
      if (!this.speechQueue.length) {
        this.isSpeaking = false;
        this._setVoiceState("IDLE");
        return;
      }

      const item = this.speechQueue.shift();
      this.isSpeaking = true;
      this._setVoiceState("SPEAKING");

      if (item.type === "audio") {
        this.audioPlayer.src = item.data;
        this.audioPlayer.play().catch(e => {
          console.warn("[JARVIS Audio Player] Playback error, falling back to speech synthesis", e);
          this._speakBrowserTTS(item.text);
        });
      } else {
        this._speakBrowserTTS(item.text);
      }
    }

    _speakBrowserTTS(text) {
      if (!window.speechSynthesis) {
        this.isSpeaking = false;
        this._setVoiceState("IDLE");
        return;
      }
      window.speechSynthesis.cancel();
      const utt = new SpeechSynthesisUtterance(text);
      utt.rate = 1.0;
      utt.pitch = 1.0;

      const voices = window.speechSynthesis.getVoices();
      const preferred = voices.find(v => v.lang.startsWith("en") && (v.name.includes("Natural") || v.name.includes("Google") || v.name.includes("Microsoft") || v.name.includes("David"))) || voices.find(v => v.lang.startsWith("en"));
      if (preferred) utt.voice = preferred;

      utt.onend = () => {
        this.isSpeaking = false;
        this._setVoiceState("IDLE");
        this._playNextInQueue();
      };
      utt.onerror = () => {
        this.isSpeaking = false;
        this._setVoiceState("IDLE");
      };
      window.speechSynthesis.speak(utt);
    }

    // ---- UI Action Registry Execution ----

    async _executeActions(actions) {
      for (const act of actions) {
        if (!this.ACTION_REGISTRY.has(act.action)) {
          console.warn("[JARVIS] Blocked unregistered action:", act.action);
          continue;
        }
        await this._runAction(act);
        await this._delay(300);
      }
    }

    async _runAction(act) {
      try {
        fetch("/api/v1/jarvis/action_ack", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ action: act.action, status: "success", timestamp: Date.now() })
        }).catch(() => {});
      } catch (e) {}

      switch (act.action) {
        case "navigate_tab": {
          const tabName = act.tab || "interactive-overlay";
          const btn = document.querySelector(`[data-tab="${tabName}"]`);
          if (btn) btn.click();
          break;
        }

        case "show_overlay":
          document.querySelector('[data-tab="interactive-overlay"]')?.click();
          break;

        case "show_side_by_side":
          document.querySelector('[data-tab="side-by-side"]')?.click();
          break;

        case "show_change_mask":
          document.querySelector('[data-tab="change-mask"]')?.click();
          break;

        case "show_diff_map":
        case "show_heatmap":
          document.querySelector('[data-tab="diff-map"]')?.click();
          break;

        case "show_grounding":
        case "show_vqa":
        case "show_optical_sar":
          document.querySelector('[data-tab="interactive-overlay"]')?.click();
          break;

        case "show_geospatial":
          document.querySelector('[data-tab="interactive-overlay"]')?.click();
          if (this._dom.provenanceStrip) this._dom.provenanceStrip.scrollIntoView({ behavior: "smooth" });
          break;

        case "show_agent_progress":
          if (act.steps) this._showTrace(act.steps);
          break;

        case "select_cluster":
        case "highlight_cluster":
        case "zoom_to_cluster": {
          const idx = typeof act.index === "number" ? act.index : 0;
          this.ctx.selectedCluster = idx;
          this._highlightCluster(idx);
          break;
        }

        case "open_evidence":
        case "show_evidence":
        case "show_why_this_result":
          if (this._dom.whyResultCard) this._dom.whyResultCard.style.display = "block";
          if (this._dom.evidenceCard) this._dom.evidenceCard.style.display = "block";
          break;

        case "open_impact":
        case "show_impact":
        case "show_why_this_matters":
          if (this._dom.whyMattersCard) this._dom.whyMattersCard.classList.add("visible");
          break;

        case "open_summary":
        case "show_summary":
        case "show_executive_summary":
          if (this._dom.summaryCard) this._dom.summaryCard.classList.add("visible");
          break;

        case "open_trace":
        case "show_trace":
          if (this._dom.traceCard) this._dom.traceCard.classList.add("visible");
          break;

        case "speak_response":
          if (act.text) this._enqueueSpeechSynthesis(act.text);
          break;

        case "update_answer": {
          const el = document.getElementById("ai-answer");
          if (el && act.text) el.textContent = act.text;
          break;
        }

        case "generate_report":
          document.getElementById("btn-download-md")?.click();
          break;

        case "clear_selection":
        case "reset_view":
          document.querySelectorAll(".jarvis-cluster-highlight").forEach(el => el.remove());
          break;

        default:
          break;
      }
    }

    _highlightCluster(idx) {
      document.querySelectorAll(".jarvis-cluster-highlight").forEach(el => el.remove());

      const result = this.ctx.lastResult;
      const bboxes = result?.evidence?.bboxes || currentResult?.bboxes || [];
      if (!bboxes || idx >= bboxes.length) return;

      const bboxEntry = bboxes[idx];
      const rawBox = bboxEntry.bbox || bboxEntry;
      if (!Array.isArray(rawBox) || rawBox.length < 4) return;

      const [ymin, xmin, ymax, xmax] = rawBox;

      // Navigate to overlay to show cluster
      document.querySelector('[data-tab="interactive-overlay"]')?.click();

      setTimeout(() => {
        const stage = document.getElementById("stage-overlay");
        const baseImg = document.getElementById("view-base-img");
        if (!stage || !baseImg || !baseImg.naturalWidth) return;

        const scaleX = baseImg.offsetWidth  / baseImg.naturalWidth;
        const scaleY = baseImg.offsetHeight / baseImg.naturalHeight;

        const hl = document.createElement("div");
        hl.className = "jarvis-cluster-highlight";
        hl.style.cssText = [
          `left:${xmin * scaleX}px`,
          `top:${ymin * scaleY}px`,
          `width:${(xmax - xmin) * scaleX}px`,
          `height:${(ymax - ymin) * scaleY}px`
        ].join(";");

        const label = document.createElement("div");
        label.className = "jarvis-cluster-label";
        label.textContent = `Region ${idx + 1}`;
        hl.appendChild(label);

        const bboxLayer = document.getElementById("bbox-overlay-layer");
        if (bboxLayer) bboxLayer.appendChild(hl);
      }, 250);
    }

    _showProvenance(p) {
      if (!this._dom.provenanceStrip) return;
      this._dom.provenanceStrip.style.display = "flex";
      this._dom.provenanceSource.textContent = p.source_model || "ChangeFormerV6";
      this._dom.provenanceId.textContent = `ID: ${p.result_id || "N/A"}`;
    }

    _showWhyThisResult(data) {
      if (!this._dom.whyResultCard) return;
      this._dom.whyResultCard.style.display = "block";
      if (this._dom.whyResultModel) this._dom.whyResultModel.textContent = data.model || "\u2014";
      if (this._dom.whyResultInputs) this._dom.whyResultInputs.textContent = data.inputs || "\u2014";
      if (this._dom.whyResultMeaning) this._dom.whyResultMeaning.textContent = data.output_meaning || "\u2014";
      if (this._dom.whyResultConf) this._dom.whyResultConf.textContent = data.confidence_type || "\u2014";
      if (this._dom.whyResultLimits) this._dom.whyResultLimits.textContent = data.limitations || "\u2014";
    }

    _showEvidencePanel(ev) {
      if (!this._dom.evidenceCard) return;
      this._dom.evidenceCard.style.display = "block";
      if (this._dom.evidenceItems) {
        const items = [];
        if (ev.change_percentage !== undefined) {
          items.push(`&#10003; Detected ${ev.change_percentage.toFixed(2)}% spatial difference across ${ev.bboxes ? ev.bboxes.length : 0} regions`);
        }
        if (ev.confidence !== undefined && ev.confidence !== null) {
          items.push(`&#10003; Model confidence: ${(ev.confidence * 100).toFixed(1)}% (${ev.confidence_source || 'Posterior Margin'})`);
        }
        if (ev.num_regions !== undefined) {
          items.push(`&#10003; OWL-ViT localized ${ev.num_regions} candidate bounding boxes`);
        }
        if (ev.optical_evidence) {
          items.push(`&#10003; Optical surface variance confirmed`);
        }
        if (ev.sar_evidence) {
          items.push(`&#10003; SAR radar backscatter variance registered`);
        }
        if (!items.length) {
          items.push("&#10003; Optical sensor input active");
        }
        this._dom.evidenceItems.innerHTML = items.map(it => `<li>${it}</li>`).join("");
      }
    }

    _showWhyThisMatters(text) {
      if (!this._dom.whyMattersCard) return;
      this._dom.whyMattersCard.classList.add("visible");
      this._dom.whyMattersText.textContent = text;
    }

    _showRecommendedNext(text) {
      if (!this._dom.recommendedCard) return;
      this._dom.recommendedCard.style.display = "block";
      this._dom.recommendedText.textContent = text;
    }

    _showExecutiveSummary(data) {
      if (!this._dom.summaryCard) return;
      this._dom.summaryCard.classList.add("visible");
      const rows = [
        ["Finding",            data.finding            || "\u2014"],
        ["Largest Change",     data.largest_change     || "\u2014"],
        ["Evidence",           data.evidence           || "\u2014"],
        ["Geospatial Info",    data.geospatial_info    || "\u2014"],
        ["Potential Relevance",data.potential_relevance|| "\u2014"],
        ["Limitations",        data.limitations        || "\u2014"]
      ];
      this._dom.summaryBody.innerHTML = rows.map(([label, val]) =>
        `<div class="exec-row">`+
        `<span class="exec-label">${label}</span>`+
        `<span class="exec-value">${val}</span>`+
        `</div>`
      ).join("");
    }

    _showTrace(steps) {
      if (!this._dom.traceCard) return;
      this._dom.traceCard.classList.add("visible");
      this._dom.traceList.innerHTML = "";
      (steps || []).forEach((step, i) => {
        const li = document.createElement("li");
        li.className = "trace-item complete";
        li.innerHTML = `<span class="step-num">${i + 1}</span> ${step}`;
        this._dom.traceList.appendChild(li);
      });
    }

    _addMsg(role, text) {
      const div = document.createElement("div");
      div.className = `jarvis-msg jarvis-msg-${role}`;
      const prefix = role === "user" ? "\u25B6" : role === "jarvis" ? "JARVIS" : "\u26A0";
      div.innerHTML = `<span class="jarvis-msg-role">${prefix}</span><span class="jarvis-msg-text">${text}</span>`;
      this._dom.conv.appendChild(div);
      this._dom.conv.scrollTop = this._dom.conv.scrollHeight;
    }

    _showThinking() {
      const d = document.createElement("div");
      d.id = "jarvis-thinking";
      d.className = "jarvis-thinking";
      d.innerHTML = "<span class=\"thinking-dot\"></span><span class=\"thinking-dot\"></span><span class=\"thinking-dot\"></span>";
      this._dom.conv.appendChild(d);
      this._dom.conv.scrollTop = this._dom.conv.scrollHeight;
    }

    _hideThinking() {
      document.getElementById("jarvis-thinking")?.remove();
    }

    _delay(ms) { return new Promise(r => setTimeout(r, ms)); }
  }

  // Instantiate JARVIS 2.0 Controller
  const jarvis = new JarvisController();
  window._jarvis = jarvis;

});


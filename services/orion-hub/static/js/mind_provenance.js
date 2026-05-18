(function (global) {
  const SOURCE_TAG_LABELS = new Set([
    "identity_yaml",
    "snapshot_source",
    "projection",
    "autonomy",
    "social_bridge",
  ]);

  function escapeHtml(value) {
    return String(value ?? "")
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;");
  }

  function asObject(value) {
    return value && typeof value === "object" && !Array.isArray(value) ? value : null;
  }

  function asList(value) {
    return Array.isArray(value) ? value : [];
  }

  function asBool(value) {
    return value === true;
  }

  function shortId(value, len) {
    const s = String(value || "").trim();
    if (!s) return "—";
    if (s.length <= (len || 8)) return s;
    return `${s.slice(0, len || 8)}…`;
  }

  function parseMindJsonbField(value) {
    if (value == null) return null;
    if (typeof value === "object") return value;
    if (typeof value === "string") {
      try {
        return JSON.parse(value);
      } catch (_err) {
        return { _unparsed: value };
      }
    }
    return null;
  }

  function machineContract(resultParsed) {
    const brief = asObject(resultParsed?.brief);
    return asObject(brief?.machine_contract) || asObject(resultParsed?.machine_contract) || {};
  }

  function phaseTelemetryRecords(mc) {
    return asList(mc["mind.phase_telemetry"]);
  }

  function findPhaseTelemetry(mc, phaseName) {
    return phaseTelemetryRecords(mc).find((row) => row && row.phase_name === phaseName) || null;
  }

  function trajectoryModelIds(resultParsed) {
    const patches = asList(resultParsed?.trajectory?.patches);
    return patches
      .map((p) => asObject(p)?.provenance?.model_id)
      .filter((m) => typeof m === "string" && m.trim());
  }

  function normalizeMindRunProvenance(run) {
    const resultParsed = parseMindJsonbField(run?.result_jsonb);
    const reqParsed = parseMindJsonbField(run?.request_summary_jsonb);
    const mc = machineContract(resultParsed);
    const brief = asObject(resultParsed?.brief);
    const llmEnabled = asBool(mc["mind.llm_synthesis_enabled"]);
    const llmAttempted = asBool(mc["mind.llm_synthesis_attempted"]);
    const failOpenToDeterministic = asBool(mc["mind.llm_fail_open_to_deterministic"]);
    const orchHttpFailed = asBool(mc["mind.orch_http_failed"]);
    const failedPhase = mc["mind.llm_synthesis_failed_phase"] || null;
    const mindQuality = String(brief?.mind_quality || resultParsed?.mind_quality || mc["mind.quality"] || "unknown");
    const authorizedSkip = asBool(brief?.mind_authorized_for_stance_skip) || asBool(mc["mind.authorized_for_stance_skip"]);
    const authorizedUse = asBool(mc["mind.authorized_for_stance_use"]);
    const models = trajectoryModelIds(resultParsed);
    const deterministicTrajectory = models.length > 0 && models.every((m) => m === "deterministic");

    let cognitionPath = "unknown";
    if (orchHttpFailed) {
      cognitionPath = "orch_mind_http_failed";
    } else if (run && run.ok === false && !orchHttpFailed) {
      cognitionPath = "error";
    } else if (failOpenToDeterministic || (llmAttempted && (deterministicTrajectory || mindQuality !== "meaningful_synthesis"))) {
      cognitionPath = "llm_fail_open_to_deterministic";
    } else if (llmAttempted && mindQuality === "meaningful_synthesis") {
      cognitionPath = "llm_synthesis";
    } else if (!llmEnabled && mindQuality === "shadow_synthesis") {
      cognitionPath = "deterministic_shadow";
    } else if (mindQuality === "fallback_contract_only") {
      cognitionPath = "deterministic_contract";
    } else if (mindQuality === "shadow_synthesis") {
      cognitionPath = "deterministic_shadow";
    } else if (deterministicTrajectory) {
      cognitionPath = "deterministic_shadow";
    }

    let finalSource = "unknown";
    if (mindQuality === "meaningful_synthesis" && authorizedUse) {
      finalSource = "mind_llm_handoff";
    } else if (mindQuality === "shadow_synthesis") {
      finalSource = "deterministic_shadow";
    } else if (mindQuality === "fallback_contract_only") {
      finalSource = "contract_only";
    } else if (deterministicTrajectory && !llmAttempted) {
      finalSource = "legacy_chat_stance";
    } else if (cognitionPath === "llm_fail_open_to_deterministic") {
      finalSource = mindQuality === "shadow_synthesis" ? "deterministic_shadow" : "contract_only";
    }

    const timing = asObject(resultParsed?.timing_ms_by_phase) || {};
    const totalMs = Number(timing.total_ms ?? timing["total_ms"] ?? mc["mind.wall_time_elapsed_ms"] ?? 0);

    return {
      cognition_path: cognitionPath,
      final_source: finalSource,
      llm_enabled: llmEnabled,
      llm_attempted: llmAttempted,
      fail_open_to_deterministic: failOpenToDeterministic,
      orch_http_failed: orchHttpFailed,
      failed_phase: failedPhase,
      error_code: run?.error_code || mc["mind.llm_synthesis_error_code"] || resultParsed?.error_code || null,
      error: mc["mind.llm_synthesis_error"] || null,
      routes: {
        semantic: mc["mind.semantic_route"] || null,
        appraisal: mc["mind.appraisal_route"] || null,
        stance: mc["mind.stance_route"] || null,
      },
      mind_quality: mindQuality,
      authorized_for_stance_skip: authorizedSkip,
      authorized_for_stance_use: authorizedUse,
      snapshot_hash: shortId(run?.snapshot_hash || resultParsed?.snapshot_hash, 12),
      mind_run_id: shortId(run?.mind_run_id || resultParsed?.mind_run_id, 12),
      correlation_id: shortId(run?.correlation_id || reqParsed?.correlation_id, 12),
      session_visibility: run?.session_visibility || null,
      total_ms: Number.isFinite(totalMs) ? Math.round(totalMs) : null,
      router_profile_id: run?.router_profile_id || null,
    };
  }

  function phaseStatusFromTelemetry(telemetry, mc, phaseName) {
    if (!telemetry) {
      if (phaseName === "deterministic_fallback") {
        return mc["mind.llm_fail_open_to_deterministic"] ? "fallback" : "not_attempted";
      }
      return "not_attempted";
    }
    if (telemetry.skipped) return "skipped";
    if (telemetry.status === "schema_invalid") return "schema_invalid";
    if (telemetry.ok === false) return "failed";
    if (phaseName === "semantic_synthesis" && telemetry.validation_ok === false) return "filtered";
    if (telemetry.ok === true) return "ok";
    return "failed";
  }

  function normalizeMindPhaseRows(run) {
    const resultParsed = parseMindJsonbField(run?.result_jsonb);
    const mc = machineContract(resultParsed);
    const prov = normalizeMindRunProvenance(run);
    const phases = ["semantic_synthesis", "active_frontier_judge", "stance_handoff"];
    const rows = phases.map((phaseName) => {
      const telemetry = findPhaseTelemetry(mc, phaseName);
      const routeKey =
        phaseName === "semantic_synthesis"
          ? "semantic"
          : phaseName === "active_frontier_judge"
            ? "appraisal"
            : "stance";
      let outputCount = null;
      if (phaseName === "semantic_synthesis") {
        outputCount = mc["mind.semantic_claim_count"];
      } else if (phaseName === "active_frontier_judge") {
        outputCount = mc["mind.active_frontier_selected_count"];
      }
      return {
        phase: phaseName,
        status: phaseStatusFromTelemetry(telemetry, mc, phaseName),
        route: (telemetry && telemetry.route) || prov.routes[routeKey] || null,
        model: (telemetry && telemetry.model) || null,
        elapsed_ms: telemetry && Number.isFinite(Number(telemetry.elapsed_ms)) ? Math.round(Number(telemetry.elapsed_ms)) : null,
        parse_ok: telemetry ? telemetry.parse_ok : null,
        validation_ok: telemetry ? telemetry.validation_ok : null,
        token_usage: telemetry ? telemetry.token_usage || {} : {},
        error: (telemetry && (telemetry.error || telemetry.skip_reason)) || null,
        output_count: outputCount,
      };
    });
    if (prov.orch_http_failed || prov.cognition_path === "orch_mind_http_failed") {
      const timing = asObject(resultParsed?.timing_ms_by_phase) || {};
      rows.push({
        phase: "orch_mind_http",
        status: "failed",
        route: null,
        model: null,
        elapsed_ms: Number.isFinite(Number(timing.orch_mind_http_timeout_ms))
          ? Math.round(Number(timing.orch_mind_http_timeout_ms))
          : prov.total_ms,
        parse_ok: null,
        validation_ok: null,
        token_usage: {},
        error: mc["mind.orch_http_error_type"] || resultParsed?.error_code || "orch_http_failed",
        output_count: null,
      });
    }
    if (prov.cognition_path === "llm_fail_open_to_deterministic" || prov.fail_open_to_deterministic) {
      const timing = asObject(resultParsed?.timing_ms_by_phase) || {};
      rows.push({
        phase: "deterministic_fallback",
        status: "fallback",
        route: "deterministic",
        model: "deterministic",
        elapsed_ms: Number.isFinite(Number(timing.loops_ms)) ? Math.round(Number(timing.loops_ms)) : null,
        parse_ok: true,
        validation_ok: true,
        token_usage: {},
        error: prov.error || (asList(mc["mind.fallback_reason"])[0] || null),
        output_count: null,
      });
    }
    return rows;
  }

  function textBlob(run) {
    const resultParsed = parseMindJsonbField(run?.result_jsonb);
    const mc = machineContract(resultParsed);
    const diag = asList(resultParsed?.diagnostics).join(" ");
    const fallback = asList(mc["mind.fallback_reason"]).join(" ");
    const labels = asList(mc["mind.semantic_claim_labels"]).join(" ");
    return `${diag} ${fallback} ${labels} ${mc["mind.llm_synthesis_error"] || ""}`.toLowerCase();
  }

  function hasSourceTagLeakage(run, mc) {
    const labels = asList(mc["mind.semantic_claim_labels"]);
    if (labels.some((label) => SOURCE_TAG_LABELS.has(String(label || "").toLowerCase()))) return true;
    const resultParsed = parseMindJsonbField(run?.result_jsonb);
    const brief = asObject(resultParsed?.brief);
    const suppressed = asList(brief?.semantic_synthesis?.suppressed);
    return suppressed.some((s) => String(s?.reason || "").includes("source_tag"));
  }

  function normalizeMindDerailments(run) {
    const resultParsed = parseMindJsonbField(run?.result_jsonb);
    const mc = machineContract(resultParsed);
    const brief = asObject(resultParsed?.brief);
    const prov = normalizeMindRunProvenance(run);
    const blob = textBlob(run);
    const callouts = [];
    const push = (item) => callouts.push(item);

    if (prov.orch_http_failed || asBool(mc["mind.orch_http_failed"])) {
      push({
        id: "orch_mind_http_failed",
        severity: "error",
        stage: "orch_mind_http",
        title: "Orch timed out waiting for Mind.",
        explanation:
          "Orch attempted /v1/mind/run but did not receive a MindRunResultV1 before ORION_MIND_TIMEOUT_SEC elapsed. Chat continued through legacy path.",
        next_action:
          "Ensure MIND_LLM_TIMEOUT_SEC < ORION_MIND_TIMEOUT_SEC; inspect Mind and LLM gateway logs for the same correlation.",
        evidence: {
          error_code: resultParsed?.error_code || run?.error_code,
          error_type: mc["mind.orch_http_error_type"],
          timeout_sec: mc["mind.orch_http_timeout_sec"],
          base_url: mc["mind.orch_http_base_url"],
        },
      });
    }

    if (prov.llm_attempted && (prov.fail_open_to_deterministic || prov.failed_phase)) {
      push({
        id: "llm_fail_open",
        severity: "error",
        stage: prov.failed_phase || "llm_pipeline",
        title: "Mind LLM synthesis failed open before a usable handoff.",
        explanation:
          `LLM synthesis was enabled and attempted, but ${prov.failed_phase || "the pipeline"} did not produce an authorized handoff. Mind fell back to ${prov.mind_quality} / deterministic output; Exec should not skip legacy chat_stance.`,
        next_action:
          `Check gateway routes (${prov.routes.semantic || "?"}/${prov.routes.appraisal || "?"}/${prov.routes.stance || "?"}), semantic JSON validity, and evidence_refs in semantic output.`,
        evidence: {
          error_code: prov.error_code,
          error: prov.error,
          failed_phase: prov.failed_phase,
          fallback_reason: asList(mc["mind.fallback_reason"]),
        },
      });
    }

    const semTelemetry = findPhaseTelemetry(mc, "semantic_synthesis");
    if (prov.failed_phase === "semantic_synthesis" || (semTelemetry && semTelemetry.ok === false)) {
      push({
        id: "semantic_failed",
        severity: "error",
        stage: "semantic_synthesis",
        title: "Semantic synthesis phase failed.",
        explanation: semTelemetry?.error
          ? `Semantic phase reported: ${semTelemetry.error}.`
          : "Semantic phase did not return usable claims.",
        next_action: `Inspect semantic route \`${prov.routes.semantic || "quick"}\` and LLM JSON output.`,
        evidence: { telemetry: semTelemetry, diagnostics: asList(resultParsed?.diagnostics) },
      });
    }

    if (
      (semTelemetry && semTelemetry.validation_ok === false) ||
      (Number(mc["mind.semantic_claim_count"]) === 0 && (semTelemetry || prov.llm_attempted))
    ) {
      push({
        id: "semantic_filtered",
        severity: "warning",
        stage: "semantic_synthesis",
        title: "Semantic claims were filtered or empty after guardrails.",
        explanation:
          "The semantic model responded, but guardrails removed all claims (unsupported evidence, source tags, or weak grounding).",
        next_action: "Review semantic claim labels, evidence_refs, and suppressed reasons in the raw synthesis JSON.",
        evidence: {
          semantic_claim_count: mc["mind.semantic_claim_count"],
          semantic_claim_labels: asList(mc["mind.semantic_claim_labels"]),
          suppressed: asList(brief?.semantic_synthesis?.suppressed),
        },
      });
    }

    if (hasSourceTagLeakage(run, mc)) {
      push({
        id: "source_tag_leakage",
        severity: "warning",
        stage: "semantic_synthesis",
        title: "Source-tag labels leaked into semantic claims.",
        explanation: "Semantic output used infrastructure/source labels instead of user-meaningful claims.",
        next_action: "Tune semantic prompt and verify guardrails suppress source_tag labels.",
        evidence: { semantic_claim_labels: asList(mc["mind.semantic_claim_labels"]) },
      });
    }

    if (blob.includes("unsupported_or_weak") || blob.includes("evidence_ref")) {
      push({
        id: "missing_evidence_refs",
        severity: "warning",
        stage: "semantic_synthesis",
        title: "Claims lacked supported evidence references.",
        explanation: "Diagnostics or fallback reasons mention missing/unsupported evidence refs.",
        next_action: "Ensure semantic claims cite current_turn / recall / projection refs present in the evidence pack.",
        evidence: { diagnostics: asList(resultParsed?.diagnostics), fallback_reason: asList(mc["mind.fallback_reason"]) },
      });
    }

    if (prov.mind_quality === "shadow_synthesis" && !prov.authorized_for_stance_skip) {
      push({
        id: "shadow_only",
        severity: "info",
        stage: "deterministic",
        title: "Shadow synthesis only — stance skip denied.",
        explanation: "Mind produced shadow synthesis from projection but did not authorize Exec to skip legacy stance.",
        next_action: "Inspect projection richness and shadow_synthesis fields; expect legacy chat_stance path.",
        evidence: { mind_quality: prov.mind_quality, authorized_for_stance_skip: prov.authorized_for_stance_skip },
      });
    }

    if (prov.mind_quality === "fallback_contract_only") {
      push({
        id: "contract_only",
        severity: "info",
        stage: "deterministic",
        title: "Contract-only Mind output.",
        explanation: "No meaningful synthesis; Mind returned a minimal contract for routing only.",
        next_action: "Check whether LLM synthesis was disabled, failed open, or projection-starved.",
        evidence: { mind_quality: prov.mind_quality, cognition_path: prov.cognition_path },
      });
    }

    const handoff = asObject(brief?.stance_handoff);
    if ((handoff && handoff.authorized_for_stance_use === false) || (prov.llm_attempted && !prov.authorized_for_stance_use)) {
      push({
        id: "handoff_rejected",
        severity: "warning",
        stage: "stance_handoff",
        title: "Mind handoff not authorized for Exec stance skip.",
        explanation:
          "Mind produced a handoff object but did not authorize stance use. Exec should keep legacy chat_stance synthesis.",
        next_action: "Review authorization_reasons and mind_quality; do not expect mind_skip_stance_synthesis.",
        evidence: {
          authorization_reasons: asList(mc["mind.authorization_reasons"]),
          mind_authorized_for_stance_skip: prov.authorized_for_stance_skip,
          authorized_for_stance_use: prov.authorized_for_stance_use,
        },
      });
    }

    const severityRank = { error: 0, warning: 1, info: 2 };
    callouts.sort((a, b) => (severityRank[a.severity] ?? 9) - (severityRank[b.severity] ?? 9));
    return callouts;
  }

  function normalizeMindDecision(run) {
    const prov = normalizeMindRunProvenance(run);
    const resultParsed = parseMindJsonbField(run?.result_jsonb);
    const mc = machineContract(resultParsed);
    let execHandoff = "unknown";
    if (prov.authorized_for_stance_use && prov.mind_quality === "meaningful_synthesis") {
      execHandoff = "would_skip_legacy_chat_stance";
    } else if (prov.llm_attempted || prov.mind_quality !== "meaningful_synthesis") {
      execHandoff = "legacy_chat_stance_expected";
    }
    return {
      exec_handoff: execHandoff,
      mind_skip_stance_synthesis: prov.authorized_for_stance_skip && prov.authorized_for_stance_use,
      authorization_reasons: asList(mc["mind.authorization_reasons"]),
    };
  }

  function chip(label, value, tone) {
    const toneClass =
      tone === "ok"
        ? "border-emerald-800/60 bg-emerald-950/40 text-emerald-200"
        : tone === "warn"
          ? "border-amber-800/60 bg-amber-950/40 text-amber-200"
          : tone === "bad"
            ? "border-rose-800/60 bg-rose-950/40 text-rose-200"
            : "border-gray-700 bg-gray-900/60 text-gray-200";
    return `<div class="rounded border px-2 py-1 ${toneClass}"><div class="text-[9px] uppercase tracking-wide text-gray-500">${escapeHtml(label)}</div><div class="mt-0.5 text-[11px] font-medium">${escapeHtml(value)}</div></div>`;
  }

  function renderMindProvenance(provenance) {
    const p = provenance || {};
    const authTone = p.authorized_for_stance_use ? "ok" : p.authorized_for_stance_skip ? "warn" : "bad";
    const chips = [
      chip("Cognition path", p.cognition_path || "unknown", p.cognition_path === "llm_synthesis" ? "ok" : "warn"),
      chip("Final source", p.final_source || "unknown", "neutral"),
      chip("LLM enabled", p.llm_enabled ? "yes" : "no", p.llm_enabled ? "ok" : "neutral"),
      chip("LLM attempted", p.llm_attempted ? "yes" : "no", p.llm_attempted ? "ok" : "neutral"),
      chip("Mind quality", p.mind_quality || "unknown", "neutral"),
      chip("Authorized", p.authorized_for_stance_use ? "yes" : "no", authTone),
    ];
    if (p.failed_phase) chips.push(chip("Failed phase", p.failed_phase, "bad"));
    if (p.routes?.semantic) chips.push(chip("Semantic route", p.routes.semantic, "neutral"));
    if (p.routes?.appraisal) chips.push(chip("Appraisal route", p.routes.appraisal, "neutral"));
    if (p.routes?.stance) chips.push(chip("Stance route", p.routes.stance, "neutral"));
    if (p.snapshot_hash) chips.push(chip("Snapshot", p.snapshot_hash, "neutral"));
    if (p.total_ms != null) chips.push(chip("Wall time", `${p.total_ms} ms`, "neutral"));
    return `<div class="mb-3 rounded border border-gray-800 bg-gray-950/50 p-3"><div class="text-[10px] font-semibold uppercase tracking-wide text-gray-400">Provenance</div><div class="mt-2 grid grid-cols-2 gap-2 md:grid-cols-3">${chips.join("")}</div><div class="mt-2 text-[10px] text-gray-500">run ${escapeHtml(p.mind_run_id || "—")} · corr ${escapeHtml(p.correlation_id || "—")} · visibility ${escapeHtml(p.session_visibility || "—")}</div></div>`;
  }

  function calloutSeverityClass(severity) {
    if (severity === "error") return "border-rose-800/70 bg-rose-950/30 text-rose-100";
    if (severity === "warning") return "border-amber-800/70 bg-amber-950/30 text-amber-100";
    return "border-emerald-800/60 bg-emerald-950/20 text-emerald-100";
  }

  function renderMindDerailmentCallouts(callouts) {
    if (!callouts || !callouts.length) {
      return `<div class="mb-3 rounded border border-emerald-900/50 bg-emerald-950/20 p-3"><div class="text-[11px] font-medium text-emerald-200">Mind path completed without derailment.</div></div>`;
    }
    const primary = callouts[0];
    const primaryHtml = `<div class="mb-3 rounded border p-3 ${calloutSeverityClass(primary.severity)}"><div class="text-[10px] uppercase tracking-wide opacity-80">Where it went off rails · ${escapeHtml(primary.stage || "mind")}</div><div class="mt-1 text-sm font-semibold">${escapeHtml(primary.title || "")}</div><div class="mt-2 text-[11px] leading-relaxed opacity-95">${escapeHtml(primary.explanation || "")}</div><div class="mt-2 text-[10px] uppercase tracking-wide opacity-70">Next action</div><div class="mt-1 text-[11px]">${escapeHtml(primary.next_action || "")}</div><details class="mt-2"><summary class="cursor-pointer text-[10px] opacity-80">Evidence</summary><pre class="mt-1 whitespace-pre-wrap break-words font-mono text-[10px] opacity-90">${escapeHtml(JSON.stringify(primary.evidence || {}, null, 2))}</pre></details></div>`;
    if (callouts.length === 1) return primaryHtml;
    const rest = callouts
      .slice(1)
      .map(
        (c) =>
          `<details class="rounded border border-gray-800 bg-gray-950/40 p-2"><summary class="cursor-pointer text-[11px] text-gray-300">[${escapeHtml(c.severity)}] ${escapeHtml(c.title || c.id || "callout")}</summary><div class="mt-2 text-[11px] text-gray-300">${escapeHtml(c.explanation || "")}</div><pre class="mt-2 whitespace-pre-wrap break-words font-mono text-[10px] text-gray-400">${escapeHtml(JSON.stringify(c.evidence || {}, null, 2))}</pre></details>`,
      )
      .join("");
    return `${primaryHtml}<details class="mb-3 rounded border border-gray-800 bg-gray-950/30 p-2"><summary class="cursor-pointer text-[11px] text-gray-300">Additional derailments (${callouts.length - 1})</summary><div class="mt-2 space-y-2">${rest}</div></details>`;
  }

  function renderMindPhaseTable(rows) {
    const header =
      '<thead><tr class="text-left text-[10px] uppercase tracking-wide text-gray-500"><th class="px-2 py-1">Phase</th><th class="px-2 py-1">Status</th><th class="px-2 py-1">Route</th><th class="px-2 py-1">Model</th><th class="px-2 py-1">Elapsed</th><th class="px-2 py-1">Parse</th><th class="px-2 py-1">Validation</th><th class="px-2 py-1">Output</th><th class="px-2 py-1">Error / skip</th></tr></thead>';
    const body = (rows || [])
      .map((row) => {
        return `<tr class="border-t border-gray-800/70 text-[11px] text-gray-200"><td class="px-2 py-1 font-mono">${escapeHtml(row.phase)}</td><td class="px-2 py-1">${escapeHtml(row.status)}</td><td class="px-2 py-1">${escapeHtml(row.route || "—")}</td><td class="px-2 py-1">${escapeHtml(row.model || "—")}</td><td class="px-2 py-1">${row.elapsed_ms != null ? `${row.elapsed_ms} ms` : "—"}</td><td class="px-2 py-1">${row.parse_ok == null ? "—" : row.parse_ok ? "ok" : "fail"}</td><td class="px-2 py-1">${row.validation_ok == null ? "—" : row.validation_ok ? "ok" : "fail"}</td><td class="px-2 py-1">${row.output_count != null ? escapeHtml(String(row.output_count)) : "—"}</td><td class="px-2 py-1 text-gray-400">${escapeHtml(row.error || "—")}</td></tr>`;
      })
      .join("");
    return `<div class="mb-3 overflow-x-auto rounded border border-gray-800 bg-gray-950/40 p-2"><div class="text-[10px] font-semibold uppercase tracking-wide text-gray-400">Phase provenance</div><table class="mt-2 w-full min-w-[720px] border-collapse">${header}<tbody>${body}</tbody></table></div>`;
  }

  function renderMindEvidenceSummary(run) {
    const resultParsed = parseMindJsonbField(run?.result_jsonb);
    const mc = machineContract(resultParsed);
    const rows = [
      ["cognitive_projection_seen", mc["mind.cognitive_projection_seen"]],
      ["cognitive_projection_id", mc["mind.cognitive_projection_id"]],
      ["cognitive_projection_item_count", mc["mind.cognitive_projection_item_count"]],
      ["semantic_claim_count", mc["mind.semantic_claim_count"]],
      ["active_frontier_selected_count", mc["mind.active_frontier_selected_count"]],
      ["active_frontier_top_labels", asList(mc["mind.active_frontier_top_labels"]).join(", ")],
      ["semantic_claim_labels", asList(mc["mind.semantic_claim_labels"]).join(", ")],
      ["fallback_reason", asList(mc["mind.fallback_reason"]).join(", ")],
    ]
      .filter(([, val]) => val != null && String(val).trim() !== "")
      .map(
        ([label, val]) =>
          `<div class="flex flex-col gap-0.5 border-b border-gray-800/60 pb-2 last:border-0 last:pb-0"><div class="text-[10px] uppercase tracking-wide text-gray-500">${escapeHtml(label)}</div><div class="break-words text-[11px] text-gray-200">${escapeHtml(String(val))}</div></div>`,
      );
    if (!rows.length) return "";
    return `<div class="mb-3 rounded border border-gray-800 bg-gray-950/50 p-3"><div class="text-[10px] font-semibold uppercase tracking-wide text-gray-400">Evidence / synthesis summary</div><div class="mt-2 space-y-2">${rows.join("")}</div></div>`;
  }

  function renderMindDecisionPanel(decision) {
    const d = decision || {};
    return `<div class="mb-3 rounded border border-gray-800 bg-gray-950/40 p-3 text-[11px] text-gray-300"><div class="text-[10px] font-semibold uppercase tracking-wide text-gray-400">Exec handoff expectation</div><div class="mt-2">Expected path: <span class="font-mono text-gray-100">${escapeHtml(d.exec_handoff || "unknown")}</span></div><div class="mt-1">mind_skip_stance_synthesis: <span class="font-mono">${d.mind_skip_stance_synthesis ? "true" : "false"}</span></div></div>`;
  }

  function itemCard(title, subtitle, bodyLines, tone) {
    const toneClass =
      tone === "semantic"
        ? "border-indigo-800/60 bg-indigo-950/25"
        : tone === "frontier"
          ? "border-violet-800/60 bg-violet-950/25"
          : tone === "shadow"
            ? "border-amber-800/60 bg-amber-950/25"
            : "border-gray-800 bg-gray-950/50";
    const body = (bodyLines || [])
      .filter((line) => line && String(line).trim())
      .map(
        (line) =>
          `<div class="text-[11px] leading-relaxed text-gray-200">${escapeHtml(String(line))}</div>`,
      )
      .join("");
    const bodyHtml = body
      ? '<div class="mt-2 space-y-1">' + body + '</div>'
      : "";
    return (
      '<div class="rounded border p-3 ' +
      toneClass +
      '"><div class="text-[11px] font-semibold text-gray-100">' +
      escapeHtml(title || "") +
      '</div><div class="mt-0.5 text-[10px] uppercase tracking-wide text-gray-500">' +
      escapeHtml(subtitle || "") +
      '</div>' +
      bodyHtml +
      '</div>'
    );
  }

  function renderMindBriefItems(run) {
    const resultParsed = parseMindJsonbField(run?.result_jsonb);
    const brief = asObject(resultParsed?.brief);
    if (!brief) return "";
    const cards = [];

    const sem = asObject(brief.semantic_synthesis);
    asList(sem?.claims).forEach((claimRaw) => {
      const claim = asObject(claimRaw);
      if (!claim) return;
      cards.push(
        itemCard(
          claim.label || claim.claim_id || "Semantic claim",
          claim.claim_kind || "claim",
          [
            claim.summary,
            claim.evidence_refs && claim.evidence_refs.length
              ? `evidence: ${claim.evidence_refs.join(", ")}`
              : null,
            claim.recommended_effect ? `effect: ${claim.recommended_effect}` : null,
          ],
          "semantic",
        ),
      );
    });

    const frontier = asObject(brief.active_frontier);
    asList(frontier?.selected).forEach((matterRaw) => {
      const matter = asObject(matterRaw);
      if (!matter) return;
      cards.push(
        itemCard(
          matter.label || matter.matter_id || "Frontier matter",
          matter.matter_kind || "selected",
          [matter.summary, matter.reason_selected, matter.recommended_effect ? `effect: ${matter.recommended_effect}` : null],
          "frontier",
        ),
      );
    });

    const shadow = asObject(brief.shadow_synthesis);
    if (shadow && (shadow.present || asList(shadow.attention_focus).length || asList(shadow.curiosity_candidate).length)) {
      cards.push(
        itemCard(
          "Shadow synthesis",
          shadow.relationship_frame || "deterministic shadow",
          [
            asList(shadow.attention_focus).length ? `attention: ${asList(shadow.attention_focus).join(" · ")}` : null,
            asList(shadow.curiosity_candidate).length ? `curiosity: ${asList(shadow.curiosity_candidate).join(" · ")}` : null,
            asList(shadow.projection_refs_used).length ? `projection refs: ${asList(shadow.projection_refs_used).join(" · ")}` : null,
            shadow.rationale,
            asList(shadow.hazards).length ? `hazards: ${asList(shadow.hazards).join(" · ")}` : null,
          ],
          "shadow",
        ),
      );
    }

    if (typeof brief.summary_one_paragraph === "string" && brief.summary_one_paragraph.trim()) {
      cards.push(itemCard("Mind summary", "brief", [brief.summary_one_paragraph.trim()], "neutral"));
    }

    if (!cards.length) return "";
    return `<div class="mb-3 rounded border border-gray-800 bg-gray-950/50 p-3"><div class="text-[10px] font-semibold uppercase tracking-wide text-gray-400">Mind synthesis items</div><div class="mt-2 grid gap-2">${cards.join("")}</div></div>`;
  }

  function renderMindProvenanceSections(run) {
    const provenance = normalizeMindRunProvenance(run);
    const callouts = normalizeMindDerailments(run);
    const phaseRows = normalizeMindPhaseRows(run);
    const decision = normalizeMindDecision(run);
    return [
      renderMindProvenance(provenance),
      renderMindBriefItems(run),
      renderMindDerailmentCallouts(callouts),
      renderMindPhaseTable(phaseRows),
      renderMindEvidenceSummary(run),
      renderMindDecisionPanel(decision),
    ].join("");
  }

  const MIND_PROVENANCE_FIXTURES = {
    failOpen: {
      ok: true,
      mind_run_id: "00000000-0000-4000-8000-000000000001",
      correlation_id: "corr-fail-open-001",
      snapshot_hash: "snap-fail-open",
      session_visibility: "session_match",
      result_jsonb: {
        ok: true,
        mind_quality: "fallback_contract_only",
        diagnostics: ["llm_fail_open:semantic_synthesis_failed", "llm_failed_phase:semantic_synthesis", "fake_exhausted"],
        timing_ms_by_phase: { semantic_synthesis_ms: 12.5, loops_ms: 4.2, total_ms: 28.0 },
        brief: {
          mind_quality: "fallback_contract_only",
          mind_authorized_for_stance_skip: false,
          machine_contract: {
            "mind.llm_synthesis_enabled": true,
            "mind.llm_synthesis_attempted": true,
            "mind.llm_fail_open_to_deterministic": true,
            "mind.llm_synthesis_failed_phase": "semantic_synthesis",
            "mind.llm_synthesis_error_code": "semantic_synthesis_failed",
            "mind.llm_synthesis_error": "fake_exhausted",
            "mind.semantic_route": "quick",
            "mind.appraisal_route": "metacog",
            "mind.stance_route": "chat",
            "mind.quality": "fallback_contract_only",
            "mind.authorized_for_stance_skip": false,
            "mind.authorized_for_stance_use": false,
            "mind.semantic_claim_count": 0,
            "mind.phase_telemetry": [
              {
                phase_name: "semantic_synthesis",
                route: "quick",
                model: "quick",
                ok: false,
                parse_ok: false,
                error: "fake_exhausted",
              },
            ],
            "mind.fallback_reason": ["semantic_synthesis_failed", "fake_exhausted"],
          },
        },
        trajectory: { patches: [{ provenance: { model_id: "deterministic" } }] },
      },
    },
    orchHttpFailed: {
      ok: false,
      mind_run_id: "00000000-0000-4000-8000-000000000099",
      correlation_id: "corr-orch-http-001",
      error_code: "mind_http_timeout",
      session_visibility: "session_match",
      result_jsonb: {
        ok: false,
        error_code: "mind_http_timeout",
        diagnostics: [
          "Orch failed calling /v1/mind/run",
          "exc_type=ReadTimeout",
          "ORION_MIND_TIMEOUT_SEC=45",
        ],
        timing_ms_by_phase: { orch_mind_http_timeout_ms: 45010.0 },
        brief: {
          mind_quality: "error",
          mind_authorized_for_stance_skip: false,
          summary_one_paragraph:
            "Orch timed out waiting for Mind before a MindRunResultV1 was returned.",
          machine_contract: {
            "mind.orch_http_failed": true,
            "mind.orch_http_error_type": "ReadTimeout",
            "mind.orch_http_timeout_sec": 45,
            "mind.orch_http_base_url": "http://orion-mind:6611",
            "mind.expected_path": "legacy_chat_stance_expected",
          },
        },
      },
    },
    shadowSynthesis: {
      ok: true,
      mind_run_id: "00000000-0000-4000-8000-000000000003",
      correlation_id: "corr-shadow-001",
      session_visibility: "session_match",
      result_jsonb: {
        ok: true,
        mind_quality: "shadow_synthesis",
        brief: {
          mind_quality: "shadow_synthesis",
          mind_authorized_for_stance_skip: false,
          shadow_synthesis: {
            present: true,
            attention_focus: ["evening plans with Amanda"],
            curiosity_candidate: ["ask about the show tone"],
            projection_refs_used: ["projection:relationship:0"],
            relationship_frame: "shared life moment",
            hazards: ["do not invent show details"],
            rationale: "Projection-rich shadow only.",
          },
          machine_contract: {
            "mind.llm_synthesis_enabled": false,
            "mind.quality": "shadow_synthesis",
            "mind.authorized_for_stance_skip": false,
            "mind.cognitive_projection_item_count": 2,
          },
        },
        trajectory: { patches: [{ provenance: { model_id: "deterministic" } }] },
      },
    },
    success: {
      ok: true,
      mind_run_id: "00000000-0000-4000-8000-000000000002",
      correlation_id: "corr-success-001",
      snapshot_hash: "snap-success",
      session_visibility: "session_match",
      result_jsonb: {
        ok: true,
        mind_quality: "meaningful_synthesis",
        timing_ms_by_phase: { semantic_synthesis_ms: 120, appraisal_ms: 80, stance_handoff_ms: 40, total_ms: 260 },
        brief: {
          mind_quality: "meaningful_synthesis",
          mind_authorized_for_stance_skip: true,
          semantic_synthesis: {
            claims: [
              {
                claim_id: "c1",
                label: "shared evening moment",
                summary: "User mentions watching a show with Amanda.",
                claim_kind: "relationship_claim",
                evidence_refs: ["current_turn:0"],
              },
            ],
            suppressed: [],
          },
          active_frontier: {
            selected: [
              {
                matter_id: "m1",
                label: "shared evening moment",
                summary: "Grounded in the current turn.",
                matter_kind: "relationship_opportunity",
              },
            ],
          },
          stance_handoff: { authorized_for_stance_use: true, authorization_reasons: ["valid_chat_stance_brief"] },
          machine_contract: {
            "mind.llm_synthesis_enabled": true,
            "mind.llm_synthesis_attempted": true,
            "mind.quality": "meaningful_synthesis",
            "mind.authorized_for_stance_skip": true,
            "mind.authorized_for_stance_use": true,
            "mind.semantic_claim_count": 1,
            "mind.active_frontier_selected_count": 1,
            "mind.semantic_route": "quick",
            "mind.appraisal_route": "metacog",
            "mind.stance_route": "chat",
            "mind.phase_telemetry": [
              { phase_name: "semantic_synthesis", route: "quick", ok: true, parse_ok: true, validation_ok: true, elapsed_ms: 120 },
              { phase_name: "active_frontier_judge", route: "metacog", ok: true, parse_ok: true, validation_ok: true, elapsed_ms: 80 },
              { phase_name: "stance_handoff", route: "chat", ok: true, parse_ok: true, validation_ok: true, elapsed_ms: 40 },
            ],
          },
        },
        trajectory: { patches: [{ provenance: { model_id: "chat" } }] },
      },
    },
  };

  const api = {
    normalizeMindRunProvenance,
    normalizeMindDerailments,
    normalizeMindPhaseRows,
    normalizeMindDecision,
    renderMindProvenance,
    renderMindDerailmentCallouts,
    renderMindPhaseTable,
    renderMindEvidenceSummary,
    renderMindDecisionPanel,
    renderMindBriefItems,
    renderMindProvenanceSections,
    MIND_PROVENANCE_FIXTURES,
  };

  global.OrionMindProvenance = api;
  if (typeof module !== "undefined" && module.exports) {
    module.exports = api;
  }
})(typeof window !== "undefined" ? window : globalThis);
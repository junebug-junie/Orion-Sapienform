(function (global) {
  const DEFAULT_RERUN_PROMPTS = {
    dream_cycle: 'Run your dream cycle',
    journal_pass: 'Do a journal pass',
    journal_discussion_window_pass: 'Journal the last hour',
    self_review: 'Run a self review',
    concept_induction_pass: 'Run through your concept induction graphs',
  };

  function toObject(value) {
    return value && typeof value === 'object' ? value : null;
  }

  function normalizeStatus(value) {
    const raw = String(value || '').trim().toLowerCase();
    if (!raw) return null;
    if (raw === 'success' || raw === 'completed' || raw === 'complete') return 'completed';
    if (raw === 'fail' || raw === 'failed' || raw === 'error') return 'failed';
    if (raw === 'requested' || raw === 'running') return raw;
    return raw;
  }

  function registryEntryFor(workflowId, registry) {
    const items = Array.isArray(registry) ? registry : [];
    return items.find((item) => item && typeof item === 'object' && String(item.workflow_id || '') === String(workflowId || '')) || null;
  }

  function normalizeWorkflow(workflowLike, options = {}) {
    const workflow = toObject(workflowLike);
    if (!workflow) return null;
    const rawMetadata = toObject(workflow.raw_metadata) || workflow;
    const workflowId = String(
      workflow.id
      || workflow.workflow_id
      || rawMetadata.id
      || rawMetadata.workflow_id
      || ''
    ).trim();
    if (!workflowId) return null;
    const registryEntry = registryEntryFor(workflowId, options.availableWorkflows || workflow.available_workflows || rawMetadata.available_workflows);
    const request = toObject(workflow.request) || toObject(workflow.workflow_request) || toObject(options.workflowRequest) || {};
    const persisted = Array.isArray(workflow.persisted) ? workflow.persisted : Array.isArray(rawMetadata.persisted) ? rawMetadata.persisted : [];
    const scheduled = Array.isArray(workflow.scheduled) ? workflow.scheduled : Array.isArray(rawMetadata.scheduled) ? rawMetadata.scheduled : [];
    const displayName = String(
      workflow.display_name
      || rawMetadata.display_name
      || (registryEntry && registryEntry.display_name)
      || workflowId.replace(/_/g, ' ')
    ).trim();
    const summary = String(
      workflow.summary
      || workflow.main_result
      || rawMetadata.main_result
      || rawMetadata.summary
      || ''
    ).trim();
    const status = normalizeStatus(
      workflow.status
      || rawMetadata.status
      || options.status
    );
    const userInvocable = typeof workflow.user_invocable === 'boolean'
      ? workflow.user_invocable
      : typeof rawMetadata.user_invocable === 'boolean'
        ? rawMetadata.user_invocable
        : Boolean((registryEntry && registryEntry.user_invocable) || false);
    const actionAssistanceUsed = Boolean(
      (Array.isArray(rawMetadata.actions_used) && rawMetadata.actions_used.length)
      || (Array.isArray(rawMetadata.scheduled) && rawMetadata.scheduled.length)
      || (Array.isArray(rawMetadata.notifications) && rawMetadata.notifications.length)
      || false
    );
    const rerunPrompt = String(
      workflow.rerun_prompt
      || rawMetadata.rerun_prompt
      || request.matched_alias
      || DEFAULT_RERUN_PROMPTS[workflowId]
      || ''
    ).trim() || null;
    return {
      id: workflowId,
      display_name: displayName,
      status,
      summary: summary || null,
      user_invocable: userInvocable,
      persisted,
      scheduled,
      action_assistance_used: actionAssistanceUsed,
      matched_alias: request.matched_alias || null,
      invoked_from_chat: Boolean(request.invoked_from_chat || workflow.invoked_from_chat),
      rerun_prompt: rerunPrompt,
      raw_metadata: rawMetadata,
      request,
      registry_entry: registryEntry,
    };
  }

  function extractWorkflow(messageLike) {
    if (!messageLike || typeof messageLike !== 'object') return null;
    const direct = normalizeWorkflow(messageLike.workflow, { status: messageLike.status });
    if (direct) return direct;
    const metadata = toObject(messageLike.metadata) || toObject(messageLike.raw && messageLike.raw.metadata) || {};
    const fromMetadata = normalizeWorkflow(metadata.workflow, {
      status: messageLike.status,
      availableWorkflows: metadata.available_workflows,
      workflowRequest: metadata.workflow_request,
    });
    if (fromMetadata) return fromMetadata;
    const context = toObject(messageLike.context) || {};
    return normalizeWorkflow(context.workflow, { status: messageLike.status });
  }

  function shouldShowWorkflow(workflowLike) {
    return Boolean(normalizeWorkflow(workflowLike));
  }

  function shouldShowWorkflowForMessage(messageLike) {
    return Boolean(extractWorkflow(messageLike));
  }

  function getWorkflowBadgeLabel(workflowLike) {
    const workflow = normalizeWorkflow(workflowLike);
    if (!workflow) return '';
    return `Workflow · ${workflow.display_name}`;
  }

  function getWorkflowStatusLabel(status) {
    switch (normalizeStatus(status)) {
      case 'requested': return 'Requested';
      case 'running': return 'Running';
      case 'completed': return 'Completed';
      case 'failed': return 'Failed';
      default: return '';
    }
  }

  function buildWorkflowDetailRows(workflowLike) {
    const workflow = normalizeWorkflow(workflowLike);
    if (!workflow) return [];
    return [
      ['Workflow', workflow.display_name],
      ['Workflow ID', workflow.id],
      ['Status', getWorkflowStatusLabel(workflow.status) || workflow.status || '--'],
      ['Summary', workflow.summary || '--'],
      ['Persisted', workflow.persisted.length ? workflow.persisted.join(', ') : 'none'],
      ['Scheduled', workflow.scheduled.length ? workflow.scheduled.join(', ') : 'none'],
      ['Actions assistance', workflow.action_assistance_used ? 'used' : 'not used'],
      ['User-invocable', workflow.user_invocable ? 'yes' : 'no'],
    ].filter((row) => row[1] !== null && row[1] !== undefined && row[1] !== '');
  }

  function canRunAgain(workflowLike) {
    const workflow = normalizeWorkflow(workflowLike);
    if (!workflow) return false;
    if (!workflow.user_invocable) return false;
    if (workflow.status !== 'completed') return false;
    return Boolean(workflow.rerun_prompt);
  }

  function normalizeConceptInductionDetails(workflowLike) {
    const workflow = normalizeWorkflow(workflowLike);
    if (!workflow || workflow.id !== 'concept_induction_pass') return null;
    const raw = toObject(workflow.raw_metadata) || {};
    const details = toObject(raw.concept_induction_details);
    if (!details) return null;
    const profiles = Array.isArray(details.profiles) ? details.profiles.filter((item) => item && typeof item === 'object') : [];
    const trace = toObject(details.trace) || {};
    const boundedTraceRows = Array.isArray((trace.artifacts || {}).lookup_rows)
      ? trace.artifacts.lookup_rows.filter((row) => row && typeof row === 'object').slice(0, 12)
      : [];
    return {
      generated_at: typeof details.generated_at === 'string' ? details.generated_at : null,
      profiles: profiles.slice(0, 6).map((profile) => ({
        subject: String(profile.subject || '').trim() || null,
        profile_id: String(profile.profile_id || '').trim() || null,
        revision: Number.isFinite(Number(profile.revision)) ? Number(profile.revision) : null,
        created_at: typeof profile.created_at === 'string' ? profile.created_at : null,
        window_start: typeof profile.window_start === 'string' ? profile.window_start : null,
        window_end: typeof profile.window_end === 'string' ? profile.window_end : null,
        concept_count: Number.isFinite(Number(profile.concept_count)) ? Number(profile.concept_count) : 0,
        cluster_count: Number.isFinite(Number(profile.cluster_count)) ? Number(profile.cluster_count) : 0,
        concepts: Array.isArray(profile.concepts) ? profile.concepts.slice(0, 30) : [],
        clusters: Array.isArray(profile.clusters) ? profile.clusters.slice(0, 20) : [],
        state_estimate: toObject(profile.state_estimate) || null,
      })),
      trace: {
        repository_resolution: toObject(trace.repository_resolution) || {},
        artifacts: {
          lookup_rows: boundedTraceRows,
        },
      },
    };
  }

  function buildConceptInductionSections(workflowLike) {
    const details = normalizeConceptInductionDetails(workflowLike);
    if (!details) return [];
    return ['Overview', 'Profiles', 'Concepts', 'Clusters', 'State estimate', 'Trace / Artifacts'];
  }

  const api = {
    normalizeWorkflow,
    extractWorkflow,
    shouldShowWorkflow,
    shouldShowWorkflowForMessage,
    getWorkflowBadgeLabel,
    getWorkflowStatusLabel,
    buildWorkflowDetailRows,
    canRunAgain,
    normalizeConceptInductionDetails,
    buildConceptInductionSections,
  };

  global.OrionWorkflowUI = api;
  if (typeof module !== 'undefined' && module.exports) {
    module.exports = api;
  }
})(typeof window !== 'undefined' ? window : globalThis);

const test = require('node:test');
const assert = require('node:assert/strict');
const workflowUi = require('./workflow-ui.js');

test('normalizeWorkflow preserves compact summary for existing workflow surfaces', () => {
  const workflow = workflowUi.normalizeWorkflow({
    workflow_id: 'concept_induction_pass',
    status: 'completed',
    main_result: 'orion rev 2 with 5 concepts / 2 clusters',
  });
  assert.equal(workflow.summary, 'orion rev 2 with 5 concepts / 2 clusters');
});

test('normalizeConceptInductionDetails provides bounded trace artifacts', () => {
  const normalized = workflowUi.normalizeConceptInductionDetails({
    workflow_id: 'concept_induction_pass',
    concept_induction_details: {
      generated_at: '2026-03-30T00:00:00Z',
      profiles: [{ subject: 'orion', profile_id: 'profile-1', revision: 3, concepts: [{ concept_id: 'c1' }], clusters: [], concept_count: 1, cluster_count: 0 }],
      trace: {
        repository_resolution: { requested_backend: 'graph', resolved_backend: 'graph', fallback_used: false },
        artifacts: {
          lookup_rows: Array.from({ length: 20 }, (_, i) => ({ subject: `s${i}`, availability: 'available' })),
        },
      },
    },
  });
  assert.equal(normalized.profiles[0].profile_id, 'profile-1');
  assert.equal(normalized.trace.artifacts.lookup_rows.length, 12);
});

test('buildConceptInductionSections returns modal sections for concept induction pass', () => {
  const sections = workflowUi.buildConceptInductionSections({
    workflow_id: 'concept_induction_pass',
    concept_induction_details: { profiles: [], trace: {} },
  });
  assert.deepEqual(sections, ['Overview', 'Profiles', 'Concepts', 'Clusters', 'State estimate', 'Trace / Artifacts']);
});

const test = require('node:test');
const assert = require('node:assert/strict');
const socialInspection = require('./social-inspection.js');

test('buildOperatorSummary keeps operator-facing routing fields and bounded badges', () => {
  const routeDebug = {
    social_thread_routing: {
      routing_decision: 'reply_to_peer',
      audience_scope: 'peer',
      primary_thread_summary: 'Pacing the room after a correction.',
    },
    social_epistemic_decision: {
      decision_kind: 'qualify_claim',
    },
    social_repair_decision: {
      decision_kind: 'repair_softly',
    },
    social_gif_policy: {
      decision: 'plain_text_only',
    },
    social_handoff_signal: {
      handoff_kind: 'yield_to_peer',
    },
    social_inspection: {
      platform: 'callsyne',
      room_id: 'room-alpha',
      participant_id: 'peer-1',
      thread_key: 'thread-1',
      summary: '11 sections · 4 context candidates selected · 1 softened · 2 excluded',
      metadata: { safety_omissions: '1' },
      sections: [
        {
          section_kind: 'context_window',
          selected_state: ['thread: pacing after a correction', 'peer continuity: Juniper asked for a grounded answer'],
          softened_state: ['ritual: older playful cue'],
          excluded_state: ['freshness_hint: stale summary'],
          included_artifact_summaries: ['thread: pacing after a correction'],
          decision_traces: [],
        },
        {
          section_kind: 'claims',
          selected_state: ['claim: the room is split on pacing'],
          softened_state: ['consensus: summary language'],
          excluded_state: ['consensus: older settled framing'],
          included_artifact_summaries: ['claim: the room is split on pacing'],
          decision_traces: [],
        },
        {
          section_kind: 'commitments',
          selected_state: ['summarize_room: give a brief recap before switching topics'],
          softened_state: [],
          excluded_state: [],
          included_artifact_summaries: ['summarize_room: give a brief recap before switching topics'],
          decision_traces: [],
        },
        {
          section_kind: 'freshness',
          selected_state: ['claim_consensus: refresh before treating as settled'],
          softened_state: [],
          excluded_state: ['consensus: older settled framing'],
          included_artifact_summaries: ['claim_consensus: refresh before treating as settled'],
          decision_traces: [],
        },
        {
          section_kind: 'resumptive',
          selected_state: ['Resume from pacing the room, but verify that it is still live.'],
          softened_state: [],
          excluded_state: [],
          included_artifact_summaries: ['Resume from pacing the room, but verify that it is still live.'],
          decision_traces: [],
        },
        {
          section_kind: 'safety',
          selected_state: [],
          softened_state: [],
          excluded_state: ['1 blocked/private summaries omitted from inspection'],
          included_artifact_summaries: [],
          decision_traces: [],
        },
      ],
    },
  };

  const model = socialInspection.buildOperatorSummary(routeDebug, routeDebug.social_inspection, null);

  assert.equal(model.summaryRows[0].value, 'reply_to_peer');
  assert.equal(model.summaryRows[1].value, 'peer');
  assert.equal(model.summaryRows[2].value, 'Pacing the room after a correction.');
  assert.equal(model.summaryRows[3].value, 'qualify_claim');
  assert.equal(model.summaryRows[4].value, 'repair_softly · yield_to_peer');
  assert.equal(model.badges.find((badge) => badge.label === 'Context').value, '2 selected');
  assert.equal(model.badges.find((badge) => badge.label === 'Safety').value, '1 omitted');
});

test('normalizeSnapshot omits blocked/private phrases before rendering', () => {
  const snapshot = socialInspection.normalizeSnapshot({
    platform: 'callsyne',
    room_id: 'room-alpha',
    summary: '2 sections',
    sections: [
      {
        section_kind: 'claims',
        included_artifact_summaries: ['claim: visible summary', 'claim: private thing'],
        selected_state: ['claim: visible summary', 'claim: sealed note'],
        softened_state: [],
        excluded_state: ['claim: secret detail'],
        decision_traces: [
          {
            trace_kind: 'claim_divergence',
            summary: 'private thing',
            why_it_mattered: 'should not render',
          },
        ],
      },
    ],
  });

  assert.deepEqual(snapshot.sections[0].included_artifact_summaries, ['claim: visible summary']);
  assert.deepEqual(snapshot.sections[0].selected_state, ['claim: visible summary']);
  assert.deepEqual(snapshot.sections[0].excluded_state, []);
  assert.equal(snapshot.sections[0].decision_traces.length, 0);
});

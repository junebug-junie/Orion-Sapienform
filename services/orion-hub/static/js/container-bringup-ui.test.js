const test = require('node:test');
const assert = require('node:assert/strict');
const containerBringupUi = require('./container-bringup-ui.js');

test('summaryLabelForOpenState reflects open state so the label never goes stale after a manual collapse', () => {
  assert.equal(containerBringupUi.summaryLabelForOpenState(true), 'Result (click to collapse)');
  assert.equal(containerBringupUi.summaryLabelForOpenState(false), 'Result (click to expand)');
});

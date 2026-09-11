select
    outcome_key,
    outcome_label,
    outcome_description,
    sort_order
from (
    values
        ('pressure_discharged', 'Pressure discharged', 'At least one thought was recorded and the observed pressure reached the code-defined discharge threshold.', 1),
        ('max_steps', 'Step limit reached', 'The chain recorded the configured maximum number of thought steps.', 2),
        ('no_coalition', 'No coalition', 'The first chain step produced no thought from the current attention coalition.', 3),
        ('low_salience', 'Below configured salience floor', 'The chain stopped because its existing EMA salience fell below the configured minimum.', 4),
        ('refractory', 'Refractory suppression', 'Declared by ReverieChainV1, but the current runner returns before persisting a text chain when a theme is suppressed.', 5),
        ('generation_failed', 'Generation failed', 'The visual chain stopped because image generation or local artifact storage raised an error.', 6),
        ('run_deadline_exceeded', 'Run deadline exceeded', 'The visual chain exceeded its whole-run deadline and released its single-flight lock.', 7),
        ('thermal_refused', 'Thermal refusal', 'The visual chain recorded a deliberate refusal to spend GPU capacity while the configured thermal gate was active.', 8)
) as outcomes(outcome_key, outcome_label, outcome_description, sort_order)

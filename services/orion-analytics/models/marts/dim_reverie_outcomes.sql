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
        ('refractory', 'Refractory suppression', 'Declared by ReverieChainV1, but the current runner returns before persisting a chain when a theme is suppressed.', 5)
) as outcomes(outcome_key, outcome_label, outcome_description, sort_order)

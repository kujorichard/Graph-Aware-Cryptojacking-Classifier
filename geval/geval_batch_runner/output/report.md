# Adapted G-Eval evaluation report
Valid evaluations: 261
Counts by status: {"success": 261, "generation_error": 1}
Hallucination/faithfulness denominators include only valid completed judge evaluations; other outcomes are reported separately.
Sample SD (n-1); unavailable when fewer than two valid ratings.
N/A means required data is unavailable, not a score of zero. Percentages are 0-100; quality ratings are 1-5.
Consistency uses repeated original explanations, never repeated judge responses. Function agreement uses exact unordered cited-name sets.
No claim-level rate is computed. A high score does not establish detection accuracy or independent external validation.

## Table 4.13. Hallucination rate

| Explanation | Count | Percentage |
| --- | --- | --- |
| Explanations with no unsupported claims | 0 | 0.0000 |
| Explanations with >=1 unsupported claim | 261 | 100.0000 |
| Total evaluated | 261 | 100.0000 |

## Table 4.14. Faithfulness assessment

| Assessment | Count | Percentage |
| --- | --- | --- |
| Faithful to SHAP sign and magnitude | 205 | 78.5441 |
| Faithfulness discrepancy | 56 | 21.4559 |
| Total evaluated | 261 | 100.0000 |

## Table 4.15. Consistency across repeated generator runs

| Measure | Value |
| --- | --- |
| Samples re-run | 0 |
| Runs per sample | N/A |
| Tier agreement across runs | N/A |
| Agreement on top cited functions | N/A |

## Table 4.16. G-Eval scores by quality dimension

| Dimension | Mean score | SD | Scale used | N |
| --- | --- | --- | --- | --- |
| Technical accuracy | 3.2337 | 0.5433 | 1-5 | 261 |
| Coherence | 4.0575 | 0.3171 | 1-5 | 261 |
| Cybersecurity relevance | 4.1303 | 0.4866 | 1-5 | 261 |
| Actionable utility | 4.0383 | 0.2747 | 1-5 | 261 |

Consistency: 0 complete repeat groups; 0 groups with comparable nonempty function lists.

Review a representative subset with human raters using the same rubric before interpreting automated scores.

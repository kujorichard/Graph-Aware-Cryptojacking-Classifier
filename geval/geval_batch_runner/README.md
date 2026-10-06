# Batch G-Eval runner — Claude Sonnet 5.5

Python 3.10+; no pip packages required. This is an adapted G-Eval evaluation with
integer ratings and an unsupported-claim audit, not original probability-
weighted G-Eval. It evaluates explanations, not RF detection accuracy.

## Windows: run all 261 results

1. Extract the ZIP. Open the folder containing `geval_batch.py`.
2. Open `geval_batch.py` in a text editor. Near the top, replace:

```python
API_KEY = "PASTE_YOUR_ANTHROPIC_API_KEY_HERE"
```

with your actual Anthropic API key inside the quotes. Save the file.
`ANTHROPIC_API_KEY` in the environment is also accepted as a fallback. The code
key takes precedence when it is not the placeholder. Keys are excluded from
saved configuration and redacted from errors and saved responses.

3. Put your 261 analysis JSON files directly inside `input_files/`. The provided
   `example_analysis.json` is outside that folder so it is not accidentally
   counted as an extra sample. Inputs must be combined `_analysis.json` reports
   like your upload, or successful analyzer-result JSON files. Failed explanation
   results remain in the batch counts but are not scored. Do not add raw Ghidra
   disassembly JSON or generated G-Eval outputs to this input folder.
4. In File Explorer, open the folder containing the script, type `powershell`
   into the address bar and press Enter.
5. Check Python and perform a no-API validation:

```powershell
python --version
python geval_batch.py --dry-run
```

The dry run validates and prepares all files, then writes reports with no judged
scores. `N/A` at this stage is expected. It makes no API calls and needs no key.
A file count different from 261 prints a notice and uses the actual count.

6. For a first live check, evaluate three files:

```powershell
python geval_batch.py --limit 3
```

This makes paid API requests to your configured Anthropic account. Review the
three result files and rubric during your pilot before final evaluation. If you
refine the rubric, treat those as pilot results and use a separate final output
folder with the revised, frozen rubric. Do not keep only favorable pilot scores.

7. Run the full batch:

```powershell
python geval_batch.py
```

With unchanged settings, the first three successful results are reused, so the
runner evaluates only the remaining eligible files. Completed valid results are
resumed on subsequent runs without additional API calls. Failed/interrupted
samples are attempted again. Use Ctrl+C to stop; saved progress is retained.
The interrupted in-flight request might incur provider usage even if no response
was received. The manifest records this uncertainty.

8. Open `output/report.html` in your browser for the tables. Use the CSV files
   for Excel or copying values into Chapter 4, and inspect `output/summary.json`
   for full counts, settings and denominators.

To use another folder without editing code:

```powershell
python geval_batch.py --input "D:\YourProject\analysis_results" --output "D:\YourProject\geval_output"
```

The input scan is nonrecursive: put the JSON files directly in the selected folder.
Output must be outside input. Each binary/ngram identity is evaluated once;
duplicate copies are reported rather than charged/evaluated twice.

## Outputs matching your screenshots

| File | Contents |
|---|---|
| `table_4_13_hallucination.csv` | No unsupported claims; at least one unsupported claim; total evaluated; counts and percentages |
| `table_4_14_faithfulness.csv` | Faithful to SHAP sign/magnitude; discrepancy; total evaluated; counts and percentages |
| `table_4_15_consistency.csv` | Samples re-run; runs per sample; tier agreement; cited-function agreement |
| `table_4_16_quality.csv` | Technical accuracy, coherence, cybersecurity relevance, actionable utility; mean, sample SD, 1–5 scale and N |
| `per_sample_scores.csv` | One row per selected input, including failures and duplicates |
| `unsupported_claims.csv` | Exact candidate quotes, categories, evidence references and short reasons |
| `faithfulness_discrepancies.csv` | Mechanical numeric discrepancies and judge-identified semantic discrepancies |
| `consistency_details.json` | Per-sample repeat agreement, evidence mismatch/missing-data reasons and repeat-input errors |
| `report.html`, `report.md` | Readable versions of the four tables and their qualifications |
| `summary.json`, `batch_manifest.json` | Aggregates and complete outcome ledger |
| `results/*.json` | Individual scores, justifications, raw provider blocks, attempt history, usage and provenance |
| `prepared_inputs/*.json` | Blinded judge inputs containing evidence and unchanged final explanations |
| `rubric.json`, `response_schema.json` | Exact evaluation protocol and response structure used |

CSV percentages use 0–100, not fractions. Quality scores use 1–5. CSVs use UTF-8
with BOM for Excel. Missing numeric values are blank in CSV and `N/A` in the
readable reports. Untrusted formula-like text is escaped in CSV exports.

## Table 4.15: original-model consistency needs actual repeats

The runner cannot infer repeated explanations from one explanation per sample.
Re-running the G-Eval judge assesses judge variability, not the stability of your
original explanation pipeline. The runner does not regenerate original
explanations because the uploaded reports do not contain all original prompts.

Run the **original explanation pipeline** three times on an appropriate,
predefined subset using identical evidence, prompt, generator model and settings.
The original result stays in `input_files/`. Put the second and third generator
results here:

```text
repeated_explanations/
  run_2/
    same_binary_analysis.json
  run_3/
    same_binary_analysis.json
```

The JSON must retain the same `sample.binary_name` and RF n-gram size (or the
corresponding analyzer metadata). Files are matched by binary/ngram identity,
not their filenames. Repeats for unknown primary samples are not included.
`CONSISTENCY_RUNS = 3` at the top controls the required total runs.

Run `python geval_batch.py` again after adding the repeats. Saved successful
judge results are reused, and the consistency table is recalculated locally.
No additional judge requests are needed for already scored unchanged inputs.

Definitions:

- **Samples re-run:** samples with all required valid generator runs and
  identical normalized supplied evidence. Partial groups, ambiguous duplicate
  repeat files and changed-evidence groups are excluded and documented.
- **Runs per sample:** total required runs for eligible complete groups, including
  the original. With no complete groups, this is `N/A`.
- **Tier agreement:** percentage of eligible samples whose original-model tier
  matches exactly in every required run. Denominator: eligible complete groups.
- **Agreement on top cited functions:** percentage with identical unordered
  cited-function name sets in every run. Denominator: eligible groups with a
  nonempty identifiable function list in every run. This is set agreement,
  not ranked-order agreement. The report records that operational definition.

For reliable function comparison, have the original generator save a structured
`analysis.cited_functions` array of function names. When absent, the runner uses
exact known function-name matching and Ghidra `FUN_...` tokens in candidate text;
this fallback should be checked manually and cannot recover every arbitrary
function name or citation rank. Empty/unknown function lists do not count as
100% agreement. Your uploaded example contains no cited function names, so
function agreement cannot be calculated from that artifact.

Without repeat data, the table records zero complete repeat samples and `N/A`
for agreement. These values are not substituted with invented percentages.

## Hallucination and SHAP-faithfulness definitions

The hallucination table measures **explanation-level unsupported-claim rate**
relative to the supplied evidence: one or more unsupported factual assertions
makes an explanation positive for this measure. It does not claim those
assertions are false in the real world. Purely conditional follow-up actions
and accurate generic instruction definitions are not automatically hallucinations.

The judge's quoted findings are validated against candidate text/value locations.
The runner also audits structured SHAP magnitudes against the input using an
absolute display-rounding tolerance of 0.0000005. A deterministic magnitude
mismatch counts as an unsupported numerical claim and a faithfulness discrepancy.
Judge-identified SHAP prose/sign discrepancies also count. General unsupported
behavioral claims can coexist with faithfully copied SHAP values.

Faithfulness is true only when both the numeric audit and judge's semantic
sign/magnitude assessment are faithful. The program validates score ranges,
quote locations and internal flag/list consistency; it cannot certify the
judge's subjective accuracy or all evidence references. Check a representative
subset with human reviewers using the same rubric.

The denominator for Tables 4.13, 4.14 and 4.16 is valid completed judge evaluations,
not all files discovered. API failures, invalid judge forms, invalid inputs,
generation failures, duplicates and unattempted/interrupted files remain visible
in the manifest and outcome counts. No failed evaluation receives score zero or
is labeled benign. Report these counts alongside your thesis tables.

The optional claim-level hallucination rate is **not** produced: the package
does not enumerate every atomic claim, so it has no valid total-claim denominator.
Quality means and SD are descriptive summaries of ordinal 1–5 ratings. SD is
sample SD using n−1 and is unavailable for fewer than two ratings. No percentage
conversion or unvalidated pass/fail threshold is used.

## Important configuration and evidence assumptions

`SHAP_TARGET_CLASS = 1` assumes your upstream SHAP values explain cryptojacking.
Confirm that extraction contract before the final evaluation. Change it to 0
only if your explanations genuinely target class 0, using a separate protocol.
The class-1 RF probability remains P(cryptojacking) regardless of predicted class.

The uploaded example contains only top-k features, with no SHAP baseline, full
attribution vector, assembly or occurrence mappings. Its feature values are
fractional and their normalization is not documented. The rubric disallows
assuming raw occurrence counts, inferring all-feature signs from the top-k
subset or treating generic SIMD mnemonics as verified mining behavior. If the
original generator received evidence absent from the saved artifact, recover
that exact evidence before claiming full input-grounded faithfulness. The
runner assesses support relative to the artifact it receives.

Actual labels, class-revealing source paths, original binary name and generator
identity are omitted from the judge input and retained only in private result
metadata. Candidate wording and the RF prediction remain unchanged because they
are part of what is being evaluated. Generator self-identification embedded in
candidate prose cannot be removed without modifying the target explanation.

This judge uses Sonnet 5.5 as requested. If that is also your explanation model,
automated evaluation may share model/family biases; blind the generator metadata,
keep the judge fixed across comparisons and validate against human ratings.
These scores do not replace detection metrics or external dataset validation.

## Errors, retries and resuming

- Rate-limit and transient network/server failures retry up to `MAX_ATTEMPTS`.
  Malformed final score forms also retry with validation feedback.
- Refusals, truncation/context-limit responses and other incomplete turns are
  saved as errors without pretending that thinking is final output. They are
  not automatically retried at an unchanged output budget.
- HTTP 400/401/403/404 stops further provider calls for the batch. Three
  consecutive unsuccessful samples also stops calls. Later valid files are
  recorded as unattempted; fix the cause and run again.
- Successes with identical judge input and protocol hashes are reused. Corrupt
  cached results and failed results are archived and re-evaluated.
- Changing model/rubric/settings requires a new `--output` folder. This prevents
  combining scores from different protocols or silently paying to replace a
  completed run. New output folders cause fresh evaluations.
- Saved usage covers attempts in the current active results; historical failed
  or replaced results are preserved in `archived_results/` and are not added to
  current table denominators. Usage not returned by a provider is unknown, not
  assumed to be zero. No monetary cost estimate is computed.

Inspect a saved result's `error` and `attempts` if a file fails. A missing model
or unavailable account access is an API error; the runner does not silently
substitute another model. If output is truncated, revise budget/effort during
pilot and use a new output folder for the resulting protocol. The code contains
no SDK `Messages.create()` calls, so old SDK keyword-signature mismatches do not
apply to this runner.

## Verification performed

The package has 24 offline/mock regression tests, including a 261-file batch
and a subsequent resume that issues no requests; rounding, sample SD, blinding,
input failures, retries, interruption, quote validation, secret redaction,
repeat comparisons and corrupt-cache recovery. Tests use synthetic judge scores
only inside temporary directories. They are not thesis results.

To run them locally without any API requests or key:

```powershell
python -m unittest discover -p test_geval_batch.py -v
```

A real provider call has not been tested here because no key was supplied.
Local tests and official documentation checks do not guarantee all live account,
service, model or judge-quality conditions. Perform the small live pilot first.

## Documentation checked on 2026-10-06

- Sonnet 5.5 model ID and migration settings:
  https://platform.claude.com/docs/en/models/sonnet-5-5/migration-guide
- Messages endpoint and authentication headers:
  https://platform.claude.com/docs/en/api/messages/create
- Structured output format and supported schema constraints:
  https://platform.claude.com/docs/en/build-with-claude/structured-outputs
- Original G-Eval method:
  https://aclanthology.org/2023.emnlp-main.153/

For Sonnet 5.5 the runner sends adaptive thinking with summarized display,
`output_config.effort` and `output_config.format`. It omits manual thinking budgets
and sampling parameters. Turning THINKING off uses `between_tools`, not the
unsupported `disabled` setting. All content blocks are read by type. Numeric
range constraints are enforced locally rather than sending unsupported raw
schema `minimum`/`maximum` keywords.

# Replication Artifact

Replication artifact for a mixed-methods study of AI use and verification in
vulnerability detection, explanation, and repair.

## Repository structure

```text
inputs/                                  De-identified inputs to the analysis
  analysis_dataset_deidentified.csv        Survey responses (152 retained respondents)
  thematic_coding_final_artifact.xlsx      Open-ended responses with their codes and themes,
                                           codebook, code frequencies, and co-occurrences
  Final_Reliability_coding.xlsx            Both coders' independent coding of the reliability
                                           sample, binary decisions, code-level statistics
  calibration_codes_side_by_side.xlsx      Calibration codes from both coders
scripts/
  run_analysis_artifact.py                 Reproduces all analyses and figures
  make_deidentified_artifact.py            Documents how the public inputs were produced
outputs/                                 All statistical outputs and figures
  revision/                                Analyses R01-R25 (see table below)
survey/
  Survey.pdf                               Survey instrument (Appendix A of the paper)
requirements.txt                         Python dependencies
```

## Reproducing the results

```bash
pip install -r requirements.txt
python scripts/run_analysis_artifact.py --inputs inputs --outputs outputs
```

The script regenerates every file in `outputs/`.

## Where each result appears in the paper

| Paper section | Output files |
|---|---|
| Table 1, Section 3.2 (sample) | `revision/R01`, `revision/R02`, `descriptive_tables.xlsx` |
| RQ1 (tool use and allocation) | `revision/R10`, `R11`, `R17`, `R18`, `R19`, `better_tool_by_task.png` |
| RQ2 (performance and risk) | `revision/R03`, `wilcoxon_llm_vs_coding_tools.csv`, `risk_distribution_scale_order.png` |
| RQ3 (verification) | `role_verification_method_comparisons.csv`, `revision/R13`, `verification_methods.png` |
| RQ4 (failures) | `revision/R12`, `R21`, `R22`, `R23` |
| RQ5 primary analysis | `revision/R04`-`R08`, `future_use_by_risk_and_task.png` |
| RQ5 underlying factors | `revision/R14`, `R15`, `R16`, `R20` |
| RQ5 secondary regression | `revision/R09`, `regression_future_intent_standardized_OLS.csv` |
| Section 3.6 (intercoder reliability) | `revision/R24`, `R24b`, `R25` |

## Privacy

Email addresses, IP addresses, location, Prolific participant, study, and
session IDs, Qualtrics response IDs, and timestamps were removed before release.
The qualitative files include participants' open-ended responses, alongside
their code assignments and themes, so that the coding can be checked directly.
Identifying details inside responses, such as an employer's name, were replaced
with placeholders (e.g., "[employer]"). In the reliability workbook, coder
memos were optional and were recorded by Coder A.
Each respondent keeps the anonymous ID (P001-P153) used across all files; one
consent-only record (P148) was excluded, leaving 152 respondents.

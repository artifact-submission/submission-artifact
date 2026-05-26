#!/usr/bin/env python3
"""
Standalone artifact analysis script.

Run:
    python run_analysis_artifact.py

Inputs expected in the same folder as this script:
    cleaned_responses.xlsx                 # participant-level data
    analysis_dataset_deidentified.csv      # de-identified participant-level data
    failure_mode_frequencies.csv           # aggregate thematic counts
    general_thematic_frequencies.csv       # aggregate thematic counts

All generated files are written to:
    artifact_outputs/
"""

from __future__ import annotations

from pathlib import Path
import textwrap
import warnings

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt

try:
    import statsmodels.api as sm
    HAS_STATSMODELS = True
except Exception:
    HAS_STATSMODELS = False


# ------------------------- GLOBAL SETTINGS -------------------------

OUT = Path("artifact_outputs")
OUT.mkdir(exist_ok=True)

COLUMN_W = 3.45
DOUBLE_W = 7.1

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 9.0,
    "axes.titlesize": 9.0,
    "axes.labelsize": 9.0,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
    "figure.titlesize": 9.0,
    "axes.linewidth": 0.7,
    "xtick.major.width": 0.7,
    "ytick.major.width": 0.7,
    "lines.linewidth": 1.0,
    "patch.linewidth": 0.7,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "axes.unicode_minus": False,
})


# ------------------------- HELPERS -------------------------

def save_readable(fig, path_without_ext):
    path_without_ext = Path(path_without_ext)
    fig.savefig(
        path_without_ext.with_suffix(".png"),
        dpi=600,
        bbox_inches="tight",
        pad_inches=0.03,
    )
    plt.close(fig)


def wrap_label(label, width=22):
    label = str(label)
    replacements = {
        "AI-based coding tools": "AI coding tools",
        "General-purpose LLMs": "General-purpose\nLLMs",
        "Depends on the situation": "Depends on\nsituation",
        "About the same": "About the\nsame",
        "I generally do not verify": "No verification",
        "Use static/dynamic analysis tools": "Static/dynamic\nanalysis",
        "Compare multiple AI answers": "Compare AI\nanswers",
        "Consult documentation": "Consult docs",
        "Manual code review": "Manual code\nreview",
        "Run tests": "Run tests",
    }
    label = replacements.get(label, label)
    if "\n" in label:
        return label
    return "\n".join(textwrap.wrap(label, width=width))


def looks_like_metadata(row):
    text = " ".join(row.dropna().astype(str).head(30))
    return ("ImportId" in text) or ("Source file name" in text) or ("Start Date" in text)


def load_dataset():
    if Path("analysis_dataset_deidentified.csv").exists():
        input_file = Path("analysis_dataset_deidentified.csv")
        df = pd.read_csv(input_file)
    elif Path("cleaned_responses.xlsx").exists():
        input_file = Path("cleaned_responses.xlsx")
        raw = pd.read_excel(input_file)
        df = raw.loc[~raw.apply(looks_like_metadata, axis=1)].copy().reset_index(drop=True)
    else:
        print("No participant-level dataset found. Skipping survey analyses.")
        return None, None

    print(f"Loaded {len(df)} rows from {input_file}")
    return df, input_file


def get_column_map(df):
    col = {
        "role": "Background",
        "security_frequency": "Q89",
        "experience": "QID1219536837",
        "languages": "QID1219536841",
        "traditional_tools_used": "QID1219536853",
        "used_llm": "Usage Screening",
        "llm_tools": "QID1219536865",
        "used_coding_tool": "QID139",
        "coding_tools": "QID1219536827",
        "contexts": "Tool Usage",
        "verification": "QID160",
        "explanation_challenges": "QID150",

        "use_llm_detection": "QID142_1",
        "use_llm_explanation": "QID142_2",
        "use_llm_repair": "QID142_3",
        "use_coding_detection": "QID142_4",
        "use_coding_explanation": "QID142_5",
        "use_coding_repair": "QID142_6",

        "acc_llm_detection": "QID143_1",
        "acc_llm_explanation": "QID143_2",
        "acc_llm_repair": "QID143_3",
        "acc_coding_detection": "QID143_4",
        "acc_coding_explanation": "QID143_5",
        "acc_coding_repair": "QID143_6",

        "eff_llm_detection": "QID144_1",
        "eff_llm_explanation": "QID144_2",
        "eff_llm_repair": "QID144_3",
        "eff_coding_detection": "QID144_4",
        "eff_coding_explanation": "QID144_5",
        "eff_coding_repair": "QID144_6",

        "severity": "QID145",
        "risk": "QID163",
        "fix_compiles": "QID146_1",
        "fix_resolves": "QID146_2",
        "fix_new_bug": "QID146_3",
        "fix_new_vuln": "QID146_4",
        "fix_unintended": "QID146_5",

        "explanation_source": "QID147",
        "confidence": "QID148",
        "clear_llm": "QID164_1",
        "clear_coding": "QID164_2",
        "helpful_llm": "QID149_1",
        "helpful_coding": "QID149_2",

        "future_detection": "Future Intent_1",
        "future_explanation": "Future Intent_2",
        "future_repair": "Future Intent_3",

        "better_detection": "QID154_1",
        "better_repair": "QID154_2",
        "better_explanation": "QID154_3",

        "trad_use_detection": "QID151_1",
        "trad_use_explanation": "QID151_2",
        "trad_use_repair": "QID151_3",
        "prefer_trad_detection": "QID152_1",
        "prefer_trad_explanation": "QID152_2",
        "prefer_trad_repair": "QID152_3",
        "trad_combo_detection": "QID153_1",
        "trad_combo_explanation": "QID153_2",
        "trad_combo_repair": "QID153_3",

        "misleading_case": "QID155",
        "final_comment": "QID158",
    }
    return {k: v for k, v in col.items() if v in df.columns}


ORDER = {
    "frequency5": {"Never": 1, "Rarely": 2, "Sometimes": 3, "Often": 4, "Always": 5},
    "frequency_helpful": {"Never": 1, "Rarely": 2, "Sometimes": 3, "Frequently": 4, "Always": 5},
    "accuracy": {
        "Not accurate at all": 1,
        "Slightly accurate": 2,
        "Moderately accurate": 3,
        "Very accurate": 4,
        "Extremely accurate": 5,
    },
    "effectiveness": {
        "Not effective at all": 1,
        "Slightly effective": 2,
        "Moderately effective": 3,
        "Very effective": 4,
        "Extremely effective": 5,
    },
    "risk": {"Not risky": 1, "Slightly": 2, "Moderately": 3, "Very": 4, "Extremely": 5},
    "future": {
        "Extremely unlikely": 1,
        "Somewhat unlikely": 2,
        "Neither likely nor unlikely": 3,
        "Somewhat likely": 4,
        "Extremely likely": 5,
    },
    "confidence": {"Not confident": 1, "Slightly": 2, "Moderately": 3, "Very": 4, "Extremely": 5},
    "clarity": {
        "Not clear at all": 1,
        "Slightly clear": 2,
        "Moderately clear": 3,
        "Very clear": 4,
        "Extremely clear": 5,
    },
    "prefer_trad": {
        "Never": 1,
        "Rarely": 2,
        "Sometimes": 3,
        "About half the time": 4,
        "Most of the time": 5,
        "Always": 6,
    },
    "security_frequency": {"Never": 1, "Rarely": 2, "Sometimes": 3, "Often": 4, "Very often": 5},
    "experience": {"Less than one": 1, "One to two": 2, "Three to five": 3, "Six to ten": 4, "Eleven or more": 5},
}


GROUPS = {
    "usage_frequency": (
        "frequency5",
        ["use_llm_detection", "use_llm_explanation", "use_llm_repair",
         "use_coding_detection", "use_coding_explanation", "use_coding_repair"],
    ),
    "accuracy": (
        "accuracy",
        ["acc_llm_detection", "acc_llm_explanation", "acc_llm_repair",
         "acc_coding_detection", "acc_coding_explanation", "acc_coding_repair"],
    ),
    "effectiveness": (
        "effectiveness",
        ["eff_llm_detection", "eff_llm_explanation", "eff_llm_repair",
         "eff_coding_detection", "eff_coding_explanation", "eff_coding_repair"],
    ),
    "repair_outcomes": (
        "frequency5",
        ["fix_compiles", "fix_resolves", "fix_new_bug", "fix_new_vuln", "fix_unintended"],
    ),
    "helpfulness": ("frequency_helpful", ["helpful_llm", "helpful_coding"]),
    "future_intent": ("future", ["future_detection", "future_explanation", "future_repair"]),
    "traditional_use": ("frequency5", ["trad_use_detection", "trad_use_explanation", "trad_use_repair"]),
    "prefer_traditional": ("prefer_trad", ["prefer_trad_detection", "prefer_trad_explanation", "prefer_trad_repair"]),
}


MULTI_OPTIONS = {
    "verification": [
        "Manual code review",
        "Run tests",
        "Use static/dynamic analysis tools",
        "Compare multiple AI answers",
        "Consult documentation",
        "I generally do not verify",
    ],
    "contexts": [
        "While coding in the IDE (inline support, autocomplete, inline analysis)",
        "During code review (pull requests, merge checks)",
        "As part of CI/CD pipelines (automated scans, build checks)",
        "During testing or fuzzing (runtime/dynamic checks)",
        "For security learning or documentation (understanding vulnerabilities, training)",
        "None of the above",
    ],
    "explanation_challenges": [
        "Explanations were too vague or generic",
        "Explanations were too technical or jargon-heavy",
        "Explanations lacked sufficient context about the codebase",
        "Explanations overstated or understated the severity",
        "Explanations were incorrect or misleading",
        "Explanations recommended insecure patterns",
    ],
}


def add_numeric_columns(df, col):
    def code(key, scale):
        if key in col:
            df[key + "_num"] = df[col[key]].map(ORDER[scale])

    for _, (scale_name, keys) in GROUPS.items():
        for key in keys:
            code(key, scale_name)

    for key, scale_name in [
        ("risk", "risk"),
        ("confidence", "confidence"),
        ("clear_llm", "clarity"),
        ("clear_coding", "clarity"),
        ("security_frequency", "security_frequency"),
        ("experience", "experience"),
        ("helpful_llm", "frequency_helpful"),
        ("helpful_coding", "frequency_helpful"),
    ]:
        code(key, scale_name)


def pct_table(series):
    s = series.dropna().astype(str)
    counts = s.value_counts()
    out = pd.DataFrame({"n": counts, "percent": (counts / len(s) * 100).round(1)})
    out.index.name = series.name
    return out.reset_index()


def likert_summary(df, keys):
    rows = []
    for key in keys:
        colname = key + "_num"
        if colname not in df:
            continue
        x = df[colname].dropna()
        rows.append({
            "variable": key,
            "n": int(x.count()),
            "median": float(x.median()),
            "IQR": float(x.quantile(.75) - x.quantile(.25)),
            "mean": round(float(x.mean()), 2),
            "sd": round(float(x.std()), 2),
        })
    return pd.DataFrame(rows)


def split_multiselect(series):
    items = []
    for value in series.dropna().astype(str):
        if value.startswith('{"ImportId"'):
            continue
        for item in value.split(","):
            item = item.strip()
            if item:
                items.append(item)
    counts = pd.Series(items).value_counts()
    return pd.DataFrame({
        "item": counts.index,
        "n": counts.values,
        "percent_of_respondents": (counts.values / len(series) * 100).round(1),
    })


def selected_options(df, col, key):
    if key not in col:
        return pd.DataFrame(index=df.index)
    values = df[col[key]].fillna("").astype(str)
    options = MULTI_OPTIONS.get(key, [])
    out = pd.DataFrame(index=df.index)
    for opt in options:
        out[opt] = values.str.contains(opt, regex=False)
    return out


def row_count_multiselect(df, col, key, exclude_terms=None):
    if key not in col:
        return pd.Series(np.nan, index=df.index)
    exclude_terms = exclude_terms or []
    opts = selected_options(df, col, key)
    if not opts.empty:
        keep = [c for c in opts.columns if not any(term.lower() in c.lower() for term in exclude_terms)]
        return opts[keep].sum(axis=1)
    counts = []
    for value in df[col[key]].fillna("").astype(str):
        items = [x.strip() for x in value.split(",") if x.strip()]
        items = [x for x in items if not any(term.lower() in x.lower() for term in exclude_terms)]
        counts.append(len(items))
    return pd.Series(counts, index=df.index)


def available_num(df, keys):
    return [k + "_num" for k in keys if k + "_num" in df.columns]


def row_mean(df, keys):
    cols = available_num(df, keys)
    if not cols:
        return pd.Series(np.nan, index=df.index)
    return df[cols].mean(axis=1, skipna=True)


def cronbach_alpha(df, cols):
    data = df[cols].dropna()
    k = data.shape[1]
    if k < 2 or len(data) < 5:
        return np.nan, len(data)
    item_vars = data.var(axis=0, ddof=1)
    total_var = data.sum(axis=1).var(ddof=1)
    if total_var == 0:
        return np.nan, len(data)
    alpha = (k / (k - 1)) * (1 - item_vars.sum() / total_var)
    return float(alpha), len(data)


def spearman_pair(df, x, y):
    pair = df[[x, y]].dropna()
    if len(pair) < 5 or pair[x].nunique() < 2 or pair[y].nunique() < 2:
        return {"x": x, "y": y, "n": len(pair), "rho": np.nan, "p": np.nan}
    rho, p = stats.spearmanr(pair[x], pair[y])
    return {"x": x, "y": y, "n": len(pair), "rho": rho, "p": p}


def spearman_test(df, xkey, ykey):
    x = df[xkey + "_num"]
    y = df[ykey + "_num"]
    pair = pd.concat([x, y], axis=1).dropna()
    if len(pair) < 5:
        return {"x": xkey, "y": ykey, "n": len(pair), "spearman_rho": np.nan, "p_value": np.nan}
    rho, p = stats.spearmanr(pair.iloc[:, 0], pair.iloc[:, 1])
    return {"x": xkey, "y": ykey, "n": len(pair), "spearman_rho": rho, "p_value": p}


def wilcoxon_pair(df, key_a, key_b, label_a, label_b):
    if key_a + "_num" not in df.columns or key_b + "_num" not in df.columns:
        return None
    a = df[key_a + "_num"]
    b = df[key_b + "_num"]
    pair = pd.concat([a, b], axis=1).dropna()
    if len(pair) < 5:
        return None
    diff = pair.iloc[:, 1] - pair.iloc[:, 0]
    nonzero = diff[diff != 0]
    if len(nonzero) == 0:
        stat, p = np.nan, 1.0
        effect_r = 0.0
    else:
        stat, p = stats.wilcoxon(pair.iloc[:, 0], pair.iloc[:, 1], zero_method="wilcox", alternative="two-sided")
        n = len(nonzero)
        mean_w = n * (n + 1) / 4
        sd_w = np.sqrt(n * (n + 1) * (2 * n + 1) / 24)
        z = (stat - mean_w) / sd_w if sd_w else np.nan
        effect_r = abs(z) / np.sqrt(n) if n else np.nan
    return {
        "comparison": f"{label_b} minus {label_a}",
        "n_pairs": len(pair),
        f"median_{label_a}": pair.iloc[:, 0].median(),
        f"median_{label_b}": pair.iloc[:, 1].median(),
        "mean_difference": round((pair.iloc[:, 1] - pair.iloc[:, 0]).mean(), 3),
        "wilcoxon_statistic": stat,
        "p_value": p,
        "effect_r_approx": effect_r,
    }


def mann_whitney_by_binary(df, group_col, outcome_col, positive_label):
    sub = df[[group_col, outcome_col]].dropna()
    a = sub[sub[group_col] == positive_label][outcome_col]
    b = sub[sub[group_col] != positive_label][outcome_col]
    if len(a) < 5 or len(b) < 5:
        return None
    U, p = stats.mannwhitneyu(a, b, alternative="two-sided")
    return {
        "group": group_col,
        "positive_label": positive_label,
        "outcome": outcome_col,
        "n_positive": len(a),
        "n_other": len(b),
        "median_positive": a.median(),
        "median_other": b.median(),
        "U": U,
        "p": p,
    }


def add_composite_scores(df, col):
    df["llm_usage_score"] = row_mean(df, ["use_llm_detection", "use_llm_explanation", "use_llm_repair"])
    df["coding_usage_score"] = row_mean(df, ["use_coding_detection", "use_coding_explanation", "use_coding_repair"])
    df["overall_ai_usage_score"] = row_mean(df, ["use_llm_detection", "use_llm_explanation", "use_llm_repair", "use_coding_detection", "use_coding_explanation", "use_coding_repair"])

    df["llm_accuracy_score"] = row_mean(df, ["acc_llm_detection", "acc_llm_explanation", "acc_llm_repair"])
    df["coding_accuracy_score"] = row_mean(df, ["acc_coding_detection", "acc_coding_explanation", "acc_coding_repair"])
    df["llm_effectiveness_score"] = row_mean(df, ["eff_llm_detection", "eff_llm_explanation", "eff_llm_repair"])
    df["coding_effectiveness_score"] = row_mean(df, ["eff_coding_detection", "eff_coding_explanation", "eff_coding_repair"])

    df["llm_performance_score"] = row_mean(df, ["acc_llm_detection", "acc_llm_explanation", "acc_llm_repair", "eff_llm_detection", "eff_llm_explanation", "eff_llm_repair"])
    df["coding_performance_score"] = row_mean(df, ["acc_coding_detection", "acc_coding_explanation", "acc_coding_repair", "eff_coding_detection", "eff_coding_explanation", "eff_coding_repair"])
    df["overall_performance_score"] = row_mean(df, ["acc_llm_detection", "acc_llm_explanation", "acc_llm_repair", "acc_coding_detection", "acc_coding_explanation", "acc_coding_repair", "eff_llm_detection", "eff_llm_explanation", "eff_llm_repair", "eff_coding_detection", "eff_coding_explanation", "eff_coding_repair"])

    df["future_intent_score"] = row_mean(df, ["future_detection", "future_explanation", "future_repair"])
    df["traditional_use_score"] = row_mean(df, ["trad_use_detection", "trad_use_explanation", "trad_use_repair"])
    df["traditional_preference_score"] = row_mean(df, ["prefer_trad_detection", "prefer_trad_explanation", "prefer_trad_repair"])
    df["positive_fix_outcome_score"] = row_mean(df, ["fix_compiles", "fix_resolves"])
    df["negative_fix_outcome_score"] = row_mean(df, ["fix_new_bug", "fix_new_vuln", "fix_unintended"])
    df["explanation_quality_score"] = row_mean(df, ["clear_llm", "clear_coding", "helpful_llm", "helpful_coding"])

    df["verification_breadth_score"] = row_count_multiselect(df, col, "verification", exclude_terms=["do not verify"])
    df["sdlc_breadth_score"] = row_count_multiselect(df, col, "contexts", exclude_terms=["none of the above"])
    df["challenge_breadth_score"] = row_count_multiselect(df, col, "explanation_challenges")

    if "risk_num" in df.columns:
        df["risk_score"] = df["risk_num"]
    if "experience_num" in df.columns:
        df["experience_score"] = df["experience_num"]
    if "security_frequency_num" in df.columns:
        df["security_task_frequency_score"] = df["security_frequency_num"]


# ------------------------- MAIN SURVEY ANALYSIS -------------------------

def run_main_analysis(df, col):
    cat_keys = [
        "role", "security_frequency", "experience", "used_llm", "used_coding_tool",
        "severity", "risk", "better_detection", "better_explanation", "better_repair",
    ]

    with pd.ExcelWriter(OUT / "descriptive_tables.xlsx") as writer:
        pd.DataFrame({"N": [len(df)]}).to_excel(writer, sheet_name="sample_size", index=False)
        for key in cat_keys:
            if key in col:
                pct_table(df[col[key]]).to_excel(writer, sheet_name=key[:31], index=False)
        for name, (_, keys) in GROUPS.items():
            likert_summary(df, keys).to_excel(writer, sheet_name=(name + "_summary")[:31], index=False)
        for key in [
            "languages", "traditional_tools_used", "llm_tools", "coding_tools", "contexts",
            "verification", "explanation_challenges", "trad_combo_detection",
            "trad_combo_explanation", "trad_combo_repair",
        ]:
            if key in col:
                split_multiselect(df[col[key]]).to_excel(writer, sheet_name=(key + "_multi")[:31], index=False)

    comparisons = []
    for task in ["detection", "explanation", "repair"]:
        comparisons.append(wilcoxon_pair(df, f"acc_llm_{task}", f"acc_coding_{task}", "LLM", "CodingTool"))
        comparisons.append(wilcoxon_pair(df, f"eff_llm_{task}", f"eff_coding_{task}", "LLM", "CodingTool"))
    for a, b in [("helpful_llm", "helpful_coding"), ("clear_llm", "clear_coding")]:
        comparisons.append(wilcoxon_pair(df, a, b, "LLM", "CodingTool"))

    comparisons = pd.DataFrame([x for x in comparisons if x is not None])
    if not comparisons.empty:
        pvals = comparisons["p_value"].to_numpy(dtype=float)
        order = np.argsort(pvals)
        adjusted = np.empty_like(pvals)
        m = len(pvals)
        running_max = 0
        for rank, idx in enumerate(order):
            adj = (m - rank) * pvals[idx]
            running_max = max(running_max, adj)
            adjusted[idx] = min(running_max, 1.0)
        comparisons["p_holm"] = adjusted
    comparisons.to_csv(OUT / "wilcoxon_llm_vs_coding_tools.csv", index=False)

    corrs = []
    if "risk_num" in df:
        for y in [
            "use_llm_detection", "use_llm_explanation", "use_llm_repair",
            "use_coding_detection", "use_coding_explanation", "use_coding_repair",
            "future_detection", "future_explanation", "future_repair",
            "prefer_trad_detection", "prefer_trad_explanation", "prefer_trad_repair",
        ]:
            if y + "_num" in df:
                corrs.append(spearman_test(df, "risk", y))
    pd.DataFrame(corrs).to_csv(OUT / "spearman_risk_correlations.csv", index=False)

    kw_rows = []
    if "experience_num" in df:
        for y in ["risk", "confidence", "acc_llm_detection", "acc_coding_detection", "eff_llm_repair", "eff_coding_repair"]:
            if y + "_num" in df:
                groups = [g[y + "_num"].dropna().values for _, g in df.groupby("experience_num") if len(g[y + "_num"].dropna()) > 0]
                if len(groups) >= 2:
                    H, p = stats.kruskal(*groups)
                    kw_rows.append({"factor": "experience", "outcome": y, "H": H, "p_value": p})
    pd.DataFrame(kw_rows).to_csv(OUT / "kruskal_experience_tests.csv", index=False)

    chi_rows = []
    for xkey in ["role", "experience"]:
        if xkey in col:
            for ykey in ["better_detection", "better_explanation", "better_repair"]:
                if ykey in col:
                    table = pd.crosstab(df[col[xkey]], df[col[ykey]])
                    if table.shape[0] >= 2 and table.shape[1] >= 2:
                        chi2, p, dof, expected = stats.chi2_contingency(table)
                        chi_rows.append({
                            "x": xkey,
                            "y": ykey,
                            "chi2": chi2,
                            "dof": dof,
                            "p_value": p,
                            "min_expected_cell": expected.min(),
                        })
                        table.to_csv(OUT / f"crosstab_{xkey}_by_{ykey}.csv")
    pd.DataFrame(chi_rows).to_csv(OUT / "chi_square_tests.csv", index=False)

    if "risk" in col:
        save_bar_from_counts(df[col["risk"]], "risk_distribution")
    if "better_detection" in col:
        save_bar_from_counts(df[col["better_detection"]], "better_detection")

    if "verification" in col:
        ver = split_multiselect(df[col["verification"]]).head(10)
        fig_h = max(1.8, 0.36 * len(ver) + 0.45)
        fig, ax = plt.subplots(figsize=(COLUMN_W, fig_h))
        labels = [wrap_label(x, 20) for x in ver["item"]]
        bars = ax.barh(labels, ver["percent_of_respondents"], height=0.62)
        ax.set_xlabel("Percent of respondents")
        ax.invert_yaxis()
        ax.bar_label(bars, labels=[f"{v:.0f}%" for v in ver["percent_of_respondents"]], padding=2, fontsize=7.0)
        ax.set_xlim(0, max(ver["percent_of_respondents"].max() * 1.16, 10))
        ax.grid(axis="x", linewidth=0.35, alpha=0.35)
        ax.set_axisbelow(True)
        fig.tight_layout(pad=0.3)
        save_readable(fig, OUT / "verification_methods")

    make_boxplots(df)

    # Raw free-text responses are not exported because they may contain sensitive
    # participant, organizational, or security details.


def save_bar_from_counts(series, filename, top_n=12):
    tab = pct_table(series).head(top_n)
    fig_h = max(1.6, 0.34 * len(tab) + 0.45)
    fig, ax = plt.subplots(figsize=(COLUMN_W, fig_h))
    labels = [wrap_label(x, 20) for x in tab.iloc[:, 0].astype(str)]
    bars = ax.barh(labels, tab["percent"], height=0.62)
    ax.set_xlabel("Percent of respondents")
    ax.invert_yaxis()
    ax.bar_label(bars, labels=[f"{v:.0f}%" for v in tab["percent"]], padding=2, fontsize=7.0)
    ax.set_xlim(0, max(tab["percent"].max() * 1.18, 10))
    ax.grid(axis="x", linewidth=0.35, alpha=0.35)
    ax.set_axisbelow(True)
    fig.tight_layout(pad=0.3)
    save_readable(fig, OUT / filename)


def make_boxplots(df):
    plot_rows = []
    for construct in ["acc", "eff"]:
        for task in ["detection", "explanation", "repair"]:
            for tool in ["llm", "coding"]:
                key = f"{construct}_{tool}_{task}"
                if key + "_num" in df:
                    for v in df[key + "_num"].dropna():
                        plot_rows.append({
                            "construct": construct,
                            "task": task,
                            "tool": "LLM" if tool == "llm" else "Coding tool",
                            "score": v,
                        })

    plot_df = pd.DataFrame(plot_rows)
    for construct in ["acc", "eff"]:
        sub = plot_df[plot_df["construct"] == construct]
        if sub.empty:
            continue
        labels, data = [], []
        short_task = {"detection": "Detect.", "explanation": "Explain", "repair": "Repair"}
        for task in ["detection", "explanation", "repair"]:
            for tool in ["LLM", "Coding tool"]:
                labels.append(f"{short_task[task]}\n{tool}")
                data.append(sub[(sub["task"] == task) & (sub["tool"] == tool)]["score"].values)
        fig, ax = plt.subplots(figsize=(COLUMN_W, 2.15))
        ax.boxplot(
            data,
            tick_labels=labels,
            widths=0.55,
            medianprops={"linewidth": 1.1},
            boxprops={"linewidth": 0.8},
            whiskerprops={"linewidth": 0.8},
            capprops={"linewidth": 0.8},
            flierprops={"markersize": 2.8},
        )
        ax.set_ylabel("Likert score (1--5)")
        ax.set_ylim(0.75, 5.25)
        ax.set_yticks([1, 2, 3, 4, 5])
        ax.tick_params(axis="x", pad=1)
        ax.grid(axis="y", linewidth=0.35, alpha=0.35)
        ax.set_axisbelow(True)
        fig.tight_layout(pad=0.3)
        save_readable(fig, OUT / f"{construct}_llm_vs_coding_boxplot")


# ------------------------- DEEPER ANALYSIS -------------------------

def run_deeper_analysis(df, col):
    construct_items = {
        "llm_usage_score": available_num(df, ["use_llm_detection", "use_llm_explanation", "use_llm_repair"]),
        "coding_usage_score": available_num(df, ["use_coding_detection", "use_coding_explanation", "use_coding_repair"]),
        "llm_performance_score": available_num(df, ["acc_llm_detection", "acc_llm_explanation", "acc_llm_repair", "eff_llm_detection", "eff_llm_explanation", "eff_llm_repair"]),
        "coding_performance_score": available_num(df, ["acc_coding_detection", "acc_coding_explanation", "acc_coding_repair", "eff_coding_detection", "eff_coding_explanation", "eff_coding_repair"]),
        "future_intent_score": available_num(df, ["future_detection", "future_explanation", "future_repair"]),
        "traditional_preference_score": available_num(df, ["prefer_trad_detection", "prefer_trad_explanation", "prefer_trad_repair"]),
        "positive_fix_outcome_score": available_num(df, ["fix_compiles", "fix_resolves"]),
        "negative_fix_outcome_score": available_num(df, ["fix_new_bug", "fix_new_vuln", "fix_unintended"]),
    }

    alpha_rows = []
    for construct, cols in construct_items.items():
        alpha, n = cronbach_alpha(df, cols)
        alpha_rows.append({"construct": construct, "n_complete": n, "n_items": len(cols), "cronbach_alpha": alpha})
    pd.DataFrame(alpha_rows).to_csv(OUT / "composite_reliability_cronbach_alpha.csv", index=False)

    composites = [
        "llm_usage_score", "coding_usage_score", "overall_ai_usage_score",
        "llm_performance_score", "coding_performance_score", "overall_performance_score",
        "risk_score", "future_intent_score", "traditional_use_score", "traditional_preference_score",
        "positive_fix_outcome_score", "negative_fix_outcome_score", "explanation_quality_score",
        "verification_breadth_score", "sdlc_breadth_score", "challenge_breadth_score",
        "experience_score", "security_task_frequency_score",
    ]

    summary_rows = []
    for c in [x for x in composites if x in df.columns]:
        x = df[c].dropna()
        summary_rows.append({
            "score": c,
            "n": len(x),
            "median": x.median(),
            "IQR": x.quantile(.75) - x.quantile(.25),
            "mean": x.mean(),
            "sd": x.std(),
            "min": x.min(),
            "max": x.max(),
        })
    pd.DataFrame(summary_rows).to_csv(OUT / "composite_score_summary.csv", index=False)

    pairs = []
    for x in [
        "overall_performance_score", "llm_performance_score", "coding_performance_score",
        "risk_score", "verification_breadth_score", "negative_fix_outcome_score",
        "traditional_preference_score", "experience_score", "security_task_frequency_score",
    ]:
        for y in ["overall_ai_usage_score", "future_intent_score"]:
            if x in df.columns and y in df.columns:
                pairs.append(spearman_pair(df, x, y))

    for x in ["overall_performance_score", "risk_score", "negative_fix_outcome_score", "traditional_preference_score", "experience_score"]:
        for y in ["verification_breadth_score", "traditional_use_score"]:
            if x in df.columns and y in df.columns:
                pairs.append(spearman_pair(df, x, y))

    for x in ["negative_fix_outcome_score", "positive_fix_outcome_score", "explanation_quality_score", "challenge_breadth_score"]:
        for y in ["risk_score", "future_intent_score", "traditional_preference_score"]:
            if x in df.columns and y in df.columns:
                pairs.append(spearman_pair(df, x, y))

    corr_df = pd.DataFrame(pairs).drop_duplicates()
    corr_df.to_csv(OUT / "deep_spearman_composite_relationships.csv", index=False)

    if not corr_df.empty:
        corr_df = corr_df.copy()
        pvals = corr_df["p"].astype(float).to_numpy()
        order = np.argsort(np.nan_to_num(pvals, nan=1.0))
        adjusted = np.empty_like(pvals)
        m = len(pvals)
        running_max = 0
        for rank, idx in enumerate(order):
            adj = (m - rank) * (pvals[idx] if not np.isnan(pvals[idx]) else 1.0)
            running_max = max(running_max, adj)
            adjusted[idx] = min(running_max, 1.0)
        corr_df["p_holm"] = adjusted
        corr_df.to_csv(OUT / "deep_spearman_composite_relationships_holm.csv", index=False)

    run_regression(df)
    run_group_comparisons(df, col)
    make_personas(df)
    # Participant-level composite-score data are not exported by default because
    # IRB/consent restrictions may prohibit sharing row-level survey data.
    make_paper_heatmap(df)
    make_risk_scatter(df)
    make_persona_plot(df)


def run_regression(df):
    reg_predictors = [
        "overall_performance_score", "risk_score", "verification_breadth_score",
        "traditional_preference_score", "negative_fix_outcome_score",
        "experience_score", "security_task_frequency_score",
    ]
    reg_predictors = [c for c in reg_predictors if c in df.columns]
    reg_outcome = "future_intent_score"

    if HAS_STATSMODELS and reg_outcome in df.columns and len(reg_predictors) >= 2:
        reg = df[[reg_outcome] + reg_predictors].dropna()
        z = reg.copy()
        for c in z.columns:
            if z[c].std(ddof=0) > 0:
                z[c] = (z[c] - z[c].mean()) / z[c].std(ddof=0)
        X = sm.add_constant(z[reg_predictors])
        y = z[reg_outcome]
        model = sm.OLS(y, X).fit(cov_type="HC3")
        coef = pd.DataFrame({
            "term": model.params.index,
            "std_beta": model.params.values,
            "robust_se": model.bse.values,
            "t": model.tvalues.values,
            "p": model.pvalues.values,
        })
        coef.to_csv(OUT / "regression_future_intent_standardized_OLS.csv", index=False)
        with open(OUT / "regression_future_intent_model_summary.txt", "w", encoding="utf-8") as f:
            f.write(model.summary().as_text())
    else:
        with open(OUT / "regression_not_run.txt", "w", encoding="utf-8") as f:
            f.write("Install statsmodels to run regression: pip install statsmodels\n")


def run_group_comparisons(df, col):
    context_rows = []
    if "contexts" in col:
        ctx_df = selected_options(df, col, "contexts")
        for ctx in [c for c in ctx_df.columns if "None of the above" not in c]:
            indicator = "ctx__" + str(abs(hash(ctx)))
            df[indicator] = ctx_df[ctx]
            for outcome in [
                "overall_ai_usage_score", "overall_performance_score",
                "risk_score", "verification_breadth_score", "future_intent_score",
            ]:
                if outcome in df.columns:
                    res = mann_whitney_by_binary(df, indicator, outcome, True)
                    if res:
                        res["context"] = ctx
                        context_rows.append(res)
    pd.DataFrame(context_rows).to_csv(OUT / "sdlc_context_group_comparisons.csv", index=False)

    challenge_rows = []
    if "explanation_challenges" in col:
        ch_df = selected_options(df, col, "explanation_challenges")
        for ch in ch_df.columns:
            indicator = "challenge__" + str(abs(hash(ch)))
            df[indicator] = ch_df[ch]
            for outcome in ["risk_score", "overall_performance_score", "future_intent_score", "traditional_preference_score"]:
                if outcome in df.columns:
                    res = mann_whitney_by_binary(df, indicator, outcome, True)
                    if res:
                        res["challenge"] = ch
                        challenge_rows.append(res)
    pd.DataFrame(challenge_rows).to_csv(OUT / "explanation_challenge_group_comparisons.csv", index=False)


def make_personas(df):
    needed = ["overall_performance_score", "risk_score", "verification_breadth_score", "overall_ai_usage_score"]
    if not all(c in df.columns for c in needed):
        return
    med = df[needed].median(numeric_only=True)

    def persona(row):
        if pd.isna(row["overall_performance_score"]) or pd.isna(row["risk_score"]) or pd.isna(row["verification_breadth_score"]):
            return np.nan
        high_perf = row["overall_performance_score"] >= med["overall_performance_score"]
        high_risk = row["risk_score"] >= med["risk_score"]
        high_ver = row["verification_breadth_score"] >= med["verification_breadth_score"]
        high_use = row["overall_ai_usage_score"] >= med["overall_ai_usage_score"] if not pd.isna(row["overall_ai_usage_score"]) else False
        if high_perf and high_use and not high_risk:
            return "confident adopters"
        if high_perf and high_risk and high_ver:
            return "cautious power users"
        if not high_perf and high_risk and high_ver:
            return "skeptical verifiers"
        if not high_use and high_risk:
            return "risk-averse low users"
        return "mixed/other"

    df["persona"] = df.apply(persona, axis=1)
    persona_tab = df["persona"].value_counts(dropna=True).rename_axis("persona").reset_index(name="n")
    persona_tab["percent"] = persona_tab["n"] / persona_tab["n"].sum() * 100
    persona_tab.to_csv(OUT / "respondent_personas_rule_based.csv", index=False)


def make_paper_heatmap(df):
    heat_cols = [c for c in [
        "overall_ai_usage_score",
        "overall_performance_score",
        "risk_score",
        "future_intent_score",
        "verification_breadth_score",
        "traditional_preference_score",
        "negative_fix_outcome_score",
        "experience_score",
    ] if c in df.columns]

    if len(heat_cols) < 3:
        return

    corr = df[heat_cols].corr(method="spearman")
    corr.to_csv(OUT / "composite_correlation_matrix.csv")

    pretty = {
        "overall_ai_usage_score": "Overall\nAI usage",
        "overall_performance_score": "Perceived\nperformance",
        "risk_score": "Perceived\nrisk",
        "future_intent_score": "Future-use\nintention",
        "verification_breadth_score": "Verification\nbreadth",
        "traditional_preference_score": "Traditional-tool\npreference",
        "negative_fix_outcome_score": "Negative repair\noutcomes",
        "experience_score": "Experience",
    }
    labels = [pretty.get(c, c.replace("_score", "").replace("_", "\n")) for c in heat_cols]

    # This is the paper-style full-label heatmap, matching Figure 7.
    fig, ax = plt.subplots(figsize=(COLUMN_W, 3.20))
    im = ax.imshow(corr, vmin=-1, vmax=1, cmap="viridis", aspect="equal")

    ax.set_xticks(np.arange(len(heat_cols)))
    ax.set_yticks(np.arange(len(heat_cols)))
    ax.set_xticklabels(labels, rotation=45, ha="right", rotation_mode="anchor")
    ax.set_yticklabels(labels)

    ax.tick_params(axis="x", labelsize=6.4, pad=1)
    ax.tick_params(axis="y", labelsize=6.4, pad=1)

    for i in range(len(heat_cols)):
        for j in range(len(heat_cols)):
            ax.text(
                j,
                i,
                f"{corr.iloc[i, j]:.2f}",
                ha="center",
                va="center",
                fontsize=6.2,
                color="black",
            )

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.035)
    cbar.ax.tick_params(labelsize=6.3)

    fig.tight_layout(pad=0.15)
    fig.savefig(
        OUT / "composite_correlation_heatmap.png",
        dpi=600,
        bbox_inches="tight",
        pad_inches=0.02,
    )
    plt.close(fig)


def make_risk_scatter(df):
    if "risk_score" not in df.columns or "future_intent_score" not in df.columns:
        return
    sub = df[["risk_score", "future_intent_score"]].dropna()
    rng = np.random.default_rng(42)
    fig, ax = plt.subplots(figsize=(COLUMN_W, 2.55))
    ax.scatter(
        sub["risk_score"] + rng.normal(0, .04, len(sub)),
        sub["future_intent_score"] + rng.normal(0, .04, len(sub)),
        alpha=.55,
        s=14,
    )
    ax.set_xlabel("Perceived risk score (1--5)")
    ax.set_ylabel("Future-use intention (1--5)")
    ax.set_xlim(0.75, 5.25)
    ax.set_ylim(0.75, 5.25)
    ax.set_xticks([1, 2, 3, 4, 5])
    ax.set_yticks([1, 2, 3, 4, 5])
    ax.grid(linewidth=0.35, alpha=0.35)
    ax.set_axisbelow(True)
    fig.tight_layout(pad=0.3)
    save_readable(fig, OUT / "risk_vs_future_intent_scatter")


def make_persona_plot(df):
    if "persona" not in df.columns:
        return
    tab = df["persona"].value_counts(dropna=True).sort_values()
    pct = tab.values / tab.values.sum() * 100
    fig, ax = plt.subplots(figsize=(COLUMN_W, 2.15))
    labels = [wrap_label(x, 20) for x in tab.index]
    bars = ax.barh(labels, pct, height=0.62)
    ax.set_xlabel("Percent of respondents")
    ax.bar_label(bars, labels=[f"{v:.0f}%" for v in pct], padding=2, fontsize=7.0)
    ax.set_xlim(0, max(pct.max() * 1.18, 10))
    ax.grid(axis="x", linewidth=0.35, alpha=0.35)
    ax.set_axisbelow(True)
    fig.tight_layout(pad=0.3)
    save_readable(fig, OUT / "respondent_personas")


# ------------------------- THEMATIC FIGURES -------------------------

THEMATIC_LABEL_MAP = {
    "framework_context_failure": "Framework-context failure",
    "outdated_security_knowledge": "Outdated security knowledge",
    "overconfident_response": "Overconfident response",
    "severity_miscalibration": "Severity miscalibration",
    "incomplete_fix": "Incomplete fix",
    "verification_required": "Verification required",
    "hallucinated_vulnerability": "Hallucinated vulnerability",
    "insecure_fix": "Insecure fix",
    "introduced_regression": "Introduced regression",
    "incorrect_root_cause": "Incorrect root cause",
    "unsafe_suppression": "Unsafe suppression",
    "introduced_security_flaw": "Introduced security flaw",

    "lack_context": "Lack of context",
    "verification_needed": "Verification needed",
    "incorrect_explanation": "Incorrect explanation",
    "wrong_severity": "Wrong severity",
    "wrong_or_incomplete_fix": "Wrong or incomplete fix",
    "introduced_bug": "Introduced bug",
    "overconfident_answer": "Overconfident answer",
    "insecure_recommendation": "Insecure recommendation",
    "introduced_new_vulnerability": "Introduced new vulnerability",
}


def clean_theme_label(label, width=24):
    label = THEMATIC_LABEL_MAP.get(str(label), str(label).replace("_", " ").title())
    return "\n".join(textwrap.wrap(label, width=width))


def plot_frequency_csv(input_csv, output_name, include_zero=False, label_width=24):
    input_path = Path(input_csv)
    if not input_path.exists():
        print(f"Skipping thematic figure: {input_csv} not found.")
        return

    df = pd.read_csv(input_path)
    if "theme" not in df.columns or "count" not in df.columns:
        raise ValueError(f"{input_csv} must contain 'theme' and 'count' columns.")

    df = df.copy()
    df["count"] = pd.to_numeric(df["count"], errors="coerce").fillna(0)
    if not include_zero:
        df = df[df["count"] > 0]
    if df.empty:
        print(f"Skipping thematic figure: {input_csv} has no nonzero counts.")
        return

    df = df.sort_values("count", ascending=False)
    labels = [clean_theme_label(x, width=label_width) for x in df["theme"]]
    counts = df["count"].astype(int).to_numpy()

    fig_h = max(2.15, 0.38 * len(df) + 0.45)
    fig, ax = plt.subplots(figsize=(COLUMN_W, fig_h))
    bars = ax.barh(labels, counts, height=0.62)
    ax.invert_yaxis()
    ax.set_xlabel("Count")
    ax.set_xlim(0, max(counts.max() * 1.15, 1.0))
    ax.bar_label(bars, labels=[str(v) for v in counts], padding=2, fontsize=8.5)
    ax.grid(axis="x", linewidth=0.35, alpha=0.35)
    ax.set_axisbelow(True)
    fig.tight_layout(pad=0.25)

    out_path = OUT / output_name
    fig.savefig(out_path, dpi=600, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    print(f"Wrote {out_path}")


def run_thematic_figures():
    plot_frequency_csv(
        "failure_mode_frequencies.csv",
        "failure_mode_frequencies_readable.png",
        include_zero=False,
        label_width=24,
    )
    plot_frequency_csv(
        "general_thematic_frequencies.csv",
        "general_thematic_frequencies_readable.png",
        include_zero=False,
        label_width=24,
    )


# ------------------------- ENTRY POINT -------------------------

def main():
    warnings.filterwarnings("ignore", category=UserWarning)

    df, input_file = load_dataset()
    if df is not None:
        col = get_column_map(df)
        add_numeric_columns(df, col)

        # Aggregate tables, statistics, and figures are written to artifact_outputs/.

        add_composite_scores(df, col)
        run_main_analysis(df, col)
        run_deeper_analysis(df, col)

    run_thematic_figures()
    print(f"Done. Results written to: {OUT.resolve()}")


if __name__ == "__main__":
    main()

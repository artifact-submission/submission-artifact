#!/usr/bin/env python3
"""
All generated files are written to:
    artifact_outputs/
"""

from __future__ import annotations

from pathlib import Path
import itertools
import textwrap
import warnings

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt


# ------------------------- GLOBAL SETTINGS -------------------------

import argparse as _argparse
_parser = _argparse.ArgumentParser(description="Reproduce all analyses and figures.")
_parser.add_argument("--inputs", default=None,
                     help="Folder with the input files (default: the script's folder)")
_parser.add_argument("--outputs", default=None,
                     help="Folder for results (default: <inputs>/artifact_outputs)")
_args, _unknown = _parser.parse_known_args()
BASE = Path(_args.inputs).resolve() if _args.inputs else Path(__file__).resolve().parent
OUT = Path(_args.outputs).resolve() if _args.outputs else BASE / "artifact_outputs"
OUT.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 20260922

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
    if (BASE / "analysis_dataset_deidentified.csv").exists():
        input_file = BASE / "analysis_dataset_deidentified.csv"
        df = pd.read_csv(input_file)
    elif (BASE / "cleaned_responses.xlsx").exists():
        input_file = BASE / "cleaned_responses.xlsx"
        raw = pd.read_excel(input_file)
        df = raw.loc[~raw.apply(looks_like_metadata, axis=1)].copy().reset_index(drop=True)
    else:
        print("No participant-level dataset found. Skipping survey analyses.")
        return None, None

    # Stable anonymous identifiers (P001, P002, ...) follow the order of the
    # cleaned file and are used to link qualitative response units to survey rows.
    if "anon_id" not in df.columns:
        df["anon_id"] = [f"P{i + 1:03d}" for i in range(len(df))]

    # Exclude records that contain consent but no substantive survey answers.
    substantive = [c for c in get_column_map(df).values() if c in df.columns]
    answered = df[substantive].notna().sum(axis=1)
    consent_only = answered == 0
    if consent_only.any():
        print(f"Excluded {int(consent_only.sum())} consent-only record(s): "
              f"{', '.join(df.loc[consent_only, 'anon_id'])}")
        df = df.loc[~consent_only].copy().reset_index(drop=True)

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
        "confidence": "QID162",
        "clear_llm": "QID148_1",
        "clear_coding": "QID148_2",
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
    "languages": [
        "Python", "JavaScript or TypeScript", "Java", "C or C++", "C#",
        "Go", "Ruby", "Other",
    ],
    "traditional_tools_used": [
        "Static analysis tools (e.g., SonarQube, Fortify SCA, ESLint, FindBugs, Semgrep)",
        "Dynamic/runtime analysis tools (e.g., OWASP Zap, BurbSuite, Netsparker, Veracode)",
        "Dependency/Library vulnerability scanning tools (e.g., Snyk, Dependabot, OWASP Dependency Check, White Source)",
        "Container/infrastructure security tools (e.g., Trivy, Anchore, Clair, Aqua Security)",
        "CI/CD security scanning tools (e.g., GitLab SAST/DAST, GitHub Advanced Security, Jenkins security plugins, Azure DevOps security scans)",
        "Other",
    ],
    "llm_tools": [
        "ChatGPT (GPT-4, GPT-4o, GPT-4.1, GPT-3.5, etc.)",
        "Claude (Opus, Sonnet, Haiku)",
        "Gemini (Ultra, 1.5 Pro, 1.5 Flash)",
        "Llama (Llama 3, Llama 3.1 chat)",
        "Mistral / Mixtral (chat models)",
        "Perplexity (coding or chat models)",
        "Cohere Command R / R+",
        "Replit Chat / Ghostwriter",
        "Other (please specify)",
    ],
    "coding_tools": [
        "GitHub Copilot", "GitHub CodeQL", "Microsoft Security Copilot",
        "DeepCode (Snyk Code)", "AWS CodeWhisperer",
        "Amazon CodeGuru Reviewer", "Google Security AI Workbench / Sec-PaLM",
        "CodiumAI", "Tabnine Security", "Checkmarx AI Security",
        "None of the above", "Other",
    ],
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
    "trad_combo_detection": [
        "Traditional Static Analyzers", "Dynamic Analysis & Fuzzing Tools",
        "Dependency/Supply Chain Tools", "Container & Infrastructure Security Tools",
    ],
    "trad_combo_explanation": [
        "Traditional Static Analyzers", "Dynamic Analysis & Fuzzing Tools",
        "Dependency/Supply Chain Tools", "Container & Infrastructure Security Tools",
    ],
    "trad_combo_repair": [
        "Traditional Static Analyzers", "Dynamic Analysis & Fuzzing Tools",
        "Dependency/Supply Chain Tools", "Container & Infrastructure Security Tools",
    ],
}


SECURITY_FOCUSED_ROLES = {
    "Application Security Engineer / Security Analyst",
    "Security Researcher / Academic",
    "Site Reliability Engineer (SRE) / DevOps (Security-Focused)",
    "Developer (Security Champion or Security-Heavy Responsibilities)",
}

GENERAL_DEVELOPER_ROLE = "Developer / Software Engineer"
STUDENT_ROLE = "Student / Learner"


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


def pct_table(series, denominator=None, include_missing=False):
    """Return counts and percentages with an explicit denominator rule."""
    if include_missing:
        s = series.fillna("Missing/no response").astype(str)
    else:
        s = series.dropna().astype(str)
    counts = s.value_counts()
    denominator = len(s) if denominator is None else denominator
    out = pd.DataFrame({"n": counts, "percent": (counts / denominator * 100).round(1)})
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


def split_multiselect(series, known_options=None):
    """Count complete response options without splitting commas inside labels."""
    values = series.fillna("").astype(str)
    values = values[~values.str.startswith('{"ImportId"')]

    if known_options:
        rows = []
        for option in known_options:
            rows.append((option, int(values.str.contains(option, regex=False).sum())))
        counts = pd.Series(dict(rows), dtype=int).sort_values(ascending=False)
        counts = counts[counts > 0]
    else:
        items = []
        for value in values:
            items.extend(x.strip() for x in value.split(",") if x.strip())
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


def holm_adjust(pvalues):
    """Holm-adjust a one-dimensional family while preserving missing values."""
    pvalues = np.asarray(pvalues, dtype=float)
    adjusted = np.full_like(pvalues, np.nan)
    valid = np.flatnonzero(~np.isnan(pvalues))
    if len(valid) == 0:
        return adjusted
    order = valid[np.argsort(pvalues[valid])]
    running_max = 0.0
    m = len(order)
    for rank, idx in enumerate(order):
        running_max = max(running_max, (m - rank) * pvalues[idx])
        adjusted[idx] = min(running_max, 1.0)
    return adjusted


def chi_square_test_with_sparse_fallback(table, rng, monte_carlo_reps=20000):
    """Use asymptotic chi-square when adequate; otherwise use fixed-margin Monte Carlo."""
    chi2, asymptotic_p, dof, expected = stats.chi2_contingency(table)
    expected_below_five = int(np.sum(expected < 5))
    assumptions_met = bool(
        expected.min() >= 1
        and expected_below_five / expected.size <= 0.20
    )
    monte_carlo_p = np.nan
    method = "asymptotic_chi_square"
    reported_p = asymptotic_p

    if not assumptions_met:
        simulated = stats.random_table.rvs(
            table.sum(axis=1).to_numpy(),
            table.sum(axis=0).to_numpy(),
            size=monte_carlo_reps,
            random_state=rng,
        )
        simulated_chi2 = np.sum(
            ((simulated - expected) ** 2) / expected,
            axis=(1, 2),
        )
        monte_carlo_p = (
            1 + np.count_nonzero(simulated_chi2 >= chi2 - 1e-12)
        ) / (monte_carlo_reps + 1)
        reported_p = monte_carlo_p
        method = "fixed_margin_monte_carlo_chi_square"

    return {
        "chi2": chi2,
        "dof": dof,
        "p_value": reported_p,
        "test_method": method,
        "asymptotic_p_value": asymptotic_p,
        "monte_carlo_p_value": monte_carlo_p,
        "monte_carlo_reps": monte_carlo_reps if not assumptions_met else 0,
        "min_expected_cell": float(expected.min()),
        "expected_cells_below_5": expected_below_five,
        "expected_cell_assumptions_met": assumptions_met,
    }


def add_analysis_role_group(df, col):
    """Create the three prespecified participant-role groups used in Section 4.3.1."""
    if "role" not in col:
        df["analysis_role_group"] = np.nan
        return

    def classify(value):
        if value in SECURITY_FOCUSED_ROLES:
            return "Security-focused professionals"
        if value == GENERAL_DEVELOPER_ROLE:
            return "General developers/software engineers"
        if value == STUDENT_ROLE:
            return "Students/learners"
        return np.nan

    df["analysis_role_group"] = df[col["role"]].map(classify)


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


def wilcoxon_pair(df, key_a, key_b, label_a, label_b, measure, task):
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
        "measure": measure,
        "task": task,
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
    df["llm_explanation_quality_score"] = row_mean(df, ["clear_llm", "helpful_llm"])
    df["coding_explanation_quality_score"] = row_mean(df, ["clear_coding", "helpful_coding"])
    df["explanation_quality_score"] = row_mean(df, ["clear_llm", "clear_coding", "helpful_llm", "helpful_coding"])

    df["verification_breadth_score"] = row_count_multiselect(df, col, "verification", exclude_terms=["do not verify"])
    # Stricter assurance-grade verification score:
    # counts only manual code review, running tests, and static/dynamic analysis.
    verification_opts = selected_options(df, col, "verification")
    if not verification_opts.empty:
        assurance_cols = [
            "Manual code review",
            "Run tests",
            "Use static/dynamic analysis tools",
        ]
        available_assurance_cols = [
            c for c in assurance_cols if c in verification_opts.columns
        ]
        df["assurance_grade_verification_score"] = (
            verification_opts[available_assurance_cols].sum(axis=1)
            if available_assurance_cols
            else np.nan
        )
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
                if key == "role":
                    table = pct_table(
                        df[col[key]], denominator=len(df), include_missing=True
                    )
                else:
                    table = pct_table(df[col[key]])
                table.to_excel(writer, sheet_name=key[:31], index=False)
        for name, (_, keys) in GROUPS.items():
            likert_summary(df, keys).to_excel(writer, sheet_name=(name + "_summary")[:31], index=False)
        for key in [
            "languages", "traditional_tools_used", "llm_tools", "coding_tools", "contexts",
            "verification", "explanation_challenges", "trad_combo_detection",
            "trad_combo_explanation", "trad_combo_repair",
        ]:
            if key in col:
                split_multiselect(
                    df[col[key]], known_options=MULTI_OPTIONS.get(key)
                ).to_excel(writer, sheet_name=(key + "_multi")[:31], index=False)

    comparisons = []
    for task in ["detection", "explanation", "repair"]:
        comparisons.append(wilcoxon_pair(
            df, f"acc_llm_{task}", f"acc_coding_{task}",
            "LLM", "CodingTool", "accuracy", task,
        ))
        comparisons.append(wilcoxon_pair(
            df, f"eff_llm_{task}", f"eff_coding_{task}",
            "LLM", "CodingTool", "effectiveness", task,
        ))

    comparisons = pd.DataFrame([x for x in comparisons if x is not None])
    if not comparisons.empty:
        comparisons["p_holm"] = holm_adjust(comparisons["p_value"])
    comparisons.to_csv(OUT / "wilcoxon_llm_vs_coding_tools.csv", index=False)

    explanation_comparisons = []
    for a, b, measure in [
        ("helpful_llm", "helpful_coding", "explanation_helpfulness"),
        ("clear_llm", "clear_coding", "explanation_clarity"),
    ]:
        explanation_comparisons.append(wilcoxon_pair(
            df, a, b, "LLM", "CodingTool", measure, "explanation"
        ))
    explanation_comparisons = pd.DataFrame(
        [x for x in explanation_comparisons if x is not None]
    )
    if not explanation_comparisons.empty:
        explanation_comparisons["p_holm"] = holm_adjust(
            explanation_comparisons["p_value"]
        )
    explanation_comparisons.to_csv(
        OUT / "wilcoxon_explanation_quality.csv", index=False
    )

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
    chi_rng = np.random.default_rng(RANDOM_SEED)
    for xkey in ["role", "experience"]:
        if xkey in col:
            for ykey in ["better_detection", "better_explanation", "better_repair"]:
                if ykey in col:
                    table = pd.crosstab(df[col[xkey]], df[col[ykey]])
                    if table.shape[0] >= 2 and table.shape[1] >= 2:
                        result = chi_square_test_with_sparse_fallback(
                            table, rng=chi_rng
                        )
                        chi_rows.append({
                            "x": xkey,
                            "y": ykey,
                            **result,
                        })
                        table.to_csv(OUT / f"crosstab_{xkey}_by_{ykey}.csv")
    chi_results = pd.DataFrame(chi_rows)
    if not chi_results.empty:
        chi_results["p_holm"] = holm_adjust(chi_results["p_value"])
    chi_results.to_csv(OUT / "chi_square_tests.csv", index=False)

    if "risk" in col:
        save_bar_from_counts(df[col["risk"]], "risk_distribution")
    if "better_detection" in col:
        save_bar_from_counts(df[col["better_detection"]], "better_detection")

    if "verification" in col:
        ver = split_multiselect(
            df[col["verification"]], known_options=MULTI_OPTIONS["verification"]
        ).head(10)
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
        "llm_explanation_quality_score": available_num(df, ["clear_llm", "helpful_llm"]),
        "coding_explanation_quality_score": available_num(df, ["clear_coding", "helpful_coding"]),
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
        "positive_fix_outcome_score", "negative_fix_outcome_score",
        "llm_explanation_quality_score", "coding_explanation_quality_score",
        "verification_breadth_score", "assurance_grade_verification_score",
        "sdlc_breadth_score", "challenge_breadth_score",
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

    for x in [
        "negative_fix_outcome_score", "positive_fix_outcome_score",
        "llm_explanation_quality_score", "coding_explanation_quality_score",
        "challenge_breadth_score",
    ]:
        for y in ["risk_score", "future_intent_score", "traditional_preference_score"]:
            if x in df.columns and y in df.columns:
                pairs.append(spearman_pair(df, x, y))

    corr_df = pd.DataFrame(pairs).drop_duplicates()
    corr_df.to_csv(OUT / "deep_spearman_composite_relationships.csv", index=False)

    if not corr_df.empty:
        corr_df = corr_df.copy()
        corr_df["p_holm"] = holm_adjust(corr_df["p"])
        corr_df.to_csv(OUT / "deep_spearman_composite_relationships_holm.csv", index=False)

    run_regression(df)
    run_role_comparisons(df, col)
    run_group_comparisons(df, col)
    run_task_specific_reliance_analysis(df)
    run_quality_and_outcome_relationships(df)
    make_personas(df)
    # Participant-level composite-score data are not exported by default because
    # IRB/consent restrictions may prohibit sharing row-level survey data.
    make_paper_heatmap(df)
    make_risk_scatter(df)
    make_persona_plot(df)
    run_revision_analyses(df, col)


def fit_standardized_hc3(df, outcome, predictors, interaction=False):
    """Fit standardized OLS with HC3 covariance using NumPy and SciPy."""
    needed = [outcome] + predictors
    reg = df[needed].dropna().copy()
    if len(reg) < 10 or len(predictors) < 2:
        return None

    standard_deviations = reg.std(axis=0, ddof=0)
    if (standard_deviations == 0).any():
        raise ValueError("Regression contains a zero-variance variable.")
    z = (reg - reg.mean(axis=0)) / standard_deviations

    term_names = ["const"] + list(predictors)
    columns = [np.ones(len(z))] + [z[c].to_numpy(dtype=float) for c in predictors]
    if interaction:
        if "risk_score" not in predictors or "verification_breadth_score" not in predictors:
            raise ValueError("The interaction model requires risk and verification breadth.")
        columns.append(
            z["risk_score"].to_numpy(dtype=float)
            * z["verification_breadth_score"].to_numpy(dtype=float)
        )
        term_names.append("risk_x_verification_breadth")

    X = np.column_stack(columns)
    y = z[outcome].to_numpy(dtype=float)
    n, k = X.shape
    xtx_inv = np.linalg.pinv(X.T @ X)
    beta = xtx_inv @ X.T @ y
    residual = y - X @ beta
    leverage = np.einsum("ij,jk,ik->i", X, xtx_inv, X)
    scaled_residual_sq = (residual / (1.0 - leverage)) ** 2
    meat = X.T @ (scaled_residual_sq[:, None] * X)
    covariance = xtx_inv @ meat @ xtx_inv
    robust_se = np.sqrt(np.diag(covariance))
    statistic = beta / robust_se
    p_values = 2.0 * stats.norm.sf(np.abs(statistic))
    critical = stats.norm.ppf(0.975)

    coefficients = pd.DataFrame({
        "term": term_names,
        "std_beta": beta,
        "robust_se_hc3": robust_se,
        "z": statistic,
        "p_value": p_values,
        "ci95_low": beta - critical * robust_se,
        "ci95_high": beta + critical * robust_se,
    })

    sse = float(np.sum(residual ** 2))
    sst = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - sse / sst
    adjusted_r2 = 1.0 - (1.0 - r2) * (n - 1) / (n - k)

    slope_beta = beta[1:]
    slope_covariance = covariance[1:, 1:]
    wald_chi2 = float(
        slope_beta @ np.linalg.pinv(slope_covariance) @ slope_beta
    )
    numerator_df = k - 1
    denominator_df = n - k
    robust_f = wald_chi2 / numerator_df

    vif_rows = []
    for column_index, term in enumerate(term_names[1:], start=1):
        target = X[:, column_index]
        other = np.delete(X, column_index, axis=1)
        fitted = other @ np.linalg.lstsq(other, target, rcond=None)[0]
        target_sst = np.sum((target - target.mean()) ** 2)
        predictor_r2 = 1.0 - np.sum((target - fitted) ** 2) / target_sst
        vif_rows.append({"term": term, "vif": 1.0 / (1.0 - predictor_r2)})
    vif = pd.DataFrame(vif_rows)

    diagnostics = {
        "n": n,
        "n_predictors": numerator_df,
        "r2": r2,
        "adjusted_r2": adjusted_r2,
        "robust_wald_chi2": wald_chi2,
        "wald_df": numerator_df,
        "wald_p_value": float(stats.chi2.sf(wald_chi2, numerator_df)),
        "robust_f": robust_f,
        "f_df_num": numerator_df,
        "f_df_den": denominator_df,
        "f_p_value": float(stats.f.sf(robust_f, numerator_df, denominator_df)),
        "min_vif": float(vif["vif"].min()),
        "max_vif": float(vif["vif"].max()),
    }
    return coefficients, diagnostics, vif


def write_regression_result(result, model_label, filename_stem):
    if result is None:
        return None
    coefficients, diagnostics, vif = result
    coefficients.insert(0, "model", model_label)
    vif.insert(0, "model", model_label)
    diagnostics = {"model": model_label, **diagnostics}

    coefficients.to_csv(OUT / f"{filename_stem}.csv", index=False)
    vif.to_csv(OUT / f"{filename_stem}_vif.csv", index=False)
    pd.DataFrame([diagnostics]).to_csv(
        OUT / f"{filename_stem}_diagnostics.csv", index=False
    )
    with open(OUT / f"{filename_stem}_summary.txt", "w", encoding="utf-8") as f:
        f.write(pd.DataFrame([diagnostics]).to_string(index=False))
        f.write("\n\n")
        f.write(coefficients.to_string(index=False))
        f.write("\n\n")
        f.write(vif.to_string(index=False))
        f.write("\n")
    return diagnostics


def run_regression(df):
    """Run the primary, sensitivity, and prespecified interaction models."""
    outcome = "future_intent_score"
    main_predictors = [
        "overall_performance_score", "risk_score", "verification_breadth_score",
        "traditional_preference_score", "negative_fix_outcome_score",
        "experience_score", "security_task_frequency_score",
    ]
    assurance_predictors = [
        "overall_performance_score", "risk_score",
        "assurance_grade_verification_score", "traditional_preference_score",
        "negative_fix_outcome_score", "experience_score",
        "security_task_frequency_score",
    ]

    if outcome not in df.columns:
        raise ValueError(f"Outcome column {outcome} was not constructed.")

    main_result = fit_standardized_hc3(df, outcome, main_predictors)
    assurance_result = fit_standardized_hc3(df, outcome, assurance_predictors)
    interaction_result = fit_standardized_hc3(
        df, outcome, main_predictors, interaction=True
    )

    rows = []
    for result, label, stem in [
        (
            main_result,
            "main_broad_verification_breadth",
            "regression_future_intent_standardized_OLS",
        ),
        (
            assurance_result,
            "sensitivity_assurance_grade_verification",
            "regression_future_intent_assurance_grade_verification_OLS",
        ),
        (
            interaction_result,
            "interaction_risk_by_verification_breadth",
            "regression_future_intent_interaction_OLS",
        ),
    ]:
        row = write_regression_result(result, label, stem)
        if row is not None:
            rows.append(row)
    pd.DataFrame(rows).to_csv(
        OUT / "regression_future_intent_model_comparison.csv", index=False
    )


def run_role_comparisons(df, col):
    """Reproduce the three-group role analysis reported in Section 4.3.1."""
    if "analysis_role_group" not in df.columns:
        add_analysis_role_group(df, col)

    order = [
        "Security-focused professionals",
        "General developers/software engineers",
        "Students/learners",
    ]
    sub = df[df["analysis_role_group"].isin(order)].copy()
    counts = (
        sub["analysis_role_group"].value_counts().reindex(order, fill_value=0)
        .rename_axis("role_group").reset_index(name="n")
    )
    counts.to_csv(OUT / "role_group_counts.csv", index=False)

    outcome_labels = {
        "security_task_frequency_score": "security_task_frequency",
        "verification_breadth_score": "verification_breadth",
        "risk_score": "perceived_risk",
        "future_intent_score": "future_use_intention",
    }
    summaries = []
    tests = []
    for variable, label in outcome_labels.items():
        groups = []
        for group in order:
            values = sub.loc[sub["analysis_role_group"] == group, variable].dropna()
            groups.append(values.to_numpy())
            summaries.append({
                "outcome": label,
                "role_group": group,
                "n": len(values),
                "median": values.median(),
                "q1": values.quantile(0.25),
                "q3": values.quantile(0.75),
                "iqr": values.quantile(0.75) - values.quantile(0.25),
            })
        h_statistic, p_value = stats.kruskal(*groups)
        tests.append({
            "outcome": label,
            "H": h_statistic,
            "df": len(order) - 1,
            "p_value": p_value,
        })
    pd.DataFrame(summaries).to_csv(
        OUT / "role_group_outcome_summaries.csv", index=False
    )
    pd.DataFrame(tests).to_csv(
        OUT / "role_group_kruskal_tests.csv", index=False
    )

    affirmative_methods = [
        option for option in MULTI_OPTIONS["verification"]
        if option != "I generally do not verify"
    ]
    options = selected_options(df, col, "verification")
    options["analysis_role_group"] = df["analysis_role_group"]
    rows = []
    for method in affirmative_methods:
        table = pd.crosstab(
            options["analysis_role_group"], options[method]
        ).reindex(index=order, columns=[False, True], fill_value=0)
        chi2, p_value, degrees_freedom, expected = stats.chi2_contingency(table)
        n_total = int(table.to_numpy().sum())
        row = {
            "verification_method": method,
            "chi2": chi2,
            "df": degrees_freedom,
            "p_value": p_value,
            "cramers_v": np.sqrt(chi2 / n_total),
            "min_expected_cell": float(expected.min()),
        }
        for group, prefix in zip(order, ["security", "general", "student"]):
            group_n = int(table.loc[group].sum())
            yes_n = int(table.loc[group, True])
            row[f"{prefix}_n"] = group_n
            row[f"{prefix}_yes"] = yes_n
            row[f"{prefix}_percent"] = 100.0 * yes_n / group_n
        rows.append(row)
    verification_tests = pd.DataFrame(rows)
    verification_tests["p_holm"] = holm_adjust(verification_tests["p_value"])
    verification_tests.to_csv(
        OUT / "role_verification_method_comparisons.csv", index=False
    )


def run_group_comparisons(df, col):
    context_rows = []
    if "contexts" in col:
        ctx_df = selected_options(df, col, "contexts")
        contexts = [c for c in ctx_df.columns if "None of the above" not in c]
        for index, ctx in enumerate(contexts, start=1):
            indicator = f"context_{index:02d}"
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
    context_results = pd.DataFrame(context_rows)
    if not context_results.empty:
        context_results["p_holm"] = holm_adjust(context_results["p"])
    context_results.to_csv(OUT / "sdlc_context_group_comparisons.csv", index=False)

    challenge_rows = []
    if "explanation_challenges" in col:
        ch_df = selected_options(df, col, "explanation_challenges")
        for index, ch in enumerate(ch_df.columns, start=1):
            indicator = f"challenge_{index:02d}"
            df[indicator] = ch_df[ch]
            for outcome in ["risk_score", "overall_performance_score", "future_intent_score", "traditional_preference_score"]:
                if outcome in df.columns:
                    res = mann_whitney_by_binary(df, indicator, outcome, True)
                    if res:
                        res["challenge"] = ch
                        challenge_rows.append(res)
    challenge_results = pd.DataFrame(challenge_rows)
    if not challenge_results.empty:
        challenge_results["p_holm"] = holm_adjust(challenge_results["p"])
    challenge_results.to_csv(
        OUT / "explanation_challenge_group_comparisons.csv", index=False
    )


def _future_task_columns(df):
    """Return available numeric future-use items in manuscript task order."""
    task_columns = {
        "Detection": "future_detection_num",
        "Explanation": "future_explanation_num",
        "Repair": "future_repair_num",
    }
    return {task: column for task, column in task_columns.items() if column in df}


def _paired_wilcoxon(a, b):
    """Run a paired two-sided Wilcoxon test after pairwise deletion."""
    pair = pd.concat([a, b], axis=1).dropna()
    if len(pair) < 5:
        return None
    difference = pair.iloc[:, 1] - pair.iloc[:, 0]
    if (difference != 0).sum() == 0:
        statistic, p_value = np.nan, 1.0
    else:
        statistic, p_value = stats.wilcoxon(
            pair.iloc[:, 0], pair.iloc[:, 1],
            zero_method="wilcox", alternative="two-sided",
        )
    return {
        "n_pairs": len(pair),
        "median_task_a": pair.iloc[:, 0].median(),
        "median_task_b": pair.iloc[:, 1].median(),
        "mean_task_a": pair.iloc[:, 0].mean(),
        "mean_task_b": pair.iloc[:, 1].mean(),
        "wilcoxon_statistic": statistic,
        "p_value": p_value,
    }


def _spearman_difference_bootstrap(
    complete, outcome_a, outcome_b, rng, bootstrap_reps=20000
):
    """Paired row bootstrap for the difference between two dependent correlations."""
    risk = complete["risk_score"].to_numpy(dtype=float)
    a = complete[outcome_a].to_numpy(dtype=float)
    b = complete[outcome_b].to_numpy(dtype=float)
    observed_a = stats.spearmanr(risk, a).statistic
    observed_b = stats.spearmanr(risk, b).statistic
    observed_difference = observed_a - observed_b

    bootstrap_differences = np.empty(bootstrap_reps, dtype=float)
    n = len(complete)
    for index in range(bootstrap_reps):
        sample = rng.integers(0, n, size=n)
        bootstrap_differences[index] = (
            stats.spearmanr(risk[sample], a[sample]).statistic
            - stats.spearmanr(risk[sample], b[sample]).statistic
        )

    # Two planned dependent-correlation contrasts are reported. The 97.5%
    # interval is Bonferroni familywise-adjusted for those two contrasts.
    return {
        "n_complete": n,
        "bootstrap_reps": bootstrap_reps,
        "observed_delta_rho": observed_difference,
        "ci95_low": np.nanquantile(bootstrap_differences, 0.025),
        "ci95_high": np.nanquantile(bootstrap_differences, 0.975),
        "familywise_ci_level": 0.975,
        "ci97_5_low": np.nanquantile(bootstrap_differences, 0.0125),
        "ci97_5_high": np.nanquantile(bootstrap_differences, 0.9875),
    }


def _wilson_interval(successes, total, alpha=0.05):
    """Wilson confidence interval for a binomial proportion."""
    if total == 0:
        return np.nan, np.nan
    z = stats.norm.ppf(1 - alpha / 2)
    proportion = successes / total
    denominator = 1 + z ** 2 / total
    center = (proportion + z ** 2 / (2 * total)) / denominator
    half_width = (
        z * np.sqrt(proportion * (1 - proportion) / total + z ** 2 / (4 * total ** 2))
        / denominator
    )
    return center - half_width, center + half_width


def run_task_specific_reliance_analysis(df):
    """Test whether perceived risk sets different reliance boundaries by task."""
    task_columns = _future_task_columns(df)
    if "risk_score" not in df or len(task_columns) != 3:
        return

    # Family 1: three task-specific risk--future-use correlations.
    correlation_rows = []
    for task, column in task_columns.items():
        result = spearman_pair(df, "risk_score", column)
        correlation_rows.append({
            "task": task,
            "n": result["n"],
            "spearman_rho": result["rho"],
            "p_value": result["p"],
        })
    correlations = pd.DataFrame(correlation_rows)
    correlations["p_holm_3"] = holm_adjust(correlations["p_value"])
    correlations.to_csv(
        OUT / "task_specific_risk_future_use_correlations.csv", index=False
    )

    # Family 2: the two planned contrasts compare repair with detection and
    # explanation using the same complete participants in a paired bootstrap.
    complete = df[["risk_score", *task_columns.values()]].dropna()
    bootstrap_rng = np.random.default_rng(RANDOM_SEED)
    contrast_rows = []
    for comparator in ["Detection", "Explanation"]:
        result = _spearman_difference_bootstrap(
            complete,
            task_columns["Repair"],
            task_columns[comparator],
            bootstrap_rng,
        )
        contrast_rows.append({
            "contrast": f"Repair minus {comparator}",
            **result,
        })
    pd.DataFrame(contrast_rows).to_csv(
        OUT / "task_specific_risk_correlation_differences_bootstrap.csv",
        index=False,
    )

    # Within-participant task comparisons for the full analytic sample and the
    # substantively important high-risk subgroup (risk ratings 4--5).
    subgroup_frames = {
        "All respondents": df,
        "High perceived risk (4--5)": df[df["risk_score"] >= 4],
    }
    omnibus_rows = []
    pairwise_rows = []
    for subgroup, subgroup_df in subgroup_frames.items():
        complete_tasks = subgroup_df[list(task_columns.values())].dropna()
        if len(complete_tasks) >= 5:
            statistic, p_value = stats.friedmanchisquare(
                *[complete_tasks[column] for column in task_columns.values()]
            )
            omnibus_rows.append({
                "subgroup": subgroup,
                "n_complete": len(complete_tasks),
                "friedman_chi2": statistic,
                "df": len(task_columns) - 1,
                "p_value": p_value,
            })

        subgroup_pairwise = []
        for task_a, task_b in itertools.combinations(task_columns, 2):
            result = _paired_wilcoxon(
                subgroup_df[task_columns[task_a]],
                subgroup_df[task_columns[task_b]],
            )
            if result:
                subgroup_pairwise.append({
                    "subgroup": subgroup,
                    "task_a": task_a,
                    "task_b": task_b,
                    **result,
                })
        if subgroup_pairwise:
            subgroup_pairwise = pd.DataFrame(subgroup_pairwise)
            subgroup_pairwise["p_holm_within_subgroup_3"] = holm_adjust(
                subgroup_pairwise["p_value"]
            )
            pairwise_rows.append(subgroup_pairwise)

    omnibus = pd.DataFrame(omnibus_rows)
    if not omnibus.empty:
        omnibus["p_holm_across_2_subgroups"] = holm_adjust(omnibus["p_value"])
    omnibus.to_csv(OUT / "future_intention_task_omnibus.csv", index=False)
    if pairwise_rows:
        pd.concat(pairwise_rows, ignore_index=True).to_csv(
            OUT / "future_intention_task_pairwise.csv", index=False
        )

    # Descriptive adoption rates show how intended use changes across risk
    # levels without treating the ordinal scale as a continuous outcome model.
    risk_bands = pd.cut(
        df["risk_score"],
        bins=[0.5, 2.5, 3.5, 5.5],
        labels=["Lower risk (1--2)", "Moderate risk (3)", "High risk (4--5)"],
    )
    rate_rows = []
    for risk_band in risk_bands.cat.categories:
        mask = risk_bands == risk_band
        for task, column in task_columns.items():
            values = df.loc[mask, column].dropna()
            likely = int((values >= 4).sum())
            ci_low, ci_high = _wilson_interval(likely, len(values))
            rate_rows.append({
                "risk_band": str(risk_band),
                "task": task,
                "n": len(values),
                "mean": values.mean(),
                "sd": values.std(),
                "median": values.median(),
                "n_likely_4_or_5": likely,
                "percent_likely_4_or_5": 100 * likely / len(values) if len(values) else np.nan,
                "wilson_95_low_percent": 100 * ci_low,
                "wilson_95_high_percent": 100 * ci_high,
            })
    rates = pd.DataFrame(rate_rows)
    rates.to_csv(OUT / "future_intention_by_risk_and_task.csv", index=False)

    fig, ax = plt.subplots(figsize=(COLUMN_W, 2.35))
    x = np.arange(len(risk_bands.cat.categories))
    plot_styles = {
        "Detection": {"marker": "o", "linestyle": "-"},
        "Explanation": {"marker": "s", "linestyle": "--"},
        "Repair": {"marker": "^", "linestyle": "-."},
    }
    for task in task_columns:
        task_rates = rates[rates["task"] == task]
        y = task_rates["percent_likely_4_or_5"].to_numpy()
        lower = y - task_rates["wilson_95_low_percent"].to_numpy()
        upper = task_rates["wilson_95_high_percent"].to_numpy() - y
        ax.errorbar(
            x, y, yerr=[lower, upper], capsize=2, markersize=4,
            label=task, **plot_styles[task]
        )
    ax.set_xticks(x)
    ax.set_xticklabels(["Lower\n(1--2)", "Moderate\n(3)", "High\n(4--5)"])
    ax.set_xlabel("Perceived risk")
    ax.set_ylabel("Likely future use (%)")
    ax.set_ylim(0, 105)
    ax.legend(frameon=False, fontsize=7.2, ncol=3, loc="lower left")
    ax.grid(axis="y", linewidth=0.35, alpha=0.35)
    ax.set_axisbelow(True)
    fig.tight_layout(pad=0.3)
    save_readable(fig, OUT / "future_use_by_risk_and_task")


def run_quality_and_outcome_relationships(df):
    """Run focused quality/outcome tests with correction within theory families."""
    families = {
        "repair_outcomes_4": [
            ("positive_fix_outcome_score", "risk_score"),
            ("positive_fix_outcome_score", "future_intent_score"),
            ("negative_fix_outcome_score", "risk_score"),
            ("negative_fix_outcome_score", "future_intent_score"),
        ],
        "tool_specific_explanation_quality_4": [
            ("llm_explanation_quality_score", "risk_score"),
            ("llm_explanation_quality_score", "future_intent_score"),
            ("coding_explanation_quality_score", "risk_score"),
            ("coding_explanation_quality_score", "future_intent_score"),
        ],
    }
    rows = []
    for family, pairs in families.items():
        family_rows = []
        for x, y in pairs:
            if x in df and y in df:
                family_rows.append({"family": family, **spearman_pair(df, x, y)})
        if family_rows:
            family_df = pd.DataFrame(family_rows)
            family_df["p_holm_within_family"] = holm_adjust(family_df["p"])
            rows.append(family_df)
    if rows:
        pd.concat(rows, ignore_index=True).to_csv(
            OUT / "quality_outcome_relationships_holm.csv", index=False
        )


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

    correlation_tests = []
    for x, y in itertools.combinations(heat_cols, 2):
        pair = df[[x, y]].dropna()
        rho, p_value = stats.spearmanr(pair[x], pair[y])
        correlation_tests.append({
            "x": x,
            "y": y,
            "n": len(pair),
            "spearman_rho": rho,
            "p_value": p_value,
        })
    correlation_tests = pd.DataFrame(correlation_tests)
    correlation_tests["p_holm_28"] = holm_adjust(
        correlation_tests["p_value"]
    )
    correlation_tests.to_csv(
        OUT / "composite_correlation_tests_holm.csv", index=False
    )

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


# ------------------------- REVISION ANALYSES (TOSEM revision) -------------------------
#
# These analyses support the revised RQ1, RQ3, and RQ5 text. They need
# statsmodels (pip install statsmodels). If statsmodels is missing, the
# model-based parts are skipped and a message is printed.
# All outputs are written to artifact_outputs/revision/.

REVISION_OUT = OUT / "revision"
TASKS3 = ["detection", "explanation", "repair"]
RISK_ORDER = ["Not risky", "Slightly", "Moderately", "Very", "Extremely"]
BETTER_TOOL_ORDER = [
    "AI-based coding tools", "General-purpose LLMs", "About the same",
    "Depends on the situation", "Not sure",
]
REVISION_COVARIATES = [
    "overall_performance_score", "verification_breadth_score",
    "traditional_preference_score", "experience_score",
    "security_task_frequency_score",
]
REVISION_BASE_PREDICTORS = [
    "overall_performance_score", "risk_score", "verification_breadth_score",
    "traditional_preference_score", "negative_fix_outcome_score",
    "experience_score", "security_task_frequency_score",
]


def _zscore(s):
    return (s - s.mean()) / s.std()


def _add_revision_columns(df, col):
    """Per-method verification indicators, channel, and duration."""
    opts = selected_options(df, col, "verification").astype(int)
    rename = {
        "Manual code review": "v_review",
        "Run tests": "v_tests",
        "Use static/dynamic analysis tools": "v_sast",
        "Compare multiple AI answers": "v_compare",
        "Consult documentation": "v_docs",
        "I generally do not verify": "v_none",
    }
    for old, new in rename.items():
        if old in opts.columns:
            df[new] = opts[old]
    df["breadth_no_compare"] = df[["v_review", "v_tests", "v_sast", "v_docs"]].sum(axis=1)
    # Recruitment channel for analysis: respondents who entered a Prolific ID.
    # (One respondent who used the Prolific link did not enter an ID and is
    # grouped with the community/professional channels.)
    if "PROLIFIC_PID" in df.columns:
        pid = df["PROLIFIC_PID"].astype(str)
        df["prolific"] = df["PROLIFIC_PID"].notna() & (pid.str.len() > 5)
    if "Duration (in seconds)" in df.columns:
        df["duration_min"] = pd.to_numeric(df["Duration (in seconds)"], errors="coerce") / 60


def _ols_hc3_table(df, outcome, predictors, label):
    import statsmodels.api as sm
    s = df[[outcome] + predictors].dropna()
    zz = s.apply(_zscore)
    m = sm.OLS(zz[outcome], sm.add_constant(zz[predictors])).fit(cov_type="HC3")
    ci = m.conf_int()
    return pd.DataFrame({
        "model": label, "term": predictors,
        "beta": m.params[predictors].values, "robust_se": m.bse[predictors].values,
        "ci_low": ci.loc[predictors, 0].values, "ci_high": ci.loc[predictors, 1].values,
        "p": m.pvalues[predictors].values,
        "n": len(s), "R2": m.rsquared, "adjR2": m.rsquared_adj,
    })


def revision_sample_description(df, col):
    rows = []
    for key in ["experience", "security_frequency"]:
        if key in col:
            vc = df[col[key]].value_counts(dropna=False)
            for level, n in vc.items():
                rows.append({"variable": key, "level": level, "n": int(n),
                             "percent": round(100 * n / len(df), 1)})
    if "prolific" in df.columns:
        for level, mask in [("Prolific", df["prolific"]), ("Community/university", ~df["prolific"])]:
            rows.append({"variable": "channel", "level": level, "n": int(mask.sum()),
                         "percent": round(100 * mask.mean(), 1)})
    if "duration_min" in df.columns:
        d = df["duration_min"]
        rows.append({"variable": "duration_min", "level": "median", "n": None, "percent": round(d.median(), 1)})
        rows.append({"variable": "duration_min", "level": "Q1", "n": None, "percent": round(d.quantile(.25), 1)})
        rows.append({"variable": "duration_min", "level": "Q3", "n": None, "percent": round(d.quantile(.75), 1)})
        rows.append({"variable": "duration_min", "level": "under 4 min", "n": int((d < 4).sum()), "percent": None})
    pd.DataFrame(rows).to_csv(REVISION_OUT / "R01_sample_description.csv", index=False)

    # Channel comparison on key measures
    if "prolific" in df.columns:
        rows = []
        for v in ["future_intent_score", "risk_score", "overall_performance_score",
                  "verification_breadth_score", "overall_ai_usage_score",
                  "experience_score", "security_task_frequency_score"]:
            a = df.loc[df["prolific"], v].dropna()
            b = df.loc[~df["prolific"], v].dropna()
            if len(a) and len(b):
                u = stats.mannwhitneyu(a, b)
                rows.append({"variable": v, "median_prolific": a.median(), "n_prolific": len(a),
                             "median_other": b.median(), "n_other": len(b), "p_MWU": u.pvalue})
        pd.DataFrame(rows).to_csv(REVISION_OUT / "R02_channel_comparison.csv", index=False)

    # Item-level Ns for accuracy / effectiveness / usage (display logic)
    rows = []
    for construct in ["use", "acc", "eff"]:
        for tool in ["llm", "coding"]:
            for task in TASKS3:
                c = f"{construct}_{tool}_{task}_num"
                if c in df.columns:
                    rows.append({"construct": construct, "tool": tool, "task": task,
                                 "n": int(df[c].notna().sum())})
    pd.DataFrame(rows).to_csv(REVISION_OUT / "R03_item_level_n.csv", index=False)


def revision_task_risk_robustness(df):
    subsets = [("All", pd.Series(True, index=df.index))]
    if "prolific" in df.columns:
        subsets += [("Prolific", df["prolific"]), ("Non-Prolific", ~df["prolific"])]
    if "duration_min" in df.columns:
        subsets += [("Excluding under 4 min", df["duration_min"] >= 4)]
    rows = []
    for label, mask in subsets:
        for t in TASKS3:
            s = df.loc[mask, ["risk_score", f"future_{t}_num"]].dropna()
            if len(s) > 5:
                r, p = stats.spearmanr(s.iloc[:, 0], s.iloc[:, 1])
                rows.append({"subset": label, "task": t, "n": len(s), "rho": r, "p": p})
    pd.DataFrame(rows).to_csv(REVISION_OUT / "R04_risk_by_task_robustness.csv", index=False)

    rows = []
    for t in TASKS3:
        x = df[f"future_{t}_num"].dropna()
        rows.append({"task": t, "n": len(x), "top_box_pct": 100 * (x == 5).mean(),
                     "top_two_pct": 100 * (x >= 4).mean()})
    pd.DataFrame(rows).to_csv(REVISION_OUT / "R05_future_use_ceiling.csv", index=False)

    rows = []
    for a, b in [("repair", "explanation"), ("repair", "detection"), ("detection", "explanation")]:
        s = df[["risk_score", f"future_{a}_num", f"future_{b}_num"]].dropna()
        r, p = stats.spearmanr(s["risk_score"], s[f"future_{a}_num"] - s[f"future_{b}_num"])
        rows.append({"difference": f"{a} - {b}", "n": len(s), "rho_with_risk": r, "p": p})
    pd.DataFrame(rows).to_csv(REVISION_OUT / "R06_within_person_difference_scores.csv", index=False)


def revision_task_risk_models(df):
    """Primary RQ5 model (risk x task) and task-specific ordinal models."""
    import statsmodels.formula.api as smf
    from statsmodels.miscmodels.ordinal_model import OrderedModel

    d = df.copy()
    d["pid"] = np.arange(len(d))
    long = d.melt(
        id_vars=["pid", "risk_score"] + REVISION_COVARIATES,
        value_vars=[f"future_{t}_num" for t in TASKS3],
        var_name="task", value_name="future_use",
    ).dropna()
    long["task"] = pd.Categorical(
        long["task"].str.extract(r"future_(\w+)_num")[0],
        categories=["explanation", "detection", "repair"],
    )
    for c in ["risk_score"] + REVISION_COVARIATES:
        long[c + "_z"] = _zscore(long[c])
    formula = "future_use ~ C(task) * risk_score_z + " + " + ".join(c + "_z" for c in REVISION_COVARIATES)
    m = smf.ols(formula, long).fit(cov_type="cluster", cov_kwds={"groups": long["pid"]})
    ci = m.conf_int()
    out = pd.DataFrame({"term": m.params.index, "coef": m.params.values, "se": m.bse.values,
                        "ci_low": ci[0].values, "ci_high": ci[1].values, "p": m.pvalues.values})
    out["n_observations"] = len(long)
    out["n_respondents"] = long["pid"].nunique()
    out.to_csv(REVISION_OUT / "R07_risk_by_task_interaction_model.csv", index=False)

    rows = []
    for t in TASKS3:
        s = df[[f"future_{t}_num", "risk_score"] + REVISION_COVARIATES].dropna()
        X = s[["risk_score"] + REVISION_COVARIATES].apply(_zscore)
        om = OrderedModel(s[f"future_{t}_num"].astype(int), X, distr="logit").fit(method="bfgs", disp=False)
        lo, hi = om.conf_int().loc["risk_score"]
        rows.append({"task": t, "n": len(s), "logit_risk": om.params["risk_score"],
                     "OR_per_SD": np.exp(om.params["risk_score"]),
                     "OR_ci_low": np.exp(lo), "OR_ci_high": np.exp(hi),
                     "p": om.pvalues["risk_score"]})
    pd.DataFrame(rows).to_csv(REVISION_OUT / "R08_ordinal_logit_by_task.csv", index=False)


def revision_regression_sensitivity(df):
    """Decomposition of verification breadth and omitted-predictor checks."""
    base = REVISION_BASE_PREDICTORS
    no_breadth = [b for b in base if b != "verification_breadth_score"]
    models = [
        ("M0 original", "future_intent_score", base),
        ("M1 breadth without compare-AI", "future_intent_score",
         [b if b != "verification_breadth_score" else "breadth_no_compare" for b in base]),
        ("M2 per-method", "future_intent_score",
         no_breadth + ["v_review", "v_tests", "v_sast", "v_compare", "v_docs"]),
        ("M3 + AI usage", "future_intent_score", base + ["overall_ai_usage_score"]),
        ("M4 + positive repair outcomes", "future_intent_score", base + ["positive_fix_outcome_score"]),
    ] + [(f"M5 task outcome: {t}", f"future_{t}_num", base) for t in TASKS3]
    tables = [_ols_hc3_table(df, y, X, label) for label, y, X in models]
    pd.concat(tables).to_csv(REVISION_OUT / "R09_regression_sensitivity.csv", index=False)


def revision_tool_allocation(df, col):
    """Preferred tool category by task (QID154) and within-person McNemar test."""
    keys = {t: col.get(f"better_{t}") for t in TASKS3}
    if any(v is None for v in keys.values()):
        return
    b = pd.DataFrame({t: df[keys[t]] for t in TASKS3})
    counts = pd.DataFrame({t: b[t].value_counts() for t in TASKS3}).reindex(BETTER_TOOL_ORDER)
    pct = pd.DataFrame({t: b[t].value_counts(normalize=True) * 100 for t in TASKS3}).reindex(BETTER_TOOL_ORDER)
    pd.concat({"n": counts, "percent": pct.round(1)}, axis=1).to_csv(REVISION_OUT / "R10_better_tool_by_task.csv")

    short = {"General-purpose LLMs": "LLM", "AI-based coding tools": "Coding"}
    m = b.dropna().apply(lambda s: s.map(lambda x: short.get(x, "Other")))
    both = m[(m["explanation"] != "Other") & (m["repair"] != "Other")]
    llm_to_coding = int(((both["explanation"] == "LLM") & (both["repair"] == "Coding")).sum())
    coding_to_llm = int(((both["explanation"] == "Coding") & (both["repair"] == "LLM")).sum())
    # Exact McNemar test = two-sided binomial test on the discordant pairs.
    p = stats.binomtest(min(llm_to_coding, coding_to_llm), llm_to_coding + coding_to_llm, 0.5).pvalue
    pd.DataFrame([{
        "n_specific_choice_both_tasks": len(both),
        "LLM_for_explanation_Coding_for_repair": llm_to_coding,
        "Coding_for_explanation_LLM_for_repair": coding_to_llm,
        "exact_mcnemar_p": p,
    }]).to_csv(REVISION_OUT / "R11_mcnemar_explanation_vs_repair.csv", index=False)

    # Replacement for Figure 3: grouped horizontal bars, readable in grayscale.
    fig, ax = plt.subplots(figsize=(COLUMN_W, 2.7))
    y = np.arange(len(BETTER_TOOL_ORDER))
    h = 0.26
    styles = [("Detection", "#1f77b4", ""), ("Explanation", "#ff7f0e", "////"), ("Repair", "#2ca02c", "....")]
    for i, (t, (label, color, hatch)) in enumerate(zip(TASKS3, styles)):
        vals = pct[t].fillna(0).values
        bars = ax.barh(y + (i - 1) * h, vals, h, label=label, color=color, hatch=hatch,
                       edgecolor="black", linewidth=0.4)
        ax.bar_label(bars, labels=[f"{v:.0f}%" for v in vals], padding=1.5, fontsize=6.5)
    ax.set_yticks(y)
    ax.set_yticklabels([wrap_label(x, 20) for x in BETTER_TOOL_ORDER])
    ax.invert_yaxis()
    ax.set_xlim(0, max(65, np.nanmax(pct.values) * 1.2))
    ax.set_xlabel("Percent of respondents")
    ax.legend(fontsize=7, frameon=False, loc="lower right")
    ax.grid(axis="x", linewidth=0.35, alpha=0.35)
    ax.set_axisbelow(True)
    fig.tight_layout(pad=0.3)
    save_readable(fig, OUT / "better_tool_by_task")


def revision_risk_figure_in_scale_order(df, col):
    """Replacement for Figure 4 with bars in scale order rather than frequency order."""
    if "risk" not in col:
        return
    tab = pct_table(df[col["risk"]]).set_index(col["risk"]).reindex(RISK_ORDER).fillna(0)
    fig, ax = plt.subplots(figsize=(COLUMN_W, 2.0))
    bars = ax.barh(RISK_ORDER, tab["percent"], height=0.62)
    ax.bar_label(bars, labels=[f"{v:.0f}%" for v in tab["percent"]], padding=2, fontsize=7.0)
    ax.set_xlabel("Percent of respondents")
    ax.invert_yaxis()
    ax.set_xlim(0, max(tab["percent"].max() * 1.18, 10))
    ax.grid(axis="x", linewidth=0.35, alpha=0.35)
    ax.set_axisbelow(True)
    fig.tight_layout(pad=0.3)
    save_readable(fig, OUT / "risk_distribution_scale_order")


def revision_explanation_challenges(df, col):
    if "explanation_challenges" in col:
        split_multiselect(
            df[col["explanation_challenges"]], known_options=MULTI_OPTIONS["explanation_challenges"]
        ).to_csv(REVISION_OUT / "R12_explanation_challenges.csv", index=False)


def revision_qual_quant_integration(df):
    """Links eligible incident codes to respondent-level survey measures.

    Requires thematic_coding_final_artifact_with_cooccurrences.xlsx next to this
    script. Anonymous ID Pxxx corresponds to row xxx-1 of the loaded dataset.
    """
    path = next((BASE / n for n in QUAL_WORKBOOK_NAMES if (BASE / n).exists()),
                BASE / QUAL_WORKBOOK_NAMES[0])
    if not path.exists():
        print("Qualitative workbook not found; skipping qual-quant integration.")
        return
    cr = pd.read_excel(path, sheet_name="Coded Responses", header=3)
    id_to_row = {a: i for i, a in zip(df.index, df["anon_id"])}
    cr["row"] = cr["Anonymous ID"].map(id_to_row)
    cr = cr.dropna(subset=["row"])
    cr["row"] = cr["row"].astype(int)
    inc = cr[(cr["Dataset"] == "Incident Responses")
             & cr["Eligibility status"].astype(str).str.startswith("Eligible")]
    codes = inc["Primary-coder code(s)"].fillna("")
    repair_fail = codes.str.contains(
        "IF_UNSAFE_FIX|IF_NEW_VULNERABILITY|IF_INCOMPLETE_FIX|IF_FUNCTIONAL_REGRESSION"
    ).values
    reporters = df.index.isin(inc["row"])
    rows = []
    for v in ["future_repair_num", "future_explanation_num", "risk_score", "verification_breadth_score"]:
        comparisons = [
            ("repair-failure incident vs other incident",
             df.loc[inc["row"][repair_fail], v], df.loc[inc["row"][~repair_fail], v]),
            ("eligible incident reporter vs non-reporter",
             df.loc[reporters, v], df.loc[~reporters, v]),
        ]
        for label, a, b in comparisons:
            a, b = a.dropna(), b.dropna()
            rows.append({"comparison": label, "variable": v, "mean_a": a.mean(), "n_a": len(a),
                         "mean_b": b.mean(), "n_b": len(b),
                         "p_MWU": stats.mannwhitneyu(a, b).pvalue})
    pd.DataFrame(rows).to_csv(REVISION_OUT / "R13_qual_quant_integration.csv", index=False)


def revision_mediation_negative_outcomes(df, n_boot=5000):
    """Negative repair outcomes -> perceived risk -> task-specific intention.

    Indirect effect a*b with percentile bootstrap CIs, estimated for repair and
    explanation, plus the bootstrap CI of their difference (moderated mediation
    by task). Cross-sectional: interpret as a consistent pattern, not causation.
    """
    import statsmodels.api as sm
    x, m = "negative_fix_outcome_score", "risk_score"
    ys = {t: f"future_{t}_num" for t in TASKS3}
    d = df[[x, m] + list(ys.values())].dropna()
    z = d.apply(_zscore)

    def paths(zz, y):
        a = sm.OLS(zz[m], sm.add_constant(zz[[x]])).fit().params[x]
        fit = sm.OLS(zz[y], sm.add_constant(zz[[x, m]])).fit()
        return a * fit.params[m], fit.params[x], a, fit.params[m]

    rng = np.random.default_rng(RANDOM_SEED)
    boots = {t: [] for t in TASKS3}
    for _ in range(n_boot):
        s = z.iloc[rng.integers(0, len(z), len(z))]
        for t, y in ys.items():
            boots[t].append(paths(s, y)[0])
    rows = []
    for t, y in ys.items():
        ind, direct, a, b = paths(z, y)
        lo, hi = np.percentile(boots[t], [2.5, 97.5])
        rows.append({"task": t, "n": len(z), "a_path": a, "b_path": b, "indirect": ind,
                     "indirect_ci_low": lo, "indirect_ci_high": hi, "direct": direct})
    for t in ["detection", "explanation"]:
        diff = np.array(boots["repair"]) - np.array(boots[t])
        point = paths(z, ys["repair"])[0] - paths(z, ys[t])[0]
        lo, hi = np.percentile(diff, [2.5, 97.5])
        rows.append({"task": f"difference repair - {t}", "n": len(z), "indirect": point,
                     "indirect_ci_low": lo, "indirect_ci_high": hi})
    pd.DataFrame(rows).to_csv(REVISION_OUT / "R14_mediation_negative_outcomes_risk.csv", index=False)


def revision_delegation_profiles(df):
    """Respondent-level delegation profiles from likely future use (ratings 4-5)."""
    f = df[[f"future_{t}_num" for t in TASKS3]].dropna()
    hi = f >= 4
    det, exp, rep = hi["future_detection_num"], hi["future_explanation_num"], hi["future_repair_num"]
    profile = np.select(
        [det & exp & rep, det & exp & ~rep, exp & ~det, ~det & ~exp & ~rep],
        ["Full delegation", "Advisory-only (withholds repair)", "Explanation-only", "Low across tasks"],
        default="Other combinations",
    )
    df.loc[f.index, "delegation_profile"] = profile
    counts = df["delegation_profile"].value_counts()
    pd.DataFrame({"n": counts, "percent": (100 * counts / counts.sum()).round(1)}).to_csv(
        REVISION_OUT / "R15_delegation_profiles.csv")

    rows = []
    a_mask = df["delegation_profile"] == "Advisory-only (withholds repair)"
    b_mask = df["delegation_profile"] == "Full delegation"
    for v in ["risk_score", "overall_performance_score", "positive_fix_outcome_score",
              "negative_fix_outcome_score", "overall_ai_usage_score",
              "verification_breadth_score", "traditional_preference_score", "experience_score"]:
        a, b = df.loc[a_mask, v].dropna(), df.loc[b_mask, v].dropna()
        if len(a) and len(b):
            rows.append({"variable": v, "mean_advisory_only": a.mean(), "n_advisory_only": len(a),
                         "mean_full": b.mean(), "n_full": len(b),
                         "p_MWU": stats.mannwhitneyu(a, b).pvalue})
    out = pd.DataFrame(rows)
    out["p_holm"] = holm_adjust(out["p_MWU"])
    out.to_csv(REVISION_OUT / "R16_advisory_only_vs_full_delegation.csv", index=False)


def revision_traditional_tools_by_task(df, col):
    """Where conventional tools stay in the loop: preference, use, and pairing by task."""
    rows = []
    for prefix in ["prefer_trad", "trad_use"]:
        s = df[[f"{prefix}_{t}_num" for t in TASKS3]].dropna()
        chi, p = stats.friedmanchisquare(*[s.iloc[:, i] for i in range(3)])
        row = {"measure": prefix, "n": len(s), "friedman_chi2": chi, "friedman_p": p}
        row.update({f"mean_{t}": s[f"{prefix}_{t}_num"].mean() for t in TASKS3})
        pairs = [("explanation", "detection"), ("explanation", "repair"), ("detection", "repair")]
        ps = [stats.wilcoxon(s[f"{prefix}_{a}_num"], s[f"{prefix}_{b}_num"]).pvalue for a, b in pairs]
        for (a, b), p_raw, p_adj in zip(pairs, ps, holm_adjust(pd.Series(ps))):
            row[f"p_{a}_vs_{b}"] = p_raw
            row[f"p_holm_{a}_vs_{b}"] = p_adj
        rows.append(row)
    pd.DataFrame(rows).to_csv(REVISION_OUT / "R17_traditional_tools_by_task.csv", index=False)

    keys = [f"trad_combo_{t}" for t in TASKS3]
    if all(k in col for k in keys):
        tabs = []
        for t, k in zip(TASKS3, keys):
            tab = split_multiselect(df[col[k]], known_options=MULTI_OPTIONS.get(k))
            tab.insert(0, "task", t)
            tabs.append(tab)
        pd.concat(tabs).to_csv(REVISION_OUT / "R18_traditional_tool_pairing_by_task.csv", index=False)
        paired = pd.DataFrame({t: df[col[k]].notna().astype(int) for t, k in zip(TASKS3, keys)})
        try:
            from statsmodels.stats.contingency_tables import cochrans_q
            q = cochrans_q(paired)
            pd.DataFrame([{**{f"pct_{t}": 100 * paired[t].mean() for t in TASKS3},
                           "n": len(paired), "cochran_q": q.statistic, "p": q.pvalue}]).to_csv(
                REVISION_OUT / "R19_any_traditional_pairing_cochran_q.csv", index=False)
        except ImportError:
            pass


def revision_role_moderation(df):
    """Does the risk-repair association differ by participant role group?"""
    import statsmodels.formula.api as smf
    if "analysis_role_group" not in df.columns:
        return
    d = df[["future_repair_num", "risk_score", "analysis_role_group"]].dropna()
    m = smf.ols("future_repair_num ~ risk_score * C(analysis_role_group)", d).fit(cov_type="HC3")
    ci = m.conf_int()
    pd.DataFrame({"term": m.params.index, "coef": m.params.values, "se": m.bse.values,
                  "ci_low": ci[0].values, "ci_high": ci[1].values, "p": m.pvalues.values,
                  "n": len(d)}).to_csv(REVISION_OUT / "R20_role_moderation_repair.csv", index=False)


QUAL_WORKBOOK_NAMES = [
    "thematic_coding_final_artifact_with_cooccurrences.xlsx",
    "thematic_coding_final_artifact.xlsx",
]
ADVISORY_STAGE_CODES = "IF_FALSE_POSITIVE|IF_FALSE_NEGATIVE|IF_SEVERITY_ERROR|IF_INCORRECT_EXPLANATION"
REMEDIATION_STAGE_CODES = "IF_UNSAFE_FIX|IF_NEW_VULNERABILITY|IF_INCOMPLETE_FIX|IF_FUNCTIONAL_REGRESSION"


def _load_qual_units():
    for name in QUAL_WORKBOOK_NAMES:
        path = BASE / name
        if path.exists():
            cr = pd.read_excel(path, sheet_name="Coded Responses", header=3)
            eligible = cr["Eligibility status"].astype(str).str.startswith("Eligible")
            return cr[eligible].copy()
    return None


def revision_failure_stage_analysis():
    """Workflow stage of reported failures and the consequences they carried.

    Failure-mode codes are grouped by the workflow stage their frozen codebook
    definitions refer to: advisory-stage failures (false positive, missed
    vulnerability, severity error, incorrect explanation) and remediation-stage
    failures (the remediation theme's failure codes). Consequence codes are then
    compared across stages.
    """
    units = _load_qual_units()
    if units is None:
        print("Qualitative workbook not found; skipping failure-stage analysis.")
        return
    inc = units[units["Dataset"] == "Incident Responses"]
    codes = inc["Primary-coder code(s)"].fillna("")
    adv = codes.str.contains(ADVISORY_STAGE_CODES)
    rem = codes.str.contains(REMEDIATION_STAGE_CODES)
    n = len(inc)
    stage = pd.DataFrame([
        {"stage": "Remediation-stage failure", "n": int(rem.sum())},
        {"stage": "Advisory-stage failure (detection/explanation)", "n": int(adv.sum())},
        {"stage": "Both", "n": int((adv & rem).sum())},
        {"stage": "Neither (e.g., outdated guidance or consequence-only)", "n": int((~adv & ~rem).sum())},
    ])
    stage["percent_of_incidents"] = (100 * stage["n"] / n).round(1)
    stage["eligible_incidents"] = n
    stage.to_csv(REVISION_OUT / "R21_failure_stage.csv", index=False)

    rows = []
    rem_only, adv_only = rem & ~adv, adv & ~rem
    for label, pattern in [
        ("Realized harm (adverse outcome or operational disruption)", "IR_ACTUAL_HARM|IR_OPERATIONAL_IMPACT"),
        ("Wasted effort or unnecessary remediation", "IR_WASTED_EFFORT"),
        ("Independent verification", "IR_VERIFICATION"),
        ("Harm prevented before adoption", "IR_PREVENTED_HARM"),
        ("Missing contextual knowledge", "IC_MISSING_CONTEXT"),
    ]:
        hit = codes.str.contains(pattern)
        a, b = int((rem_only & hit).sum()), int((adv_only & hit).sum())
        table = [[a, int(rem_only.sum()) - a], [b, int(adv_only.sum()) - b]]
        rows.append({
            "consequence_or_condition": label,
            "remediation_only_n": int(rem_only.sum()), "remediation_only_hits": a,
            "remediation_only_pct": 100 * a / max(rem_only.sum(), 1),
            "advisory_only_n": int(adv_only.sum()), "advisory_only_hits": b,
            "advisory_only_pct": 100 * b / max(adv_only.sum(), 1),
            "fisher_exact_p": stats.fisher_exact(table)[1],
        })
    pd.DataFrame(rows).to_csv(REVISION_OUT / "R22_consequences_by_failure_stage.csv", index=False)

    ref = units[units["Dataset"] == "General Reflections"]
    rc = ref["Primary-coder code(s)"].fillna("")
    benefit = rc.str.contains("GR_PRODUCTIVITY|GR_LEARNING|GR_TASK_STRENGTH")
    safeguard = rc.str.contains("GR_CONDITIONAL_TRUST|GR_VERIFICATION|GR_HUMAN_OVERSIGHT")
    productivity = rc.str.contains("GR_PRODUCTIVITY")
    pd.DataFrame([{
        "eligible_reflections": len(ref),
        "benefit_reflections": int(benefit.sum()),
        "benefit_with_safeguard": int((benefit & safeguard).sum()),
        "productivity_reflections": int(productivity.sum()),
        "productivity_with_safeguard": int((productivity & safeguard).sum()),
        "safeguard_reflections": int(safeguard.sum()),
    }]).to_csv(REVISION_OUT / "R23_benefits_coupled_with_safeguards.csv", index=False)


def revision_coding_reliability_summary():
    """Summaries of calibration and formal intercoder reliability."""
    rel_path = BASE / "Final_Reliability_coding.xlsx"
    if rel_path.exists():
        ra = pd.read_excel(rel_path, sheet_name="Reliability Analysis", header=7)
        ra = ra[ra["Dataset"].isin(["Incident Responses", "General Reflections"])].copy()
        ra["Cohen's kappa"] = pd.to_numeric(ra["Cohen's kappa"], errors="coerce")
        rows = []
        for label, sub in [("All codes", ra)] + list(ra.groupby("Dataset")):
            k = sub["Cohen's kappa"]
            rows.append({"set": label, "codes": len(sub), "kappa_mean": k.mean(),
                         "kappa_median": k.median(), "kappa_min": k.min(),
                         "codes_kappa_ge_0.80": int((k >= 0.80).sum()),
                         "codes_kappa_eq_1": int((k >= 0.9999).sum()),
                         "min_observed_agreement": pd.to_numeric(sub["Observed agreement"]).min()})
        pd.DataFrame(rows).to_csv(REVISION_OUT / "R24_intercoder_reliability_summary.csv", index=False)
        ra.to_csv(REVISION_OUT / "R24b_intercoder_reliability_by_code.csv", index=False)

    cal_path = BASE / "calibration_codes_side_by_side.xlsx"
    if cal_path.exists():
        cal = pd.read_excel(cal_path, sheet_name="Calibration Reconciliation", header=3).dropna(subset=["Dataset"])
        split = lambda x: set(t.strip() for t in str(x).split(";") if t.strip() and str(x) != "nan")
        decisions = disagreements = 0
        for _, r in cal.iterrows():
            n_codes = 17 if str(r["Dataset"]).startswith("Incident") else 12
            decisions += n_codes
            disagreements += len(split(r["Coder A assigned codes"]) ^ split(r["Coder B assigned codes"]))
        pd.DataFrame([{"calibration_units": len(cal), "binary_decisions": decisions,
                       "disagreements": disagreements,
                       "observed_agreement": 1 - disagreements / decisions}]).to_csv(
            REVISION_OUT / "R25_calibration_agreement.csv", index=False)


def run_revision_analyses(df, col):
    REVISION_OUT.mkdir(exist_ok=True)
    _add_revision_columns(df, col)
    revision_sample_description(df, col)
    revision_task_risk_robustness(df)
    revision_tool_allocation(df, col)
    revision_risk_figure_in_scale_order(df, col)
    revision_explanation_challenges(df, col)
    revision_qual_quant_integration(df)
    revision_failure_stage_analysis()
    revision_coding_reliability_summary()
    try:
        import statsmodels  # noqa: F401
    except ImportError:
        print("statsmodels not installed; skipping R07-R09 (pip install statsmodels).")
        return
    revision_task_risk_models(df)
    revision_regression_sensitivity(df)
    revision_mediation_negative_outcomes(df)
    revision_delegation_profiles(df)
    revision_traditional_tools_by_task(df, col)
    revision_role_moderation(df)


# ------------------------- ENTRY POINT -------------------------

# ------------------------- OUTPUT ORGANIZATION -------------------------
#
# After all analyses run, results are sorted into folders that follow the
# structure of the paper, so each section's results are easy to locate.

OUTPUT_LAYOUT = [
    ("sample_and_measures", [r"^R0[1-3]_", r"^descriptive_tables\.xlsx$", r"^composite_score_summary",
                             r"^composite_reliability_cronbach_alpha", r"^role_group_counts"]),
    ("rq1_tool_use", [r"^R1[0-1]_", r"^R1[7-9]_", r"^crosstab_", r"^chi_square_tests",
                      r"^sdlc_context_group_comparisons"]),
    ("rq2_performance_and_risk", [r"^wilcoxon_", r"^spearman_risk_correlations", r"^kruskal_experience_tests"]),
    ("rq3_verification", [r"^R13_", r"^role_verification_method_comparisons", r"^role_group_kruskal_tests",
                          r"^role_group_outcome_summaries"]),
    ("rq4_failures", [r"^R12_", r"^R2[1-3]_", r"^explanation_challenge_group_comparisons"]),
    ("rq5_reliance", [r"^R0[4-9]_", r"^R1[4-6]_", r"^R20_", r"^future_intention_", r"^task_specific_",
                      r"^composite_correlation_(matrix|tests)", r"^deep_spearman_", r"^quality_outcome_",
                      r"^regression_"]),
    ("reliability", [r"^R2[45]"]),
    ("supplementary", [r"^respondent_personas"]),
]


def organize_outputs():
    import os
    import re
    import shutil

    sources = [OUT, OUT / "revision"]
    moved = 0
    for src in sources:
        if not src.exists():
            continue
        for path in list(src.iterdir()):
            if not path.is_file():
                continue
            name = path.name
            if name.lower().endswith(".png"):
                folder = "figures"
            else:
                folder = next((f for f, pats in OUTPUT_LAYOUT if any(re.search(p, name) for p in pats)),
                              "other")
            dest = OUT / folder
            dest.mkdir(parents=True, exist_ok=True)
            os.replace(path, dest / name)
            moved += 1
    rev = OUT / "revision"
    if rev.exists() and not any(rev.iterdir()):
        shutil.rmtree(rev)
    print(f"Organized {moved} output files into folders under {OUT.resolve()}")


def main():
    warnings.filterwarnings("ignore", category=UserWarning)

    df, input_file = load_dataset()
    if df is not None:
        col = get_column_map(df)
        add_numeric_columns(df, col)
        add_analysis_role_group(df, col)

        # Aggregate tables, statistics, and figures are written to artifact_outputs/.

        add_composite_scores(df, col)
        run_main_analysis(df, col)
        run_deeper_analysis(df, col)
        organize_outputs()

    print(f"Done. Results written to: {OUT.resolve()}")


if __name__ == "__main__":
    main()

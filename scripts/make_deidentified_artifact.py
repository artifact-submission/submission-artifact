#!/usr/bin/env python3
"""
Create the public, de-identified inputs for the replication artifact.

What is removed or masked:
  * IP address, latitude/longitude, recipient name/email, external reference,
    Qualtrics response ID, start/end/recorded timestamps
  * Prolific participant IDs (replaced by a constant token so the recruitment
    channel can still be derived) and Prolific study and session IDs
  * All free-text answers (open-ended questions and "Other (please specify)")
  * Participant response text in the qualitative workbooks (codes, themes, and
    eligibility decisions are kept)
"""
from pathlib import Path
import pandas as pd

BASE = Path(__file__).resolve().parent
OUT = BASE / "public_artifact_inputs"
OUT.mkdir(exist_ok=True)

DROP_EXACT = {
    "IPAddress", "LocationLatitude", "LocationLongitude", "RecipientLastName",
    "RecipientFirstName", "RecipientEmail", "ExternalReference", "ResponseId",
    "StartDate", "EndDate", "RecordedDate",
    "STUDY_ID", "SESSION_ID",  # Prolific study and session identifiers
    "QID164",  # incident open-ended question
    "QID158",  # general reflection open-ended question
}

# ---- 1. Survey responses ----------------------------------------------------
raw = pd.read_excel(BASE / "cleaned_responses.xlsx")
def looks_like_metadata(row):
    text = " ".join(row.dropna().astype(str).head(30))
    return ("ImportId" in text) or ("Source file name" in text) or ("Start Date" in text)
meta_mask = raw.apply(looks_like_metadata, axis=1)
df = raw.loc[~meta_mask].copy().reset_index(drop=True)
df.insert(0, "anon_id", [f"P{i + 1:03d}" for i in range(len(df))])

text_cols = [c for c in df.columns if str(c).endswith("_TEXT")]
drop = [c for c in df.columns if c in DROP_EXACT] + text_cols
df = df.drop(columns=drop)
if "PROLIFIC_PID" in df.columns:
    df["PROLIFIC_PID"] = df["PROLIFIC_PID"].where(df["PROLIFIC_PID"].isna(), "REDACTED_PROLIFIC_ID")
df.to_csv(OUT / "analysis_dataset_deidentified.csv", index=False)
print(f"Survey: kept {df.shape[1]} columns for {len(df)} rows; removed {len(drop)} columns.")

# ---- 2. Qualitative coding (codes and themes, no response text) --------------
def write_like_original(frame, path, sheet, notes):
    with pd.ExcelWriter(path) as xw:
        pd.DataFrame([[n] for n in notes]).to_excel(xw, sheet_name=sheet, index=False, header=False)
        frame.to_excel(xw, sheet_name=sheet, index=False, startrow=3)

qual = BASE / "thematic_coding_final_artifact.xlsx"
if qual.exists():
    cr = pd.read_excel(qual, sheet_name="Coded Responses", header=3)
    cr = cr.drop(columns=[c for c in cr.columns if "text" in str(c).lower()])
    write_like_original(cr, OUT / "thematic_coding_final_artifact.xlsx", "Coded Responses",
                        ["Response-level coding decisions (response text removed for privacy).", "", ""])
    for sheet in ["Codebook", "Code Frequencies", "Theme Summary"]:
        try:
            pd.read_excel(qual, sheet_name=sheet, header=None).to_csv(
                OUT / f"qualitative_{sheet.lower().replace(' ', '_')}.csv", index=False, header=False)
        except ValueError:
            pass

rel = BASE / "Final_Reliability_coding.xlsx"
if rel.exists():
    ra = pd.read_excel(rel, sheet_name="Reliability Analysis", header=None)
    with pd.ExcelWriter(OUT / "Final_Reliability_coding.xlsx") as xw:
        ra.to_excel(xw, sheet_name="Reliability Analysis", index=False, header=False)

cal = BASE / "calibration_codes_side_by_side.xlsx"
if cal.exists():
    c = pd.read_excel(cal, sheet_name="Calibration Reconciliation", header=3)
    c = c.drop(columns=[x for x in c.columns if "text" in str(x).lower()])
    write_like_original(c, OUT / "calibration_codes_side_by_side.xlsx", "Calibration Reconciliation",
                        ["Calibration codes (response text removed for privacy).", "", ""])

# ---- 3. Safety check ---------------------------------------------------------
survey = pd.read_csv(OUT / "analysis_dataset_deidentified.csv", low_memory=False)
leaks = [c for c in survey.columns if c in DROP_EXACT or str(c).endswith("_TEXT")]
assert not leaks, f"Identifying columns still present: {leaks}"
print("De-identified inputs written to", OUT)

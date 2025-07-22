import pandas as pd
import numpy as np
from datetime import timedelta
import random

# Load Synthea CSVs
patients = pd.read_csv("../data-collection/patients.csv")
encounters = pd.read_csv("../data-collection/encounters.csv")
procedures = pd.read_csv("../data-collection/procedures.csv")
providers = pd.read_csv("../data-collection/providers.csv")

# Filter imaging procedures
imaging_keywords = ["CT", "MRI", "X-Ray", "Xray", "Radiology"]
imaging_procedures = procedures[procedures["DESCRIPTION"].str.contains('|'.join(imaging_keywords), case=False, na=False)].copy()

# Limit to 10,000 studies
imaging_procedures = imaging_procedures.sample(n=min(10000, len(imaging_procedures)), random_state=42)

# Add synthetic timestamps
def generate_timestamps(start_time):
    # Total TAT gap: 20.6 hours, broken down into 5 stages
    stage_durations = {
        "acquisition_complete": 3.5,
        "triage_delay": 8.0,
        "interpretation": 4.0,
        "finalization": 2.5,
        "distribution": 2.6
    }
    times = {}
    times["scan_time"] = start_time
    times["acquisition_complete"] = times["scan_time"] + timedelta(hours=stage_durations["acquisition_complete"])
    times["triage_start"] = times["acquisition_complete"]
    times["triage_end"] = times["triage_start"] + timedelta(hours=stage_durations["triage_delay"])
    times["interpretation_start"] = times["triage_end"]
    times["interpretation_end"] = times["interpretation_start"] + timedelta(hours=stage_durations["interpretation"])
    times["finalization_start"] = times["interpretation_end"]
    times["finalization_end"] = times["finalization_start"] + timedelta(hours=stage_durations["finalization"])
    times["distribution_time"] = times["finalization_end"] + timedelta(hours=stage_durations["distribution"])
    return times

# Dataset 1: Imaging Requests
df_requests = imaging_procedures.merge(encounters, on="ENCOUNTER", suffixes=('_proc', '_enc'))[[
    "Id_proc", "PATIENT", "ENCOUNTER", "DESCRIPTION", "START_proc"
]].rename(columns={"START_proc": "request_time"})

# Dataset 2: Radiology Report Log (Workflow timestamps)
workflow_rows = []
for _, row in df_requests.iterrows():
    base_time = pd.to_datetime(row["request_time"])
    times = generate_timestamps(base_time)
    workflow_rows.append({
        "procedure_id": row["Id_proc"],
        "scan_time": times["scan_time"],
        "acquisition_complete": times["acquisition_complete"],
        "triage_start": times["triage_start"],
        "triage_end": times["triage_end"],
        "interpretation_start": times["interpretation_start"],
        "interpretation_end": times["interpretation_end"],
        "finalization_start": times["finalization_start"],
        "finalization_end": times["finalization_end"],
        "distribution_time": times["distribution_time"]
    })

df_workflow = pd.DataFrame(workflow_rows)

# Dataset 3: Radiologist Assignment (simulate radiologist from providers)
radiologists = providers[providers["SPECIALITY"].str.contains("radiology", case=False, na=False)]
if radiologists.empty:
    radiologists = providers.sample(n=100, random_state=42).copy()
radiologist_ids = radiologists["Id"].values

df_assignments = df_requests[["Id_proc"]].copy()
df_assignments["radiologist_id"] = np.random.choice(radiologist_ids, size=len(df_assignments))
df_assignments["role"] = "Radiologist"
df_assignments["location"] = np.random.choice(["General Hospital", "Diagnostic Centre", "Clinic A"], size=len(df_assignments))

# Save datasets
df_requests.to_csv("radiology_requests.csv", index=False)
df_workflow.to_csv("radiology_workflow_log.csv", index=False)
df_assignments.to_csv("radiologist_assignments.csv", index=False)

print("✅ Datasets generated:\n- radiology_requests.csv\n- radiology_workflow_log.csv\n- radiologist_assignments.csv")
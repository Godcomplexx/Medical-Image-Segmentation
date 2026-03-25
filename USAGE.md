# Usage Guide

This document describes how to use the web interface after the service is up and running.

## 1. Authentication

1. Open the application in your browser (default: `http://localhost`).
2. Sign in with your email and password, or click **"No account? Register"** to create one.
3. After successful login you will be redirected to the Dashboard.

## 2. Dashboard

The main dashboard provides an overview of all patients and studies.

- **Statistics cards** — total patients, total studies, results ready for review.
- **Search** — filter by patient ID or study name.
- **Status tabs** — filter studies by status: `All`, `Uploaded`, `Queued`, `Processing`, `Ready`.
- **Patient table** — each row shows patient ID, last update date, and number of studies.

## 3. Adding a New Study

1. Click **"+ Add Study"** on the dashboard.
2. In the dialog:
   - **Patient ID** — leave empty to create a new patient (ID is auto-generated), or paste an existing patient ID.
   - **Study archive** — select a `.zip` file containing DICOM data.
3. Click **"Add Study"** to upload.

## 4. Working with Studies

Click **"Show"** next to a patient to expand the list of their studies. Each study has the following actions:

| Action | Description | Available when |
|--------|-------------|----------------|
| **Analyze** | Send the study for ML processing | `Uploaded` |
| **Results** | View segmentation results (localization, classification, area) | `Ready` |
| **Download** | Download result files (T1, T2, prediction, parcellation) | `Ready` |
| **Delete** | Remove the study (requires confirmation) | Any status |

## 5. Patient Management

- **Copy ID** — click the patient ID to copy it to clipboard (useful for adding studies to the same patient).
- **Delete patient** — removes the patient and **all associated studies** (requires confirmation).

## 6. Logout

Click your email in the top-right corner and select **"Logout"**.

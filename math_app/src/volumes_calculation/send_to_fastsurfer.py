import httpx
from pathlib import Path

def send_to_fastsurfer_api(t1_path: Path, subject_id: str, license_path: Path, output_dir: Path, fastsurfer_url: str):
    data = {
        "t1_path": str(t1_path),
        "subject_id": subject_id,
        "license_path": str(license_path),
        "output_dir": str(output_dir)
    }
    try:
        response = httpx.post(f"{fastsurfer_url}/run_fastsurfer/", data=data, timeout=10800)
        return response.json()
    except Exception as e:
        return {"status": "error", "detail": str(e)}


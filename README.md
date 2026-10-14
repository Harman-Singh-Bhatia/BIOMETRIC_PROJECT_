# BIOMM 2FA — Biometric Two-Factor Authentication

BIOMM 2FA is a demonstration application that combines air-signature
verification with fingerprint matching. The web interface is served by a
FastAPI backend; the accompanying Jupyter notebook contains the research and
prototype workflow.

## Project contents

```text
BIOMM_2FA_Submission/
├── app.py                         # FastAPI application
├── biometrics/                    # Signature, fingerprint, and registry logic
├── static/                        # Browser interface
├── Fingerprint/                   # Add an authorized image dataset here
├── BIOMM_2FA_PROJECT.ipynb         # Clean notebook copy (outputs removed)
├── test_api.py                    # End-to-end API smoke test
├── requirements.txt
└── README.md
```

## Setup

Use Python 3.10 or newer in a virtual environment, then install dependencies:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

The demo needs fingerprint images to complete signup and login. For privacy,
the source fingerprint dataset and generated user/enrollment records are not
included in this submission. Add a dataset you are authorized to use under
`Fingerprint/`, following the filename pattern expected by the application
(for example, `fingerprint1.png`, `fingerprint2.png`). Then start the server:

```powershell
uvicorn app:app --reload
```

Open <http://127.0.0.1:8000> in a browser and allow camera access. The browser
loads MediaPipe and fonts from public CDNs, so an internet connection is
required for those assets.

## Notebook

Open `BIOMM_2FA_PROJECT.ipynb` in Jupyter or VS Code. Core notebook packages
are listed in `requirements.txt`. For notebook webcam/hand-tracking features,
install `mediapipe` separately. TensorFlow is optional; the notebook falls
back when it is unavailable.

## API smoke test

With the server running and a fingerprint dataset configured, run:

```powershell
python test_api.py
```

This test creates a demo user and enrollment records in the local runtime data
directory. Run it only against a disposable local instance.

## Data and limitations

- Do not add real biometric samples, enrollment records, `.env` files, or
  personal identifiers to a submission unless you have permission and the
  submission rules explicitly require them.
- Runtime data is created under `biometric_2fa_data/` when the application
  runs; this generated data is intentionally excluded from the ZIP.
- This is an academic/demo implementation, not a production authentication
  system. Do not use it to protect real accounts or sensitive information.
- Camera access and the browser-hosted hand-tracking assets are required for
  the air-signature interface.

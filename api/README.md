# SANTRAZEST inference API

This is an inference-service scaffold. It does not include model weights and will not return fabricated detections.

## 1. Create an environment

Use a supported Python version (3.10–3.13 recommended for the current Ultralytics stack).

```powershell
cd api
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## 2. Configure the trained model

Set the path to the actual trained weights file. Do not commit the weights to the public repository.

```powershell
$env:MODEL_PATH="C:\path\to\your\best.pt"
$env:TARGET_CLASS_NAME="Fruit on Tree"
$env:CONFIDENCE="0.25"
$env:CORS_ORIGINS="http://localhost:5500,http://127.0.0.1:5500"
```

The target class must exactly match a class name in the trained model. Check the model's class names before relying on counts. If the training labels differ, change TARGET_CLASS_NAME accordingly.

## 3. Start the API

```powershell
python -m uvicorn main:app --host 127.0.0.1 --port 8000 --reload
```

Open http://127.0.0.1:8000/health. The API reports not_ready until a valid model path is configured and the model loads successfully.

## 4. Test prediction

Use the interactive API docs at http://127.0.0.1:8000/docs and upload one orchard image to POST /predict.

The response count includes only detections matching TARGET_CLASS_NAME and the configured confidence threshold. Review the returned bounding boxes against the photo before presenting results.

## Important limitations

- This is a local prototype scaffold, not a hosted production service.
- No user accounts, image persistence, database, orchard history or remote upload storage are implemented.
- A single photo is not a complete 360-degree tree count. Multiple views need a separate deduplication/tracking design.
- Validate precision and recall on held-out orchard images before using counts in a demonstration.
- Before public deployment, configure exact HTTPS origins, authentication, request throttling and privacy/data-retention controls.

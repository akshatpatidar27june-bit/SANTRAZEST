# SANTRAZEST Model Integration Plan

## Goal
Connect the Farmer Portal photo workflow to a real citrus-fruit detection model without displaying invented counts.

## Recommended architecture

### Stage 1 — Local exhibition prototype
Run the website and inference API together on the development laptop:
- **Frontend:** existing SANTRAZEST static website and Farmer Portal.
- **API:** a small Python FastAPI service.
- **Model runtime:** trained object-detection model, loaded once when the API starts.
- **Input:** one orchard image sent as multipart form data.
- **Output:** detected fruit count, confidence values, bounding boxes, processing time and model identifier.
- **Storage:** no permanent image or farmer-profile storage in the first prototype.

For a phone demonstration on the same Wi-Fi network, serve both the website and API from the laptop and open the laptop's local network address on the phone. This avoids calling an insecure local HTTP API from the hosted HTTPS website.

### Stage 2 — Hosted pilot
After the model is accurate and the prototype is tested:
- Deploy the API and model to a suitable server with enough memory and CPU/GPU capacity.
- Configure the website to call the HTTPS API endpoint.
- Add authentication, upload limits, rate limits, privacy notices and controlled data retention before accepting real farmer information.
- Monitor latency, failures and model quality.

Do not put model weights or private API keys in the browser or public repository.

## API contract

### GET /health
Returns service readiness and model-loading status. A healthy API response does not prove that the model is accurate.

Example response:
```json
{
  "status": "ok",
  "model_loaded": true,
  "model_name": "citrus-detector",
  "model_version": "0.1.0"
}
```

### POST /predict
- Content type: multipart/form-data
- Field: file (one image)
- Accept supported image formats only and enforce a server-side file-size limit.
- Validate the decoded image before inference.

Example response:
```json
{
  "status": "success",
  "count": 12,
  "detections": [
    {
      "class_name": "orange",
      "confidence": 0.91,
      "box": [120, 80, 188, 154]
    }
  ],
  "inference_ms": 184.2,
  "model_name": "citrus-detector",
  "model_version": "0.1.0"
}
```
The values above are illustrative schema examples only, not actual model results.

## Important counting rules
1. Count only detections belonging to the configured target-fruit class.
2. Keep the confidence threshold configurable and validate it on a held-out test set.
3. Return bounding boxes so the farmer can visually review detections.
4. Do not describe one photograph as a complete 360-degree tree count. Multiple views can contain duplicate fruit; cross-view tracking/deduplication is a separate feature.
5. Do not estimate weight, revenue or profit until those calculations have a validated method and user-supplied market inputs.
6. Display an honest error when the model is unavailable; never substitute a sample count.

## Suggested repository layout
```text
SANTRAZEST/
  index.html
  farmer-portal.html
  SANTRAZEST.png
  api/
    main.py
    requirements.txt
    README.md
  model/
    README.md
```
Keep large model-weight files out of Git unless there is a deliberate, safe hosting strategy. Document where trained weights are stored locally and how to configure their path.

## Integration sequence
1. Confirm the trained model format, target class names and local weights path.
2. Build and test the FastAPI health endpoint.
3. Load the real model and implement image validation plus POST /predict.
4. Test predictions against held-out orchard photos and review false positives/false negatives.
5. Connect the Farmer Portal upload flow to the API and render returned boxes/count.
6. Only then plan hosted inference and account/data persistence.

## Current status
This document defines the integration design only. No inference API, model weights, cloud processing, account system or persistent farmer records are created by this plan.

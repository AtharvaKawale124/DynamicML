# DynamicML

A B.Tech final-year project prototype providing a Django interface for collecting webcam images, preparing a dataset, training a YOLOv8 model, testing it locally, and downloading its weights with an inference script.

## Implemented workflow

1. Register and sign in.
2. Create an image-training project with a unique model name.
3. Add a class and capture webcam frames. The current capture routine attempts to collect 100 images after a countdown, resizing frames to 224 × 224.
4. Prepare an 80/20 training/validation split and YOLO-format labels.
5. Fine-tune pretrained YOLOv8n weights for 10 epochs by default.
6. Test using a local OpenCV webcam window; press `q` to exit.
7. Download a ZIP containing `best.pt` and `image_classification.py` for local inference.

This repository contains the Django image-training application. Earlier CSV, video, and image experiments exist in the project archive but are not represented here as working web features. The `train_model` endpoint is a placeholder, not a separate implemented training pipeline.

## Local setup

The following instructions are reconstructed from the source. A complete installation and webcam/training run have **not yet been verified**. There was no original dependency lockfile. Python 3.10 or 3.11 is a candidate environment; the repository preparation machine has Python 3.14, which has not been validated for this stack.

Run from this repository's root in PowerShell:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"
Move-Item -LiteralPath .\yolov8n.pt -Destination .\yolo_model\yolov8n.pt
python manage.py migrate
python manage.py runserver 127.0.0.1:8000
```

The weight download and package installation require internet access. Open `http://127.0.0.1:8000/`, register a new account, and create an image project. Use simple, unique alphanumeric project and class names. The original user database and captured datasets are deliberately excluded.

Webcam capture and testing run on the **computer hosting Django**, using native OpenCV windows. Run the server and browser on the same computer and close any browser camera preview before starting server-side capture. This is a local desktop demonstration, not a hosted camera service.

For a downloaded model, extract its ZIP, install `ultralytics` and `opencv-python`, and run `python image_classification.py` from the directory containing `best.pt`.

## Code map

| Location | Purpose |
| --- | --- |
| `app/views.py` | Authentication, project creation, capture/train/test/download handlers |
| `app/image_process.py` | Capture, dataset preparation, YOLO training, and older CNN helpers |
| `templates/` and `static/` | Django pages, styling, JavaScript, and bundled UI assets |
| `Training/` | Django configuration and routing |
| `codes/image_classification.py` | Inference script packaged with model downloads |
| `app/migrations/` | Database schema; includes an older CSV model record |

## Limitations

- Dataset preparation assigns a **full-frame bounding box** to each image. It does not automatically locate objects or generate accurate object-level annotations. The selected COCO class is used in the folder name, not for detector-based annotation.
- Adjacent webcam frames can be very similar. A random split can overestimate generalisation; independently recorded test data is needed.
- Training blocks a web request. There is no background task queue, progress reporting, or multi-user isolation of generated files.
- Model testing/download assumes the output path `runs/detect/train/weights/best.pt`; repeated runs can produce different training directory names.
- Several older CNN/OOD helper functions remain in the source and are not part of the active YOLO web workflow.
- Input paths, project ownership, camera failures, and training errors need further handling. Use only locally with trusted input. Development settings are not suitable for public deployment.
- No accuracy or performance result is asserted in this repository, and no full runtime verification has been completed.

## Repository preparation and attribution

The application was recovered from the extracted `teachable_machine` project folder. Its Python source matches the main ZIP backup except for an additional YOLO import in the extracted `views.py`. This copy excludes account databases, personal capture datasets, generated models, caches, videos, and backup ZIPs.

Preparation changes: this README, an inferred dependency list, ignore rules, an environment-generated Django secret, and corrected template/static directory settings. The original archive remains separate.

The project uses Django, Ultralytics YOLO, OpenCV, TensorFlow/Keras, scikit-learn, NumPy, Matplotlib, Bootstrap, jQuery, and other bundled UI resources. Existing third-party notices are retained. YOLO weights are downloaded separately. Check the applicable licenses before redistribution or deployment. An original team/contributor list and the precise origin of the UI template still need confirmation; repository ownership does not establish sole authorship.

# Dataset Creation from 3D Pokémon Assets

This repository contains code and resources used to generate synthetic datasets from 3D Pokémon assets (via Blender), convert COCO annotations to YOLO format, augment the data, train YOLO models (yolo11n / yolo11s), and run inference to produce labeled images and text files. The goal was to create a "Pokédetector" able to detect Pokémon in screenshots and use that detector for further dataset creation.

Key components:
- Dataset generation from Blender assets (`blender_script.py`).
- Data augmentation and COCO → YOLO conversion (notebook: `yolo_training.ipynb`, dataset conversion cells).
- Model training (using `ultralytics` YOLO in `yolo_training.ipynb`).
- Inference and dataset creation using the trained `Pokédetector` model (in the notebook).

Repository layout (most relevant files/folders):

- `blender_script.py`: Blender script used to render 3D assets into images (run with Blender in background).
- `yolo_training.ipynb`: Jupyter notebook that orchestrates dataset conversion, augmentation, training, memory/GPU checks and inference for dataset creation.
- `datasets/`: source datasets and converted YOLO datasets.
- `downloaded_models/`: pre-downloaded base model weights such as `yolo11n.pt`, `yolo11s.pt`.
- `trained_models/`: saved training outputs. A final model produced is `trained_models/last_model/pokedetector.pt`.
- `final_dataset/`: folder used for running the detector on a set of original images and collecting predictions.
- `models_results/` and `trained_models/`: project output from training runs and organized results.

Quick overview of workflows

1) Create images from 3D models (Blender)

   - Use `blender_script.py` to render many views of 3D Pokémon assets.
   - Example (run from repository root):

```powershell
blender --background --python "blender_script.py"
```

2) Convert COCO annotations to YOLO, augment, and split

   - The notebook `yolo_training.ipynb` contains a section that reads a COCO annotation file, converts annotations to YOLO format, creates `dataset.yaml` files, performs augmentations via `albumentations`, and writes augmented images and labels under `datasets/`.

3) Train YOLO models

   - Training is implemented using the `ultralytics` package inside the notebook. Typical settings used in the notebook: `imgsz=640`, `epochs=50`, `batch=16`, and device selection of GPU if available.

   - Trained outputs are saved under `trained_models/` and the notebook saves a final `pokedetector.pt` for inference.

4) Create a new dataset from images using the trained Pokédetector

   - The notebook includes an inference section that loads `trained_models/last_model/pokedetector.pt` and runs `model.predict(...)` on images in `final_dataset/original_images`, saving annotated images and `.txt` predictions.

Acknowledgements

- Built with `ultralytics` YOLO and `albumentations` for augmentation.
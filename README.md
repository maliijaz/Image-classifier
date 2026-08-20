# Image-classifier

A binary image classifier built with a Convolutional Neural Network (CNN) in TensorFlow/Keras that labels images as **happy** or **sad**.

## How it works

The notebook (`image_classification.ipynb`):

1. Loads images from a local `data` directory (one subfolder per class) and validates file types/extensions with `imghdr`, discarding anything not a valid image.
2. Builds a `tf.keras.Sequential` CNN with three `Conv2D` + `MaxPooling2D` blocks, followed by `Flatten` and `Dense` layers, ending in a single sigmoid output unit for binary classification.
3. Compiles the model with the Adam optimizer and `binary_crossentropy` loss, tracking accuracy.
4. Trains for 15 epochs, logging metrics via a TensorBoard callback.
5. Evaluates the model using `Precision`, `Recall`, and `BinaryAccuracy`.
6. Runs predictions on new images (e.g. `happytest.jpg`, `sadtest.jpg`) by resizing and normalizing them before calling `model.predict`.
7. Saves the trained model to `models/happysad_image_classifier.h5` and demonstrates reloading it with `load_model` for inference.

## Project structure

- `image_classification.ipynb` — full pipeline: data loading/cleaning, CNN definition, training, evaluation, and inference.
- `data/` — training images, organized by class (not tracked in git — see `.gitignore`).
- `models/` — saved trained model(s) (`.h5`).
- `logs/` — TensorBoard training logs.
- `happytest.jpg`, `sadtest.jpg` — sample images used to test the trained model.

## Requirements

- Python 3
- `tensorflow`
- `opencv-python` (`cv2`)
- `matplotlib`
- `numpy`

## Setup & usage

```bash
pip install tensorflow opencv-python matplotlib numpy
jupyter notebook image_classification.ipynb
```

Populate `data/happy` and `data/sad` folders with training images before running the notebook, then run all cells to train and evaluate the model. Trained weights are saved under `models/`.

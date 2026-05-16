# Project 4: Image and Text Recognition

A professional implementation of a dual-pipeline system for object detection using YOLOv3-tiny and optical character recognition (OCR) using Tesseract. This project fulfills the basic requirements of using pre-trained models and simple libraries to perform recognition on sample inputs and displaying results clearly.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Code Structure](#code-structure)
- [Results](#results)
- [Customization](#customization)
- [Author](#author)

## Overview

This project demonstrates two fundamental computer vision tasks:

1. **Object Detection**  
   Utilizes the pre-trained YOLOv3-tiny model (trained on the COCO dataset) to detect objects in an input image. The detection pipeline includes confidence filtering and non-maximum suppression to produce clean bounding boxes.

2. **Optical Character Recognition (OCR)**  
   Employs the Tesseract OCR engine to extract text from images. A preprocessing step (grayscale, Gaussian blur, adaptive thresholding) enhances recognition accuracy.

The code is written in Python and is fully executable in Google Colab or any local environment with the required dependencies.

## Features

- Pre-trained YOLOv3-tiny model for real-time object detection.
- Configurable confidence threshold (default 0.8) to filter low-confidence detections.
- Non-Maximum Suppression (NMS) to remove redundant bounding boxes.
- Image preprocessing pipeline for OCR: grayscale conversion, Gaussian blur, adaptive thresholding.
- Support for multiple languages in OCR (default English).
- Clear visualization using Matplotlib and console output.
- Modular functions for easy integration and customization.

## Requirements

- Python 3.7 or higher
- OpenCV
- NumPy
- Matplotlib
- pytesseract
- Pillow (PIL)
- Tesseract OCR engine installed on the system (or automatically installed in Colab)

The following files are downloaded automatically during execution:
- `yolov3-tiny.cfg`
- `yolov3-tiny.weights`
- `coco.names`

## Installation

### Local Environment

1. Install Tesseract OCR:
   - Ubuntu/Debian: `sudo apt install tesseract-ocr`
   - macOS: `brew install tesseract`
   - Windows: Download installer from [GitHub UB-Mannheim/tesseract](https://github.com/UB-Mannheim/tesseract/wiki)

2. Install Python packages:
   ```bash
   pip install opencv-python numpy matplotlib pytesseract pillow

3. Run the Jupyter notebook or convert to Python script.

### Google Colab

All dependencies are installed automatically at the beginning of the notebook. No additional setup is required.

## Usage

Execute the notebook cells in order. The main execution will:

1. Download YOLOv3-tiny files and a sample test image.
2. Perform object detection on the test image and display the result.
3. Create a sample text image, extract text using OCR, and print the extracted text.

To run only one of the pipelines, comment out the respective section in the `if __name__ == "__main__"` block.

### Example

```python
# Object detection
detect_objects("path/to/your/image.jpg")

# OCR
text = extract_text_from_image("path/to/text_image.png")
print(text)
```

## Code Structure

| Function / Section                     | Description                                                                 |
|----------------------------------------|-----------------------------------------------------------------------------|
| `detect_objects(image_path, conf_thresh)` | Loads image, runs YOLO forward pass, applies NMS, draws boxes, shows result. |
| `preprocess_image_for_ocr(image_path)`    | Applies grayscale, Gaussian blur, and adaptive threshold to improve OCR.     |
| `extract_text_from_image(image_path, lang)`| Performs OCR using Tesseract and cleans the extracted text.                 |
| `create_sample_image()`                   | Generates a simple text image for demonstration.                            |
| Main execution block                      | Downloads necessary files, calls detection and OCR functions, prints output.|

## Results

### Object Detection
- Annotated image displayed using Matplotlib.
- Console output lists detected objects with their confidence scores (if any meet the threshold).
- If no objects are detected, a message is printed (this is normal for images without recognizable objects at the given confidence level).

### OCR
- Extracted text printed to the console.
- For the sample image, the output will be:
  ```
  Hello, welcome to Project 4!
  This is an OCR demonstration.
  ```

## Customization

- **Change detection threshold**: Modify `CONFIDENCE_THRESHOLD` variable.
- **Use your own images**: Replace `test_image.jpg` with any image path (local or URL). The code uses `wget` for online images, but you can load local files directly.
- **OCR language**: Change the `lang` parameter in `extract_text_from_image` (e.g., `'eng'`, `'spa'`, `'fra'`). Requires corresponding Tesseract language packs.
- **Preprocessing**: Adjust Gaussian kernel size or adaptive threshold parameters in `preprocess_image_for_ocr`.
- **Non‑Maximum Suppression**: Modify the `nms_threshold` in the `detect_objects` function (default 0.4).

## Author

**Noor R Saad**  
Project 4 – Image or Text Recognition (Basic)  
This code was developed as part of an assignment to demonstrate the use of pre-trained models and simple libraries for computer vision tasks.

---

# Invisible Cloak with AI, Gesture Recognition, and Object Tracking

This project is a real-time computer vision application that simulates an "invisible cloak" effect using OpenCV. It leverages background subtraction, color masking, gesture recognition with MediaPipe, object tracking, and scene analysis with a pre-trained MobileNetV2 model from PyTorch.

## Features

* **Invisibility Cloak Effect**: Hides objects of a specified color by replacing them with the background.
* **AI-Based Scene Analysis**: Uses a pre-trained MobileNetV2 model to classify scenes or objects in the frame.
* **Gesture Recognition**: Detects hand gestures like pinch and peace signs using MediaPipe Hands.
* **Object Tracking**: Allows users to select and track objects in real-time.
* **Snapshot Saving**: Save frames with current effects applied.
* **Live Frame Information**: Displays frame count and timestamp overlay.

## Requirements

* Python 3.7+
* OpenCV
* NumPy
* Torch
* TorchVision
* MediaPipe

Install dependencies using pip:

```bash
pip install opencv-python numpy torch torchvision mediapipe
```

## Usage

1. Run the script:

```bash
python invisible_cloak.py
```

2. Move out of the frame while the background is captured.
3. Wear a blue cloth (or change HSV range as needed).
4. Use keys:

   * `q`: Quit
   * `s`: Save snapshot
   * `r`: Recalibrate background
   * `t`: Select object for tracking

## Contribution

Contributions are welcome! Whether it's adding more gesture types, improving object tracking, optimizing performance, or extending features, feel free to fork the repository and submit a pull request.

To contribute:

1. Fork this repository
2. Create a new branch: `git checkout -b feature-xyz`
3. Make your changes and commit: `git commit -am 'Add new feature'`
4. Push to the branch: `git push origin feature-xyz`
5. Open a pull request

Please ensure your code follows clean coding standards and is well-documented.

## License

This project is open source and available under the MIT License.

---

Thank you for checking out this project! If you find it useful or interesting, consider contributing or sharing your ideas to help improve it further.

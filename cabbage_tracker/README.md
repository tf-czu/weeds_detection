# Cabbage tracker

Python tool for real-time detection and mapping of young cabbage plants.
## Features
* Data collection from an RGB camera and a GNSS receiver.
* Detection of young cabbage plants using a DNN model.
* Localisation of individual plants, assignment of global coordinates, and export to a CSV file.
* Simulation of a camera and GNSS receiver for demonstration and testing purposes.

## Visualization
Below is a sample animation of the detection:

![Movement Visualization](cabbage.gif)

## Quick Start
1. Download and install the [Osgar](https://github.com/robotika/osgar) framework (or set the PYTHONPATH as required).
2. To run the Cabbage Tracker, follow these steps:
    ```bash
   git clone https://github.com/tf-czu/weeds_detection.git
   cd weeds_detection
   
   # for demo:
   python -m osgar.record cabbage_tracker/config/test-detector.json --duration 10
   
   # for real use:
   python -m osgar.record cabbage_tracker/config/cabbage_tracker.json --duration 10 
   
   # Quick analysis of results:
   python -m osgar.logger <log name>
    ```
## License

### Code
The source code of this project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

### Model and Weights
The trained model [weights](images/best2.pt) included in this repository were trained using the **Ultralytics YOLO** framework. 
- The weights are provided for research and demonstration purposes.
- Please note that the use of Ultralytics software and the resulting models is governed by the **AGPL-3.0 License**. For commercial use, a separate commercial license from Ultralytics may be required.
- The training dataset consists of custom data proprietary to the authors.

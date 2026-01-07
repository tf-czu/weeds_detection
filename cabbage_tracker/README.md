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
# CNN-LSTM Stock & Time-Series Prediction App (Demo)

This repository contains a production-ready demo implementation of a hybrid Deep Learning model designed to predict time-series data (such as spot gold or stock prices). The application integrates a deep learning backend with a Flask API containerized via Docker.

> **Note on Codebase Visibility:** The core model training pipelines and proprietary data processing modules are hosted in a private repository for security and IP protection. This demo repository showcases the controller architecture, deployment configurations, and runtime environment setups.

---

## Technical Architecture & Core Logic

### 1. Hybrid Neural Network Model
The model leverages a hybrid network combining **Convolutional Neural Networks (CNN)** and **Long Short-Term Memory (LSTM)** networks:
* **CNN Layers:** Used to extract local, structural features and patterns from the historical price sequence data.
* **LSTM Layers:** Used to capture long-term temporal dependencies and sequential trends over time.

### 2. Full-Stack Integration (Flask API)
The web application is built on top of the Flask microframework:
* Receives incoming prediction queries and features via HTTP API endpoints.
* Loads the pre-trained hybrid model (`.h5` structure) and preprocessed scaler binary.
* Sanitizes, scales, and transforms input inputs in real-time.
* Serves model predictions securely to frontend dashboards.

### 3. Containerization (Docker)
The application environment is fully containerized using Docker to ensure seamless deployment across cloud environments without dependency mismatches:
* Uses a lightweight Python base image.
* Sets up dependency configurations dynamically via `requirements.txt`.
* Configures application runtimes and exposes execution ports.

---

## Repository Breakdown

* `app.py`: The entry point of the Flask application managing routes, security contexts, and prediction pipelines.
* `Dockerfile`: Production-ready container configuration to build and run the system.
* `requirements.txt`: Specified Python library dependencies including TensorFlow, Flask, and NumPy.

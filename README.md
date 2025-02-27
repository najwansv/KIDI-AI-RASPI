<p align="center">
  <a href="" rel="noopener">
 <img src="Dashboard/logo.png" alt="KIDI AI logo"></a>
</p>

<h3 align="center">KIDI AI</h3>

<div align="center">

[![Status](https://img.shields.io/badge/status-active-success.svg)]()
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](/LICENSE)

</div>

---

<p align="center"> 🤖 KIDI AI - An intelligent vision processing system powered by Raspberry Pi 5 and HAILO AI
    <br> 
</p>

## 📝 Table of Contents

- [About](#about)
- [How it works](#working)
- [Getting Started](#getting_started)
- [Usage](#usage)
- [Built Using](#built_using)
- [Authors](#authors)
- [Acknowledgments](#acknowledgement)

## 🧐 About <a name = "about"></a>

KIDI AI is a computer vision system running on Raspberry Pi 5 that leverages HAILO AI acceleration for tasks such as face detection, recognition, depth estimation, and object classification. The system provides efficient AI processing at the edge using HAILO's neural processing unit.

## 💭 How it works <a name = "working"></a>

The system uses multiple AI models (located in the Model directory) optimized for the HAILO AI accelerator to process visual data. The main Python application coordinates these models to provide comprehensive AI-powered vision capabilities with significantly improved performance compared to CPU-only solutions.

## 🏁 Getting Started <a name = "getting_started"></a>

These instructions will help you set up the project on your local machine for development and testing.

### Prerequisites

* Raspberry Pi 5
* HAILO AI accelerator
* Python 3.x
* HAILO Runtime and SDK
* Camera module compatible with Raspberry Pi
* Display (optional)

### Installing

1. Clone this repository
2. Install required dependencies:

```py
pip install -r requirements.txt
```

3. Set up any necessary configuration

## 🎈 Usage <a name = "usage"></a>

Explain how to run and use your application:

```py
python main.py
```
For the dashboard interface, navigate to the Dashboard directory or access the web interface via:

```
http://localhost:5001
```

## ⛏️ Built Using <a name = "built_using"></a>

- [Raspberry Pi 5](https://www.raspberrypi.org/) - Edge computing platform
- [HAILO AI](https://hailo.ai/) - Neural processing accelerator
- [Python](https://www.python.org/) - Main programming language
- [OpenCV](https://opencv.org/) - Computer vision library
- [Flask](https://flask.palletsprojects.com/) - Web framework for dashboard
- [TensorFlow/PyTorch] - For AI model development and optimization

## ✍️ Authors <a name = "authors"></a>

- Najwan - Internship work

## 🎉 Acknowledgements <a name = "acknowledgement"></a>

- HAILO AI for their hardware and SDK support
- Raspberry Pi Foundation
- Contributors to the open-source AI models used in this project
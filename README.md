# 🌐 Dual-Mode Web-Based Image Processor

A deep learning-based application for **cross-modal image translation** using a compact encoder and an adversarial framework.

## 📌 Overview

This project addresses limitations in traditional image translation systems that rely on single-modality inputs. It introduces a **dual-mode lightweight encoder** capable of processing **grayscale and thermal images**, supporting:

- Grayscale to RGB colorization  
- Thermal to RGB reconstruction  
- Segmentation and edge detection  
- Real-time web-based image processing  

Built using **Python**, **OpenCV**, and **Flask**, it runs as a **web application** with an interactive frontend.

---

## 🚀 Features

- ⚡ **Single Encoder, Multi-Modal Input**
- 🔁 **Image Translation**
- 🎨 **Color Space Conversion**: RGB, HSV, HLS, YCbCr, XYZ
- 🔍 **Canny Edge Detection**
- 💻 **Web Interface**
- 🤖 **CNN Architecture**
- 📊 **Evaluation Metrics**: PSNR, SSIM, FID, IoU, Dice score

---

## 📂 Project Structure

```

Image\_Translation\_py/
├── CODE/
│   ├── app.py               # Flask app logic
│   ├── static/uploads/      # Uploaded images
│   ├── templates/           # HTML templates
│   └── model.py             # Deep learning model

````

---

## 🛠️ Tech Stack

- **Language**: Python 3.x  
- **Framework**: Flask  
- **Libraries**:
  - OpenCV (cv2)
  - NumPy, Pandas, Matplotlib
  - Scikit-learn
- **Tools**: Anaconda, Jupyter Notebook, VS Code  
- **Frontend**: HTML5, CSS3 (via Flask templates)

---

## 📦 Installation & Setup

1. **Clone the Repository**
   ```bash
   git clone https://github.com/nithishbachu/Image_Translation_py.git
   cd Image_Translation_py/CODE


2. **Create a Virtual Environment (Optional)**

   ```bash
   conda create -n image-env python=3.9
   conda activate image-env
   ```

3. **Install Dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Run the Flask App**

   ```bash
   python app.py
   ```

5. **Access the App**
   Open your browser at: [http://localhost:5000](http://localhost:5000)

---

## 📸 Sample Operations

* Grayscale to HSV
* Apply Canny Edge Detection
* Thermal → color model image generation

---

## 📊 Evaluation Metrics

* **Image Translation**: PSNR, SSIM, FID
* **Performance**: Runtime per image, memory usage

---


## 📬 Contact

**Author**: Nithish Bachu
**GitHub**: [github.com/nithishbachu](https://github.com/nithishbachu)

---

Feel free to contribute or raise issues via [GitHub Issues](https://github.com/nithishbachu/Image_Translation_py/issues)

```

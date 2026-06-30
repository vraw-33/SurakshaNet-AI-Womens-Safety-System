# 🛰️ SurakshaNet AI

> AI-powered CCTV surveillance system for early detection of persistent following behavior to improve women's safety.

SurakshaNet AI is a computer vision based surveillance prototype that detects when one person persistently follows another in CCTV footage. Using real-time object detection and multi-object tracking, the system raises alerts, logs incidents, and automatically captures evidence snapshots for security personnel.

> **⚠️ Current Version:** MVP (Hackathon Demo)

---

# ✨ Features

### ✅ Current MVP

- Real-time Person Detection (YOLO26m)
- Multi-Object Tracking (BoT-SORT)
- Persistent Following Detection
- Live Alert System
- Event Timeline
- Automatic Evidence Screenshot Capture
- Live Dashboard
- GPU Accelerated Inference (CUDA)

---

# ⚙️ How it Works

```
Input Video / CCTV
        │
        ▼
 Person Detection (YOLO26m)
        │
        ▼
 Multi-Object Tracking (BoT-SORT)
        │
        ▼
 Persistent Following Detection
        │
        ├── Alert Generation
        ├── Event Logging
        └── Screenshot Capture
```

The system continuously detects people, assigns unique IDs, tracks their movement across frames, and analyzes proximity and motion patterns. If two tracked individuals remain in suspicious following behavior for a sustained duration, an alert is generated and evidence is automatically captured.

---

# 🚀 Tech Stack

- Python
- OpenCV
- PyTorch
- Ultralytics YOLO26m
- BoT-SORT Tracker
- NumPy

---

# 📂 Project Structure

```
SurakshaNet-AI/
│
├── SurakshaNet_Detector.py
├── botsort.yaml
├── requirements.txt
├── README.md
└── surakshanet_incidents/
```

---

# ▶️ Running the Demo

1. Install dependencies

```bash
pip install -r requirements.txt
```

2. Place the YOLO model weights (`yolo26m.pt`) in the project folder.

3. Set the video path inside the script.

4. Run

```bash
python SurakshaNet_Detector.py
```

Controls:

- **SPACE** → Pause
- **Q** → Quit

---

# 🎯 Current Demo Capabilities

- Detects people in CCTV footage
- Tracks each person with unique IDs
- Detects persistent following behavior
- Displays real-time warning and confirmation alerts
- Maintains an incident timeline
- Automatically saves evidence screenshots

---

## 🚧 Future Roadmap

### Phase 1 – Smarter Tracking
- Person Re-Identification (Re-ID)
- Cross-Camera Tracking
- Pedestrian Attribute Recognition
  - Gender (confidence-based)
  - Age Group
  - Clothing Attributes
  - Backpack / Accessories

### Phase 2 – Safety Intelligence
- Persistent Following Across Cameras
- Isolation Risk Detection
- Night-Time Safety Monitoring
- Risk Score Engine
- Smart Alarm Hysteresis (False Alert Reduction)

### Phase 3 – Incident Intelligence
- Automatic Incident Summary Generation
- Search by Person Attributes
- Search by Track ID
- Evidence Export
- Cloud Dashboard & Remote Monitoring

### Phase 4 – Public Safety
- Fight Detection
- Fall / Collapse Detection
- Abandoned Object Detection
- Restricted Zone Intrusion
- Crowd Anomaly Detection

---

# 📌 Note

This repository demonstrates the MVP(Minimum Viable Product) submitted for a hackathon.

The current implementation focuses on **persistent following detection**. Future features listed above are planned and **not yet implemented**.

---

# 👨‍💻 Author

**Vaibhav Rawat**

B.Tech Artificial Intelligence & Machine Learning

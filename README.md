# SurakshaNet AI 🛰️
SurakshaNet AI is an AI-powered women's safety early warning system that detects persistent following behavior in CCTV footage using YOLO and multi-object tracking. It generates real-time alerts, logs incidents, and automatically captures evidence snapshots to assist security personnel.

```markdown


[![Language: Python](https://img.shields.io/badge/Language-Python-blue)](https://www.python.org/) [![License: MIT](https://img.shields.io/badge/License-MIT-green)](LICENSE) [![YOLO26](https://img.shields.io/badge/YOLO-26m-orange)](https://docs.ultralytics.com/models/yolo26) [![OpenCV](https://img.shields.io/badge/OpenCV-4.8-brightgreen)](https://opencv.org/) [![PyTorch](https://img.shields.io/badge/PyTorch-2.0-orange)](https://pytorch.org/)

**AI-driven surveillance demo for women’s safety via early detection of persistent-following behavior.**  
_This README follows best practices for a clear overview and ready-to-use instructions._

## Table of Contents

- [Executive Summary](#executive-summary)  
- [Demo Stage Features](#demo-stage-features)  
- [Installation & Quick Start](#installation--quick-start)  
- [Configuration](#configuration)  
- [Usage & UI Walkthrough](#usage--ui-walkthrough)  
- [Troubleshooting](#troubleshooting)  
- [File Structure](#file-structure)  
- [Roadmap & Future Work](#roadmap--future-work)  
- [Contribution & License](#contribution--license)  
- [Demo Video & Presentation](#demo-video--presentation)  
- [Demo Script & Talking Points](#demo-script--talking-points)  
- [Example Git Commands](#example-git-commands)  
- [References](#references)  

## Executive Summary

SurakshaNet AI is a proof-of-concept surveillance tool designed to **proactively detect persistent following** in public areas, enhancing women’s safety. It applies computer vision to CCTV or video streams, automatically alerting security personnel when one person closely trails another for a sustained period. The system leverages the Ultralytics YOLO26m object detector for real-time person detection and the BoT-SORT tracker for multi-object identity tracking. When a “following” event is confirmed, the demo captures an evidence snapshot, logs the incident, and displays an alert banner on the live video feed. This README provides a concise overview, installation guide, and usage instructions following GitHub best practices.

## 🚀 Demo Stage Features

This prototype implementation (demo stage) includes:

- **YOLO26 Person Detection:** Real-time bounding-box detection of people using the Ultralytics YOLO26m model.
- **BoT-SORT Tracking:** Multi-object tracker that assigns consistent IDs to each person across frames.
- **Persistent Following Detection:** Custom logic that flags when one person stays within a close proximity of another for a configurable number of frames.
- **Live Dashboard UI:** On-screen display shows processing FPS, tracking info, and current state (see screenshot below).
- **Event Timeline & Alerts:** A sidebar/timeline logs events (e.g., “Track 1 approaching Track 4”) and color-coded banners appear for **WARNING** (yellow) and **CONFIRMED** (red) states.
- **Automatic Screenshot Capture:** When a following event is *confirmed*, an evidence snapshot (`incident_<timestamp>.jpg`) is saved to `surakshanet_incidents/`.
- **Incident Logging:** Counts and details of each incident are recorded; the incident count is displayed at session end.
- **Diagnostic Self-Test:** On startup, the demo runs a quick detection/tracker test on the first frame to verify model and ID-assignment (printed to console).

*(Below is an example UI screenshot placeholder. In practice, include an actual capture showing bounding boxes, alerts, and dashboard.)*

![Demo of SurakshaNet AI detecting a following event](assets/demo_screenshot.png)  
*Screenshot: SurakshaNet UI during a detected following (red boxes).*

As recommended, example commands and visuals are provided in code blocks or images for clarity.

## 💾 Installation & Quick Start

**Requirements:**  
- Python 3.10+ (tested on 3.11)  
- GPU with CUDA support (optional; CPU can run slower)  
- [PyTorch](https://pytorch.org) (torch >=2.0), [OpenCV](https://opencv.org) (cv2), [NumPy], [Ultralytics YOLO](https://pypi.org/project/ultralytics), [lap] (for tracking). All required libraries are listed in `requirements.txt`.

**Setup:** Clone the repository and install dependencies (copy-paste commands):

```bash
git clone https://github.com/<YourUsername>/SurakshaNet-AI.git
cd SurakshaNet-AI
pip install -r requirements.txt
```

**Download YOLO26m Model:** Obtain the `yolo26m.pt` weights from Ultralytics (e.g., [Releases](https://github.com/ultralytics/ultralytics/releases)) and place it in the project root. Example (Linux/macOS):

```bash
curl -L -o yolo26m.pt https://github.com/ultralytics/ultralytics/releases/download/v3.5/yolo26m.pt
```

**Quick Start:** Run the demo on a sample video or webcam:

```bash
python surakshanet_following_detector.py --video path/to/input_video.mp4
```

On startup, you should see console messages (including a self-test), and then a video window. If using a webcam, replace `--video` with `--webcam`.  

- **Controls:** Press **SPACE** to pause/resume processing, and **Q** to quit.  
- **Expected Output:** A live video showing detected people with green/red bounding boxes. The console will log events like “Track 1 approaching Track 4” and “Following confirmed (IDs)”. Upon confirmation, a snapshot is saved (see [Usage](#usage--ui-walkthrough) below). The final console summary will show incident count and screenshot directory.

## ⚙️ Configuration

The main parameters (in `surakshanet_following_detector.py`) control detection logic and sensitivity:

- `SKIP_FRAMES`: Number of video frames between each YOLO detection (default `2`). Higher values = less frequent inference (improves speed, lowers accuracy).  
- `WARN_FRAMES`: Consecutive *detection cycles* (not raw frames) where two tracks stay close to raise a **WARNING** (e.g. 20).  
- `CONFIRM_FRAMES`: Cycles to confirm a following and record the incident (e.g. 35 for demo). Until this is reached, no screenshot is saved.  
- `PROXIMITY_PX`: Pixel distance threshold to consider two people “close” (e.g. ~100 px in a 596×336 frame).  
- `OCCLUSION_GRACE`: Number of frames to continue counting even if one person is briefly lost (default 45).

Below is a sample table of recommended values for demonstration versus real-world use. Adjust these based on video FPS and scene scale:

| Parameter        | Description                              | Demo (Short Clip) | Production (Real-World)   |
| ---------------- | ---------------------------------------- | :---------------: | :-----------------------: |
| `SKIP_FRAMES`    | Frames to skip between YOLO inferences   |        2          | 3 – 5                     |
| `WARN_FRAMES`    | Cycles of closeness to trigger *Warning* |        20         | 30 – 60                   |
| `CONFIRM_FRAMES` | Cycles to confirm following (capture)    |        35         | 60 – 120                  |
| `PROXIMITY_PX`   | Pixel distance defining "close"         |       ~100        | 100 – 200 (depends on camera) |
| `OCCLUSION_GRACE`| Frames to maintain track through brief occlusion | 45  | 60 – 90           |

_For example, on a 30 FPS video with `SKIP_FRAMES=2`, `CONFIRM_FRAMES=30` requires ~2 seconds of continuous following._ Adjust parameters or use the FPS-based formula:  
```
WARN_FRAMES = round(WARN_SECONDS * video_fps / SKIP_FRAMES)  
CONFIRM_FRAMES = round(CONFIRM_SECONDS * video_fps / SKIP_FRAMES)
```  
using desired durations (e.g. WARN_SECONDS=0.5s, CONFIRM_SECONDS=1.0s).

## 🎮 Usage & UI Walkthrough

Upon running, the application processes each frame and updates the display:

- **Pause/Resume:** Press **SPACE**.  
- **Quit:** Press **Q** or close the window.  
- **Bounding Boxes:** Normal tracked people are outlined in **green**. When a potential following is detected, involved persons are shown with **red** boxes.  
- **Status Banner:** A colored overlay at the top-left shows current state:  
  - `MONITORING` (green) – no following detected.  
  - `POSSIBLE FOLLOWING DETECTED` (yellow) – two people have been close for ≥ `WARN_FRAMES`.  
  - `FOLLOWING CONFIRMED` (red) – sustained for ≥ `CONFIRM_FRAMES`.  
- **Event Log:** A sidebar or overlay lists timestamped messages (e.g., “Track 1 approaching Track 4”). This timeline helps trace events.  
- **Screenshots:** When *FOLLOWING CONFIRMED* occurs, a snapshot is automatically saved in `surakshanet_incidents/` (e.g., `incident_20230630_193632.jpg`). By design, only one screenshot per incident is saved (to prevent duplicates) – see console log “Snapshot saved”.  
- **Demo Video:** The processing FPS is displayed; expect ~80 FPS on a modern GPU (demonstrating the pipeline efficiency).

*Example: The screenshot above illustrates the UI during an alert (red boxes and a red banner). As best practice, visuals like screenshots are used to illustrate functionality.*

## 🛠️ Troubleshooting

- **No detections / Errors:** Verify that `yolo26m.pt` exists in the project folder and is not corrupted. Ensure the `ultralytics` package is installed (YOLO26 requires a recent ultralytics version). Check Python version (should be 3.10+).  
- **Missing Tracks:** Increase `OCCLUSION_GRACE` if brief occlusions cause loss of a target. Ensure the detection confidence threshold (`CONF_THRES`) is set appropriately in the code.  
- **No screenshots saved:** Remember screenshots are only taken upon *confirmation*. If your test clip is too short, consider lowering `CONFIRM_FRAMES`. The console logs (“CALLING SCREENSHOT”, “Snapshot saved”) can help debug.  
- **False Alerts:** Increase `PROXIMITY_PX` or `WARN_FRAMES` if normal walking triggers alerts.  
- **Performance issues:** Try reducing the inference image size (`INFER_SIZE`) or increasing `SKIP_FRAMES` for faster processing. Running on CPU only will be slower.  
- **Environment issues:** Ensure all dependencies in `requirements.txt` are installed in the correct environment. Do *not* assume the user already has everything – list exact versions.  
- **Debug Prints:** Before a presentation, remove or comment out any leftover debug `print()` statements (e.g., prints inside loops or “CALLING SCREENSHOT”) to keep the console output clean.  
- **Git Push Errors:** If `git push` is rejected, make sure your local branch matches the remote (`git branch -M main` and set `origin`). See example git commands below for correct workflow.

## 📁 File Structure

```
SurakshaNet-AI/                # Project root
├── surakshanet_following_detector.py   # Main script (detection, tracking, UI)
├── botsort.yaml              # BoT-SORT tracker configuration file
├── requirements.txt          # Python dependencies
├── README.md                 # This documentation
└── surakshanet_incidents/    # Folder where snapshots are saved
```

- **`surakshanet_following_detector.py`** – The core script that initializes the YOLO model and BoT-SORT tracker, processes video frames, and implements the following-detection logic (in the `FollowingDetector` class).  
- **Tracker Config (`botsort.yaml`)** – Contains parameters for the BoT-SORT tracker (e.g., matching distance).  
- **Helper Functions** – The script defines functions like `draw_dashboard()`, `draw_event_log()`, and `draw_pause_overlay()` to render the UI, as well as `maybe_save_screenshot()` for capture logic. Reading through the well-commented code will clarify behavior.  
- **Requirements & Dependencies** – The `requirements.txt` lists needed Python libraries. We recommend virtual environments (venv/conda) to manage these.  
- **License** – The `LICENSE` file (MIT) is included, as it clarifies usage rights (recommended practice).

Providing a clear file structure and descriptions helps new users and maintainers navigate the project.

## 🚧 Roadmap & Future Work

Planned enhancements for a full system include:

- **Person Re-Identification (Re-ID):** Integrate a Re-ID model so that the same individual is recognized even if they leave the frame and re-enter (or cross-camera scenarios).  
- **Cross-Camera Tracking:** Extend the system to link tracks between multiple camera feeds for city-wide persistent tracking.  
- **Web Dashboard:** Build a web-based interface for remote monitoring, live alerts, and incident management.  
- **Dataset & Evaluation:** Assemble a dataset of following scenarios for quantitative evaluation. Define metrics (precision/recall) to measure detection performance.  
- **Advanced Analytics:** Add person attributes (e.g., estimated age/gender) or behavioral analysis to enrich alerts.  
- **Deployment & Optimization:** Test on embedded hardware; possibly switch to lighter models (e.g., YOLO nano) or faster trackers (ByteTrack) for real-time operation.  

This demo is a research MVP; these future directions align with ongoing work in AI surveillance (to be detailed in follow-up research).

## 🤝 Contribution & License

Contributions are welcome! Feel free to open issues or submit pull requests on GitHub. Before contributing, ensure the code passes any provided tests and documentation is updated. (For formal submissions, check hackathon-specific guidelines.)

This project is released under the **MIT License**. By including a clear license, others know how they can use or adapt this work. 

**Citation:** If you use this work in a report or presentation, please cite the hackathon entry or contact the author for preferred citation.  

**Acknowledgments:** This demo was developed as part of an academic hackathon on AI for public safety. Thanks to mentors and the Ultralytics and BoT-SORT teams for their open-source frameworks, which made this prototype possible.

## 🎥 Demo Video & Presentation

Watch a short demo on YouTube (unlisted link): [Demo Video](https://youtube.com/your-video-link) — the video walks through the UI and shows following detection in action. 

A sample presentation deck is available here: [Slides](https://slides.com/your-presentation-link). Feel free to use it as a template.

## 🎙️ Demo Script & Talking Points

1. **Introduce SurakshaNet AI:** Explain the goal (detect persistent following to help women's safety) and core tech (YOLO26 + BoT-SORT).  
2. **Show live demo:** Run the script. Point out UI elements: real-time video, bounding boxes, status banner, and FPS.  
3. **Trigger a following event:** Demonstrate two people walking together. Highlight when the system enters *WARNING* (yellow, “possible following”) and then *CONFIRMED* (red, alert).  
4. **Evidence capture:** Show the saved snapshot and console log. Emphasize automation (no user input needed to capture evidence).  
5. **Configuration:** Briefly mention how parameters like `WARN_FRAMES` affect sensitivity.  
6. **Performance:** Note the processing rate (e.g. “~80 FPS on GPU”) and how skipping frames reduces load.  
7. **Future Work:** Discuss planned improvements (e.g., multi-camera, Re-ID, web UI).  
8. **Q&A Prep:** Be ready to explain limitations (e.g., only works on clear video; false positives if people walk side-by-side) and ideas for validation (data, metrics).

This script ensures a concise presentation (1–2 minutes) covering motivation, functionality, and future directions.

## 🔧 Example Git Commands

To update or push changes to GitHub, use commands like:

```bash
git add .
git commit -m "Add feature/detailed README"
git remote add origin https://github.com/<YourUsername>/SurakshaNet-AI.git
git push -u origin main
```

*(Replace `<YourUsername>` and branch names as needed.)* As the Norday guide suggests, providing exact commands in README helps reviewers reproduce your setup.

## References

- Sullivan, *How to write a good README* (GitHub) – advises a brief overview and bullet highlights at top.  
- Norday, *Formatting your README* (2024) – recommends using Markdown code blocks for all setup commands.  
- Utrecht University, *Best Practices for Reproducible Code* – stresses short description, installation steps, usage examples, license.  
- Ultralytics YOLO26 Documentation – details on YOLO26m model used for detection.  

```bash
# Requirements file snippet (for reference, not executed here):
python>=3.10
ultralytics
opencv-python
torch>=2.0
numpy
lap
```
```


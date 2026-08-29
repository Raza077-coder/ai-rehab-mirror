# 🪞 AI Rehab Mirror

**AI-powered physiotherapy rehabilitation assistant** — a computer-vision system that analyzes human movement and provides real-time feedback on exercise form.

Physiotherapy is not only about completing an exercise — it's about performing every movement correctly. Small mistakes (incorrect posture, limited range of motion, uneven movement, poor control, missing hold duration) affect recovery quality. **AI Rehab Mirror** understands *how* a movement is performed, not just *whether* it was done.

> *"Building AI systems that don't just see the world, but understand human movement."*

---

## ✨ Capabilities

- **Full-body pose landmarks** via MediaPipe Pose Landmarker
- **Joint angle analysis** — shoulder abduction, elbow, torso angles
- **Shoulder movement range** tracking
- **Torso alignment** detection (lean from vertical)
- **Left/right symmetry** scoring
- **Exercise phase detection** — Ready → Raising → Holding → Lowering → Completed
- **Repetition quality** scoring (0–100)
- **Movement score** — composite quality metric per rep
- **Real-time feedback** with actionable cues
- **Medical-style visual report** with per-rep metrics

Instead of simply counting reps, the system evaluates **HOW** the movement is performed.

---

## 🔬 Processing Pipeline

```
Camera Input
    │
    ▼
Frame Processing
    │
    ▼
Pose Estimation (MediaPipe Pose Landmarker)
    │
    ▼
Landmark Smoothing (temporal EMA)
    │
    ▼
Joint Angle Analysis (NumPy vector math)
    │
    ▼
Exercise State Detection (state machine)
    │
    ▼
Form Evaluation (torso lean, symmetry, range)
    │
    ▼
Real-time Feedback (actionable cues)
    │
    ▼
Medical-style Visual Report
```

## 🏋️ First Prototype: Shoulder Abduction

The first prototype focuses on **shoulder abduction rehabilitation**.

### Movement Stages (State Machine)

```
Ready → Raising → Holding → Lowering → Completed Rep
```

### Feedback Examples

| Cue | When |
|-----|------|
| "Raise your arm higher" | Arm below the target abduction angle |
| "Keep your torso straight" | Torso lean exceeds the threshold |
| "Hold position" | Hold duration below the minimum |
| "Good repetition" | A rep is completed with acceptable form |

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| **Computer Vision** | MediaPipe Pose Landmarker, OpenCV, NumPy, temporal movement analysis |
| **Backend** | Python, FastAPI |
| **Frontend** | React, TypeScript, Vite |
| **Video Processing** | OpenCV rendering, FFmpeg export |

## 📁 Repository Structure

```
ai-rehab-mirror/
├── backend/
│   ├── app/
│   │   ├── __init__.py
│   │   ├── main.py                 # FastAPI app + endpoints
│   │   ├── pose_analysis.py        # MediaPipe wrapper, angles, smoothing
│   │   ├── exercise_state_machine.py  # Ready→Raising→Holding→Lowering→Completed
│   │   └── video_processing.py     # OpenCV rendering + FFmpeg export
│   └── requirements.txt
├── frontend/
│   ├── src/
│   │   ├── App.tsx                 # Main UI
│   │   ├── api.ts                  # Backend API client
│   │   ├── main.tsx
│   │   └── index.css
│   ├── index.html
│   ├── package.json
│   ├── tsconfig.json
│   └── vite.config.ts
├── README.md
├── LICENSE
└── .gitignore
```

## 🚀 Getting Started

### Prerequisites

- Python 3.9+
- Node.js 18+
- FFmpeg (optional, for video re-encode)

### 1. Backend

```bash
cd backend
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

The API will be available at `http://localhost:8000`. Interactive docs at `http://localhost:8000/docs`.

### 2. Frontend

```bash
cd frontend
npm install
npm run dev
```

Open `http://localhost:5173` in your browser. The Vite dev server proxies `/api` to the FastAPI backend.

### 3. Use It

- **Start Live Camera** — real-time pose analysis with live feedback
- **Analyze Image** — upload a single frame for analysis
- **Analyze Video** — upload a video; the backend returns an annotated video + per-rep metrics report

## 🔌 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/health` | Health check + MediaPipe availability |
| `GET` | `/api/exercise/state` | State machine config & thresholds |
| `POST` | `/api/analyze-frame` | Analyze a single image frame |
| `POST` | `/api/analyze-video` | Process a video → annotated video + report |
| `GET` | `/api/videos/{filename}` | Serve a processed video |
| `POST` | `/api/reset` | Reset the live session |

## 🧠 Why It Matters

A large part of rehabilitation happens **outside the clinic**. AI-assisted rehab helps:

- **Physiotherapists** evaluate movement objectively
- **Patients** get guidance between appointments
- **Clinics** create measurable recovery workflows
- **Remote rehab** becomes more accessible

The goal is **not to replace physiotherapists** but to build an intelligent assistant that supports them.

## 🗺️ Next Steps

- [ ] More rehabilitation exercises (shoulder flexion, elbow, knee, etc.)
- [ ] Personalized recovery scoring
- [ ] Patient progress tracking
- [ ] Therapist dashboard
- [ ] Remote monitoring workflows

## 📄 License

[MIT](LICENSE)

## 👤 Author

**Ali Raza** — AI & Agentic AI Developer
- GitHub: [@Raza077-coder](https://github.com/Raza077-coder)
- LinkedIn: [Ali Raza](https://www.linkedin.com/in/ali-raza-857053373)
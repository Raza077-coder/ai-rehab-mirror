# 🪞 AI Rehab Mirror

**AI-powered physiotherapy rehabilitation assistant**

A computer vision powered rehabilitation assistant that analyzes human movement and provides real-time feedback on exercise form. Physiotherapy is not only about completing an exercise — it's about performing every movement correctly. Small mistakes (incorrect posture, limited range of motion, uneven movement, poor control, missing hold duration) affect recovery quality.

> *"Building AI systems that don't just see the world, but understand human movement."*

---

## 🎯 Why It Matters

A large part of rehabilitation happens **outside the clinic**. AI-assisted rehab helps:

- 🩺 **Physiotherapists** evaluate movement objectively and scale their care
- 🧑‍🤝‍🧑 **Patients** get intelligent guidance between appointments
- 🏥 **Clinics** create measurable, data-driven recovery workflows
- 🌍 **Remote rehab** becomes more accessible to everyone

The goal is **not to replace physiotherapists** — it's to build an intelligent assistant that supports them.

---

## ✨ Capabilities

- 🧍 **Full-body pose landmarks** — 33-point MediaPipe Pose detection
- 📐 **Joint angle analysis** — precise measurement of movement angles
- 💪 **Shoulder movement range** — track abduction/adduction range
- 🧘 **Torso alignment** — detect posture and trunk compensation
- ⚖️ **Left/right symmetry** — compare both sides of the body
- 🔄 **Exercise phases** — detect each stage of the movement
- 🔁 **Repetition quality** — not just counting reps, but understanding *how* they're performed
- 📊 **Movement score** — objective, quantifiable form assessment

Instead of simply counting repetitions, the system **understands HOW the movement is performed**.

---

## 🔄 Pipeline

```
Camera Input
    │
    ▼
Frame Processing
    │
    ▼
Pose Estimation
    │
    ▼
Landmark Smoothing
    │
    ▼
Joint Angle Analysis
    │
    ▼
Exercise State Detection
    │
    ▼
Form Evaluation
    │
    ▼
Real-time Feedback
    │
    ▼
Medical-style Visual Report
```

---

## 🏋️ First Prototype: Shoulder Abduction

The first prototype focuses on **shoulder abduction rehabilitation**.

### Movement Stages
```
Ready → Raising → Holding → Lowering → Completed Rep
```

### Real-time Feedback Examples
- 🗣️ *"Raise your arm higher"*
- 🧍 *"Keep your torso straight"*
- ⏸️ *"Hold position"*
- ✅ *"Good repetition"*

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| **Computer Vision** | MediaPipe Pose Landmarker, OpenCV, NumPy, temporal movement analysis |
| **Backend** | Python, FastAPI |
| **Frontend** | React, TypeScript, Vite |
| **Video Processing** | OpenCV rendering, FFmpeg export |

---

## 🚀 Getting Started

*(Setup instructions coming soon — the project is under active development.)*

---

## 🗺️ Next Steps

- [ ] More rehabilitation exercises (beyond shoulder abduction)
- [ ] Personalized recovery scoring
- [ ] Patient progress tracking
- [ ] Therapist dashboard
- [ ] Remote monitoring workflows

---

## 📄 License

[MIT](LICENSE)

---

## 👤 Author

**Ali Raza** — AI & Agentic AI Developer
- GitHub: [@Raza077-coder](https://github.com/Raza077-coder)
- LinkedIn: [Ali Raza](https://www.linkedin.com/in/ali-raza-857053373)
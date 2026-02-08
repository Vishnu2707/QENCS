# QENCS Deployment Configuration

## Frontend: Vercel

Build Settings:
- Framework Preset: Next.js
- Root Directory: `web-app`
- Install Command: `npm install`
- Build Command: `npm run build`

Environment Variables:
| Variable | Value | Description |
|----------|-------|-------------|
| `NEXT_PUBLIC_BACKEND_URL` | `https://qencs-backend.onrender.com` | The base URL of your hosted FastAPI backend. |

---

## Backend: Render

Build Settings:
- Environment: Python 3.9+
- Build Command: `pip install -r requirements.txt` (Ensure you have a requirements.txt at the root or point to it)
- Start Command: `python3 backend/main.py`

Environment Variables:
| Variable | Value | Description |
|----------|-------|-------------|
| `PORT` | `8000` | Port for the FastAPI server. |
| `PYTHON_VERSION` | `3.9` | Specific Python version requirement. |

---

## Post-Deployment Validation
1.  Verify that the dashboard loads at your Vercel URL.
2.  Open the network tab to ensure the `/analyze` endpoint requests aren't being blocked by CORS. (The backend is currently set to `allow_origins=["*"]` for development, which is safe for Initial Phase 2/3).
3.  Check that the 'Stable' status dot in the sidebar turns green once the backend starts responding.

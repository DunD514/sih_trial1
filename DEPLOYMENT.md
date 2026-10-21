# SatQuery AI — Ubuntu AWS EC2 Production Deployment Guide

This guide describes how to deploy and run **SatQuery AI / JARVIS** on an Ubuntu AWS EC2 instance.

---

## 1. AWS EC2 Instance Sizing & Hardware Recommendations

| Deployment Profile | Recommended EC2 Instance | Hardware Specifications | Capabilities Supported |
| :--- | :--- | :--- | :--- |
| **GPU Production (Recommended)** | `g4dn.xlarge` or `g5.xlarge` | 4 vCPU, 16–24 GB RAM, NVIDIA T4/A10G (16–24 GB VRAM) | ChangeFormer V6 (CUDA), Qwen2-VL-2B, Ollama qwen3:8b, Kokoro-82M TTS |
| **CPU Production** | `t3.xlarge` or `c6i.xlarge` | 4 vCPU, 16 GB RAM | ChangeFormer V6 (CPU), Ollama qwen3:8b (CPU), Kokoro-82M TTS (AVX2 SIMD) |
| **Lightweight / Dev Test** | `t3.large` | 2 vCPU, 8 GB RAM (+ 4 GB swapfile) | Core change detection, Kokoro TTS, Ollama running on separate host or qwen2.5:3b |

**AWS Security Group / Firewall Rules:**
- Inbound **SSH**: Port `22` (restrict to your IP)
- Inbound **SatQuery Web App**: Port `8080` (or Port `80` if using Nginx)
- Outbound: All traffic (to download HuggingFace weights / Ollama models)

---

## 2. Model Requirements Breakdown for First Deployment

Do **NOT** commit large weights to Git. Transfer or download only what is needed:

| Model / Asset | Size | First Deployment Status | Download Command / Location |
| :--- | :--- | :--- | :--- |
| **ChangeFormer V6** (`best_ckpt.pt`) | ~160 MB | **Required** for change detection | Place in `checkpoints/ChangeFormer_LEVIR/best_ckpt.pt` |
| **Kokoro-82M ONNX** (`kokoro-v1.0.onnx`) | ~325 MB | **Required** for local neural voice | `checkpoints/kokoro/kokoro-v1.0.onnx` (downloaded from HF `fastrtc/kokoro-onnx`) |
| **Kokoro Voices** (`voices-v1.0.bin`) | ~28 MB | **Required** for voice styles | `checkpoints/kokoro/voices-v1.0.bin` (downloaded from HF `fastrtc/kokoro-onnx`) |
| **Ollama Model** (`qwen3:8b`) | ~4.7 GB | **Required** for autonomous reasoning | Run `ollama run qwen3:8b` via Ollama service |
| **Qwen2-VL-2B-Instruct** | ~4.5 GB | **Optional** (Can be omitted on 1st boot) | `Qwen2-VL-2B-Instruct/` directory. If omitted, app starts cleanly and uses rule-based metadata context. |
| **DSIFN Checkpoint** | ~160 MB | **Optional** (Omit unless testing DSIFN) | `checkpoints/ChangeFormer_DSIFN/` |

---

## 3. Step-by-Step Ubuntu Setup

### Step 3.1: System Packages & Dependencies
SSH into your Ubuntu 22.04 / 24.04 instance:

```bash
sudo apt update && sudo apt upgrade -y
sudo apt install -y python3-pip python3-venv git curl libsndfile1 build-essential
```

*(Optional for GPU instances)*: Install NVIDIA drivers & CUDA toolkit:
```bash
sudo apt install -y ubuntu-drivers-common
sudo ubuntu-drivers autoinstall
sudo reboot
```

### Step 3.2: Configure Swap Space (Recommended for <16 GB RAM instances)
```bash
sudo fallocate -l 4G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
```

### Step 3.3: Clone Repository & Create Virtual Environment
```bash
git clone <your-repository-url> satquery
cd satquery

python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip setuptools wheel
```

### Step 3.4: Install Python Dependencies
```bash
# If running with NVIDIA GPU CUDA 12.1:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# If running on CPU:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Install application dependencies:
pip install -r requirements-prod.txt
```

### Step 3.5: Install & Start Ollama
```bash
curl -fsSL https://ollama.com/install.sh | sh

# Pull the primary reasoning model in background
ollama pull qwen3:8b
```

### Step 3.6: Configure Environment
```bash
cp .env.example .env
```
Ensure `ENABLE_MODEL_WARMUP=false` in `.env` so server startup is instantaneous.

---

## 4. Starting the Application

### Option A: Direct Python Execution
```bash
source .venv/bin/activate
python satquery_app.py 8080
```

### Option B: Production ASGI Server (Uvicorn / Gunicorn)
```bash
source .venv/bin/activate
uvicorn satquery.server.fastapi_app:app --host 0.0.0.0 --port 8080 --workers 2
```

### Option C: Systemd Service (Recommended for Production Auto-Restart)
Create a systemd service file:
```bash
sudo nano /etc/systemd/system/satquery.service
```

Paste the following configuration (adjust `/home/ubuntu/satquery` to your path):
```ini
[Unit]
Description=SatQuery AI Remote Sensing Assistant
After=network.target ollama.service

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/home/ubuntu/satquery
EnvironmentFile=/home/ubuntu/satquery/.env
ExecStart=/home/ubuntu/satquery/.venv/bin/python satquery_app.py 8080
Restart=always
RestartSec=5
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
```

Enable and start the service:
```bash
sudo systemctl daemon-reload
sudo systemctl enable satquery
sudo systemctl start satquery
sudo systemctl status satquery
```

---

## 5. Verification & Health Checks

### Check System Health:
```bash
curl -s http://localhost:8080/health | jq
```
Expected response:
```json
{
  "status": "healthy",
  "system": "SatQuery AI Remote Sensing Assistant",
  "version": "2.0.0",
  "capabilities": {
    "web_ui": "ready",
    "fastapi_server": "running",
    "ollama_reasoning": "online",
    "tts_speech": "ready",
    "tts_provider": "kokoro"
  }
}
```

### Check JARVIS Readiness:
```bash
curl -s http://localhost:8080/api/v1/jarvis/status | jq
```

### Test Neural Speech Generation:
```bash
curl -s -X POST http://localhost:8080/api/v1/jarvis/tts \
  -H "Content-Type: application/json" \
  -d '{"text": "SatQuery AI deployment verified successfully.", "voice": "af_heart"}' | jq .success
```

---

## 6. Troubleshooting & Gotchas

1. **Frontend Assets Failing to Load from Remote Browser**:
   - Verify that your EC2 Security Group permits inbound TCP traffic on port `8080`.
   - Access via `http://<EC2_PUBLIC_IP>:8080/`. The frontend uses root-relative `/static/...` assets and binds cleanly to any public IP.
2. **Audio Generation Error (`soundfile` / `libsndfile` missing)**:
   - If audio generation reports library load errors, install: `sudo apt install -y libsndfile1`.
3. **Ollama Connection Refused (`503 Ollama unavailable`)**:
   - Check if Ollama is running: `systemctl status ollama` or run `ollama serve`.
   - Test connectivity: `curl http://localhost:11434/api/tags`.
4. **Out of Memory (OOM) Killed during Model Inference**:
   - Ensure the swapfile is active (`free -h`).
   - Leave `ENABLE_MODEL_WARMUP=false` to prevent eager 2B model preloading into system RAM.
5. **Inspecting Server Logs**:
   ```bash
   sudo journalctl -u satquery -f -n 100
   ```

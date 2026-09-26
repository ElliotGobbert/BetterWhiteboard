# 🛠️ Setup & Run Guide

We use **[uv](https://docs.astral.sh/uv/)** to manage our Python environment. It is significantly faster than standard `pip` and ensures we all have the exact same package versions via `uv.lock`.

## 1. Install `uv`
If you don't have `uv` installed on your machine yet, install it:

**Mac / Linux (Jetson/WSL):**
```bash
curl -LsSf [https://astral.sh/uv/install.sh](https://astral.sh/uv/install.sh) | sh

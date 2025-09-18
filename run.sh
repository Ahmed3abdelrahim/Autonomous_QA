#!/bin/bash
set -euo pipefail

export MODEL_DIR=out-autonet-char
export DATA_DIR=data/autonet_qa
export DEVICE=cuda

uvicorn server.app:app --host 0.0.0.0 --port 8080

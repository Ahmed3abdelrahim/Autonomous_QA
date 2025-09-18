#!/bin/bash
set -euo pipefail

curl -X POST http://localhost:8080/generate \
  -H "Content-Type: application/json" \
  -d '{"question":"What is an autonomous network in simple terms?"}'

#!/usr/bin/env bash
# =============================================================================
# run_all.sh — Compila e roda todos os estágios do projeto
# =============================================================================
set -euo pipefail

cd "$(dirname "$0")/.."

echo "============================================"
echo " Flash Attention SM75 — Register-Fused"
echo "============================================"
echo ""

# ── Compilar ────────────────────────────────────────────────────────────────
echo "[BUILD] Compilando..."
make -j4 2>&1 | grep -E "(error|warning|ptxas|Compiling|Built)" || true

echo ""

# ── Fase 0: Probe do layout ──────────────────────────────────────────────────
echo "============================================"
echo " FASE 0: Probe do Layout WMMA"
echo "============================================"
./probe_layout

echo ""

# ── Fase 1: Teste do softmax ─────────────────────────────────────────────────
echo "============================================"
echo " FASE 1: Teste Warp Register Softmax"
echo "============================================"
./test_warp_softmax

echo ""

# ── Fase 2: Kernel fusionado ─────────────────────────────────────────────────
echo "============================================"
echo " FASE 2: Kernel Fusionado (Validacao + Bench)"
echo "============================================"

echo "--- N=64, d=64 ---"
./flash_fused 64 64 200

echo ""
echo "--- N=128, d=64 ---"
./flash_fused 128 64 200

echo ""
echo "--- N=256, d=64 ---"
./flash_fused 256 64 100

echo ""
echo "Todos os testes concluidos."


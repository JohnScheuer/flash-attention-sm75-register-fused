# Análise de Performance — Flash Attention SM75

## O Problema Real: Aritmética vs Latência de Memória

### O que está acontecendo

O kernel atual tem 1 warp por bloco com 96 registradores por thread.

**RTX 2070 (SM75):**
- 46 SMs
- 65.536 registradores por SM
- 32 warps máximo por SM

**Cálculo de ocupação:**
  max_warps_por_SM = 65536 / (32 threads × 96 regs) = 21 warps
  ocupação = 21/32 = 65% — não é ruim em si

**O problema real é outro:**
  N=256, d=64 → 16 blocos de Q
  16 blocos / 46 SMs = 0.35 blocos por SM

  Apenas 16 dos 46 SMs estão ativos!
  Os outros 30 SMs ficam completamente ociosos.

### Por que N pequeno mata a performance

O Flash Attention original foi projetado para N >> 1000 (sequências longas).
Para N=256 com blocos de 16 linhas → só 16 blocos no grid inteiro.
A RTX 2070 tem 46 SMs mas processa apenas 16 blocos simultaneamente.

**Regra de ouro:** Para saturar a GPU, precisamos de:
  blocos_no_grid >> SMs × warps_por_SM = 46 × 21 = ~966 warps simultâneos

### O que vamos fazer: Multi-Warp + Maior Tile

Solução 1 (imediata): Adicionar dimensão de batch/heads ao grid
  grid = (N/16, num_heads, batch_size)
  Com 8 heads, batch=4: grid = (16, 8, 4) = 512 blocos → 512/46 = 11 por SM ✓

Solução 2 (mais impacto): Tile maior — 2 warps por bloco processando 32 linhas
  Dobra o trabalho por bloco, melhora reuso de K/V

Solução 3 (mais impacto ainda): Shared memory de K/V com prefetch
  Um warp carrega K/V enquanto o outro processa → latência escondida


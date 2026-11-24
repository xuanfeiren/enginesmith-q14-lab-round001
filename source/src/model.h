// Qwen3-14B shapes and the engine's compiled configuration (must match kernels/targets.py).
#pragma once
constexpr int W = 2;                     // tensor-parallel ranks
constexpr int C = 2 * W;                 // virtual cores
constexpr int L = 40, HID = 5120, HT = HID / 128, NQ = 40, NKV = 8, D = 128, INTER = 17408, VOCAB = 151936;
constexpr int NQL = NQ / C, NKL = NKV / C, ITL = INTER / 128 / C;
constexpr int VT = (VOCAB + 127) / 128, NVL = (VT + C - 1) / C;
constexpr int TMAX = 4608;
constexpr int MAXPOS = TMAX;             // positions a request may reach
constexpr double ROPE_THETA = 1000000.0;

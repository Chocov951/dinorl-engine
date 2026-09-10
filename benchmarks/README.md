# DinoRL Python benchmarks

The baseline measures complete deterministic matches on `arena_mvp_v1`, with
replay disabled, all nine ordered pairs of scripted controllers, and a fixed
seed list. Warm-up repetitions are run before and excluded from every metric.

Run the required process matrix with:

```powershell
.venv\Scripts\python.exe benchmarks\benchmark_matches.py `
  --processes 1 2 4 8 `
  --repetitions 5 `
  --warmup-repetitions 1 `
  --seed-count 20 `
  --output benchmarks\results\baseline-2026-09-10.json
```

The JSON report records matches and actions per second, mean and p95 match
duration, speedup relative to one process, maximum peak resident memory among
the participating processes, repetition timings, and the Python/CPU/system
environment. On Windows, peak memory uses `GetProcessMemoryInfo`; on Unix it
uses `getrusage`.

This first conforming result is a measurement baseline, not a throughput SLA.
The short performance test validates the report and workload contracts without
asserting an arbitrary speed target.

## Replay cost

The replay benchmark runs the same nine ordered controller pairs and fixed
seeds once with replay disabled and once with replay enabled. Its timed replay
path includes replay recording, SHA-256 calculation, and compact UTF-8 JSON
serialization of both the replay and the complete `simulate-response-v1`
document. It reports throughput for both paths and p50, p95, and maximum byte
sizes for replay and complete response samples.

Run the archived measurement with:

```powershell
.venv\Scripts\python.exe -m benchmarks.benchmark_replay `
  --repetitions 5 `
  --warmup-repetitions 1 `
  --seed-count 20 `
  --output benchmarks\results\replay-cost-2026-09-10.json
```

The command refuses fewer than five measured repetitions or an absent warm-up,
and fails instead of archiving a report if any replay exceeds 480 KiB or any
complete response exceeds 512 KiB.

# Current-binary replica of the July 15 Llama-1B GPU μKV-mass + watchdog-v3 run

This directory is a complete raw capture of a **protocol replica**, not a
byte-identical reproduction of the historical executable.

## Fixed controls

- Model, prompt length, decode length, seed, context, policy flags, cache type,
  and watchdog-v3 script match the visible July 15 record.
- Prompt SHA-256: `50c4497fd8c5cf2555f4a7e5bb3be22f962bc86c37519ad10a74ce5c9f32be80`.
- Current Vulkan binary SHA-256:
  `e97a23f8f293a0ec29e12cb06e9ec5a7b976c79d1eb6d1e69bbbfac988e002a2`.
- The source/binary SHA-256 guard passed immediately before launch.
- GPU-active prefill samples were all at 1200 MHz.

## Captured result

| Metric | July 15 artifact | Current v3 replica |
|---|---:|---:|
| Prefill | 117.0 s | 131.1 s |
| Decode | 31.9 tok/s | 27.0 tok/s |
| Total wall | 245.4 s | 283.3 s |
| Peak KV cells | 12,542 | 12,542 |
| Peak GPUSS / DDR | 86.9 / 79.9 C | 79.9 / 73.3 C |
| USB energy | 376.4 mWh | 249.8 mWh |

The watchdog started but recorded no cap action. During decode, the vendor GPU
governor nevertheless reduced clocks (44.5% of GPU-active samples were below
1200 MHz). Therefore the numerical delta is environmental/governor variation;
it must not be attributed to μKV or treated as a historical-binary comparison.

## Raw files

- `meta.json`, `stderr.log`, `stdout.log`, `gen_steps.csv`: benchmark result.
- `sensors.csv`: 5 Hz system and thermal trace.
- `watchdog.log`: watchdog-v3 action record.
- `prompt.txt`, `prompt.sha256`, `binary.sha256`: exact inputs and binary ID.
- `started_at.txt`, `finished_at.txt`, `exit_status.txt`, `DONE`: run lifecycle.

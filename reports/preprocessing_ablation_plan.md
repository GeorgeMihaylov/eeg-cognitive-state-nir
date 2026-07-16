# Preprocessing ablation plan

| Trial | Band-pass | Notch | CAR | Hash | Cache | Existing result | Action | Estimated new cache |
|---|---:|---:|---:|---|---|---|---|---:|
| A | false | false | false | `97bcf690085e85b3` | reuse `data\interim\raw_eeg_cache_w10_v3\raw-2251ca950a467267` | none | reuse_cache_and_run | 0 |
| B | true | false | false | `cf250ae33309023f` | missing | none | build_cache_and_run | 6511440533 |
| C | false | true | false | `3754791a2ee2375a` | missing | none | build_cache_and_run | 6511440533 |
| D | false | false | true | `3c3bd78a85d95e6b` | missing | none | build_cache_and_run | 6511440533 |
| E | true | true | false | `b5ede23d23e06dbc` | missing | none | build_cache_and_run | 6511440533 |
| F | true | false | true | `94b211d9f199ea7d` | missing | none | build_cache_and_run | 6511440533 |
| G | false | true | true | `399f7f260d322a7a` | missing | none | build_cache_and_run | 6511440533 |
| H | true | true | true | `a3b01b3aa0ec6eef` | reuse `data\interim\raw_eeg_cache_w10_v3\raw-bp-notch-car-445be3721678be51` | none | reuse_cache_and_run | 0 |

Existing reusable caches: **2**.
Missing caches: **6**.
Estimated new cache bytes: **39068643198**.
Planned benchmark runs: **8**.

## Capacity check

- Free bytes on F: 920917348352
- Estimated bytes for six missing caches: 39068643198
- Required safety reserve: 21474836480
- Capacity decision: sufficient; cache build remains gated behind explicit --build-missing-caches.

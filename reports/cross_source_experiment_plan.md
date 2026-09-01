# Cross-source experiment plan

| Trial | Direction | Mode | Model | Unit | Train | Test | Subjects | Status | Action | Est. seconds |
|---|---|---|---|---|---:|---:|---|---|---|---:|
| `gpn_data_to_old_eeg__source_exclusive__random_forest` | gpn_data -> Old_EEG | source_exclusive | random_forest | feature_window | 6348 | 6717 | 11 / 12 | valid | reuse_completed | 30.0 |
| `gpn_data_to_old_eeg__source_exclusive__torch_transformer` | gpn_data -> Old_EEG | source_exclusive | torch_transformer | feature_sequence | 6165 | 6496 | 11 / 12 | valid | reuse_completed | 180.0 |
| `gpn_data_to_old_eeg__shared_subject__random_forest` | gpn_data -> Old_EEG | shared_subject | random_forest | feature_window | 263 | 456 | 1 / 1 | invalid | skip_invalid | 30.0 |
|  |  |  |  | invalid reasons |  |  |  | train subjects=1 is below configured minimum 5; test subjects=1 is below configured minimum 3 |  |  |
| `gpn_data_to_old_eeg__shared_subject__torch_transformer` | gpn_data -> Old_EEG | shared_subject | torch_transformer | feature_sequence | 256 | 449 | 1 / 1 | invalid | skip_invalid | 180.0 |
|  |  |  |  | invalid reasons |  |  |  | train subjects=1 is below configured minimum 5; test subjects=1 is below configured minimum 3 |  |  |
| `old_eeg_to_gpn_data__source_exclusive__random_forest` | Old_EEG -> gpn_data | source_exclusive | random_forest | feature_window | 6717 | 6348 | 12 / 11 | valid | reuse_completed | 30.0 |
| `old_eeg_to_gpn_data__source_exclusive__torch_transformer` | Old_EEG -> gpn_data | source_exclusive | torch_transformer | feature_sequence | 6496 | 6165 | 12 / 11 | valid | reuse_completed | 180.0 |
| `old_eeg_to_gpn_data__shared_subject__random_forest` | Old_EEG -> gpn_data | shared_subject | random_forest | feature_window | 456 | 263 | 1 / 1 | invalid | skip_invalid | 30.0 |
|  |  |  |  | invalid reasons |  |  |  | train subjects=1 is below configured minimum 5; test subjects=1 is below configured minimum 3 |  |  |
| `old_eeg_to_gpn_data__shared_subject__torch_transformer` | Old_EEG -> gpn_data | shared_subject | torch_transformer | feature_sequence | 449 | 256 | 1 / 1 | invalid | skip_invalid | 180.0 |
|  |  |  |  | invalid reasons |  |  |  | train subjects=1 is below configured minimum 5; test subjects=1 is below configured minimum 3 |  |  |

Valid trials: **4**.
Invalid trials: **4**.
Planned runs: **0**.

## Trial details

- `gpn_data_to_old_eeg__source_exclusive__random_forest`: logical recordings train/test 21 / 14; shared subjects 31 total / 1 with residual data in both sources; removed duplicate logical recordings 33; minimum test predictions per subject 182.
  Train classes: 0:1475 / 1:1468 / 2:1248 / 3:1097 / 4:1060; test classes: 0:1602 / 1:1277 / 2:1172 / 3:1282 / 4:1384.
- `gpn_data_to_old_eeg__source_exclusive__torch_transformer`: logical recordings train/test 21 / 14; shared subjects 31 total / 1 with residual data in both sources; removed duplicate logical recordings 33; minimum test predictions per subject 162.
  Train classes: 0:1409 / 1:1421 / 2:1220 / 3:1068 / 4:1047; test classes: 0:1527 / 1:1242 / 2:1148 / 3:1259 / 4:1320.
- `gpn_data_to_old_eeg__shared_subject__random_forest`: logical recordings train/test 1 / 1; shared subjects 31 total / 1 with residual data in both sources; removed duplicate logical recordings 33; minimum test predictions per subject 456.
  Train classes: 0:130 / 1:64 / 2:43 / 3:24 / 4:2; test classes: 0:139 / 1:93 / 2:99 / 3:77 / 4:48.
- `gpn_data_to_old_eeg__shared_subject__torch_transformer`: logical recordings train/test 1 / 1; shared subjects 31 total / 1 with residual data in both sources; removed duplicate logical recordings 33; minimum test predictions per subject 449.
  Train classes: 0:123 / 1:64 / 2:43 / 3:24 / 4:2; test classes: 0:139 / 1:92 / 2:95 / 3:75 / 4:48.
- `old_eeg_to_gpn_data__source_exclusive__random_forest`: logical recordings train/test 14 / 21; shared subjects 31 total / 1 with residual data in both sources; removed duplicate logical recordings 33; minimum test predictions per subject 155.
  Train classes: 0:1602 / 1:1277 / 2:1172 / 3:1282 / 4:1384; test classes: 0:1475 / 1:1468 / 2:1248 / 3:1097 / 4:1060.
- `old_eeg_to_gpn_data__source_exclusive__torch_transformer`: logical recordings train/test 14 / 21; shared subjects 31 total / 1 with residual data in both sources; removed duplicate logical recordings 33; minimum test predictions per subject 148.
  Train classes: 0:1527 / 1:1242 / 2:1148 / 3:1259 / 4:1320; test classes: 0:1409 / 1:1421 / 2:1220 / 3:1068 / 4:1047.
- `old_eeg_to_gpn_data__shared_subject__random_forest`: logical recordings train/test 1 / 1; shared subjects 31 total / 1 with residual data in both sources; removed duplicate logical recordings 33; minimum test predictions per subject 263.
  Train classes: 0:139 / 1:93 / 2:99 / 3:77 / 4:48; test classes: 0:130 / 1:64 / 2:43 / 3:24 / 4:2.
- `old_eeg_to_gpn_data__shared_subject__torch_transformer`: logical recordings train/test 1 / 1; shared subjects 31 total / 1 with residual data in both sources; removed duplicate logical recordings 33; minimum test predictions per subject 256.
  Train classes: 0:139 / 1:92 / 2:95 / 3:75 / 4:48; test classes: 0:123 / 1:64 / 2:43 / 3:24 / 4:2.

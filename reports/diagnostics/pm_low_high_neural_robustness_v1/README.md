# PM LOW/HIGH neural robustness v1

Screening experiment for the frozen extreme-state target contract.

- reference LOW/HIGH protocol: `ac07a43b2554a2f178e9a63e28a1462cee253d613f122516e26ac0dcfe6c7431`
- new protocol hash: `e902f4dbe8f317be4ac6ed5061104cf5a3399eea5125414a675add88f4105a8d`
- models: ShallowConvNet, LSTM, Transformer
- PM / folds / fits: 7 / 5 / 105
- ShallowConvNet: raw EEG(t-10s), historical fixed configuration
- LSTM: 10 feature windows ending at t-10s, historical fixed configuration
- Transformer: 8 feature windows ending at t-10s, historical fixed configuration
- thresholds: exact outer-train Q33/Q67 from the completed LOW/HIGH contract
- inner validation: record-group disjoint
- training executed by dry-run: false

This is screening, not direct cross-architecture ranking. Sequence models are
evaluated on history-eligible cohorts; a matched-cohort tabular follow-up is
required before claiming an architecture advantage.

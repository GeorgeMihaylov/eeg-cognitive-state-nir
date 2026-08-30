# PM LOW/HIGH personalization feasibility v1

No model training or inference is performed.

Chronology:
- calibration record: earliest logical recording by actual selected-record UTC start
- calibration never crosses a logical-record boundary
- budgets: 0, 30, 60, 120, 300 seconds of elapsed recording time
- no scanning forward until LOW/HIGH labels appear
- fixed evaluation: exact-lag targets strictly after +300 s UTC boundary
- middle PM values are counted but excluded from the binary task
- missing PM values are counted separately

References:
- LOW/HIGH protocol: `ac07a43b2554a2f178e9a63e28a1462cee253d613f122516e26ac0dcfe6c7431`
- matched model-selection protocol: `e09f28dab2b37321dd665cc55653cfc08a5a29afc38927ee26bc2d2c6cc988e7`
- future personalization candidates: `xgboost`, `lightgbm`

Protocol hash: `f3961235894a7c3a931bac21b482641b1b8b753c7ef00f6d6d2184b4d304762b`
Audit executed by dry-run: `false`

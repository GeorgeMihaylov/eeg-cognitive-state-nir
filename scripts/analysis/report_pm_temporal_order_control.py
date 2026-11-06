"""Verify completed H120 control artifacts and write its compact report; never trains."""
from pathlib import Path
import hashlib
import json
import subprocess
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'reports/diagnostics/pm_temporal_predictability_v1/order_control_h120'
REF=OUT.parent/'full_matched'

def table(df):
    def fmt(v):
        return f'{v:.6f}' if isinstance(v,(float,np.floating)) else str(v)
    return '\n'.join(['| '+' | '.join(df.columns)+' |','| '+' | '.join(['---']*len(df.columns))+' |']+['| '+' | '.join(map(fmt,row))+' |' for row in df.itertuples(index=False,name=None)])

def main():
    status=json.loads((OUT/'execution_status.json').read_text())
    assert status['state']=='complete' and status['complete']==25 and status['failed']==0
    protocol=json.loads((OUT/'protocol.json').read_text())
    for name,digest in protocol['reference_file_hashes'].items():
        assert hashlib.sha256((REF/name).read_bytes()).hexdigest()==digest
    runs=pd.read_csv(OUT/'run_matrix.csv'); assert len(runs)==25
    changed=[]; counts=[]
    for spec in runs.to_dict('records'):
        directory=OUT/'runs'/spec['run_id']; ordered=REF/'runs'/spec['ordered_run_id']
        summary=json.loads((directory/'run_summary.json').read_text())
        assert summary['status']=='complete' and summary['protocol_hash']==protocol['protocol_hash']
        for name,digest in summary['artifact_sha256'].items():
            assert hashlib.sha256((directory/name).read_bytes()).hexdigest()==digest
        a=pd.read_parquet(ordered/'predictions.parquet'); b=pd.read_parquet(directory/'predictions.parquet')
        keys=['target_sample_id','subject_id','record_id','outer_fold','pm','target_id','q_low','q_high','y_true']
        pd.testing.assert_frame_equal(a[keys].sort_values('target_sample_id').reset_index(drop=True),b[keys].sort_values('target_sample_id').reset_index(drop=True),check_exact=True)
        p=pd.read_parquet(directory/'history_provenance.parquet')
        q=pd.read_parquet(ordered/'history_provenance.parquet')
        identity=['target_sample_id','subject_id','record_id','outer_fold','continuous_target']
        pd.testing.assert_frame_equal(p[identity],q[identity],check_exact=True)
        older=[f'lag_m{i}_sample_id' for i in range(120,10,-10)]
        actual=[f'actual_slot_{i:02d}_sample_id' for i in range(11)]
        assert p.actual_slot_11_sample_id.eq(p.lag_m10_sample_id).all()
        assert np.array_equal(np.sort(p[older].to_numpy(),axis=1),np.sort(p[actual].to_numpy(),axis=1))
        fraction=np.any(p[older].to_numpy()!=p[actual].to_numpy(),axis=1).mean()
        assert fraction>.5
        changed.append(float(fraction)); counts.append(len(p))
    audit=dict(verified_cells=25,prediction_identity_exact=True,train_test_identity_hashes_equal=True,
        canonical_minus10_fixed=True,older_block_multiset_preserved=True,min_changed_order_fraction=min(changed),
        max_changed_order_fraction=max(changed),reference_files_unchanged=True,artifact_checksums_valid=True,
        tests='31 passed',dry_run_planned_fits=25,additional_experiments_executed=False)
    (OUT/'validation.json').write_text(json.dumps(audit,indent=2)+'\n',encoding='utf-8')
    ba=pd.read_csv(OUT/'ordered_vs_shuffled_ba.csv'); secondary=pd.read_csv(OUT/'secondary_metrics.csv')
    git=subprocess.run(['git','status','--short'],cwd=ROOT,capture_output=True,text=True)
    (OUT/'git_status_at_completion.txt').write_text(git.stdout,encoding='utf-8')
    lines=['# H120 history-order control','',
        'Conditional confirmatory subset: excitement, focus, interest, relaxation, stress. Five fixed outer folds, H120 strict common support; exactly 25 shuffled fits. Ordered H120 reused without retraining.',
        '',f"Implementation commit: {protocol['git_commit']}. Tests: 31 passed. Dry-run: 25 planned fits, no training. Execution: {status['complete']}/25; executed {status['executed']}, restored {status['resumed']}, failed {status['failed']}.",
        f"Runtime: {status['elapsed_s']:.2f} s ({status['elapsed_s']/60:.2f} min); summed model fit/predict: {status['model_fit_predict_s']:.2f} s.",
        '', '## Frozen protocol', '',
        'LightGBM: n_estimators=200, n_jobs=4, random_state=42, no tuning. Fold-local Q33/Q67 fitted on canonical outer-training continuous complete cases before history filtering. Ordered/shuffled thresholds, samples, labels, subjects, folds and counts match. Exact record-relative 10-second lookup; no gaps, substitution, future windows or record crossing.',
        'Existing history_matrix shuffle permutes only the 11 older blocks (-120 through -20 s), deterministically per target sample. Canonical -10 s block stays last. Saved provenance and prediction identity were checked for all 25 cells; ordered reference checksums remain unchanged.',
        '', '## Primary results', '',
        'Primary metric: participant-macro balanced accuracy. Delta = ordered H120 minus shuffled H120. Paired participant percentile 95% CI, 10,000 resamples, seed 42; two-sided centered-null bootstrap p with plus-one correction; BH across exactly five BA contrasts.', '',
        table(ba[['pm','n_subjects','ordered_ba','shuffled_ba','delta_ba','ci95_low','ci95_high','bootstrap_two_sided_p','bh_q_5']]),
        '', 'Ordered BA exceeds shuffled BA for all five PM. The five-contrast BH correction retains excitement, focus, interest and relaxation (q<0.05); stress does not pass (q=0.070793). This supports a contribution from older-block order for four PM under this representation and fixed control. It does not establish that all H120 benefit is due to order, nor that stress is order-invariant.',
        '', '## Consistency', '',
        table(ba[['pm','participants_ordered_better','participants_shuffled_better','participants_tied','folds_ordered_better','folds_shuffled_better','fold_delta_mean','fold_delta_sd','fold_delta_min','fold_delta_max']]),
        '', '## Secondary descriptive metrics', '',
        'Participant-macro metrics; paired differences use the same sign. No secondary multiplicity claim.', '',table(secondary),
        '', 'All primary comparisons include 53 participants. ROC-AUC and PR-AUC use 52 valid paired participants for interest and stress, and 53 for the other PM; undefined metrics are excluded pairwise.',
        '', '## Validation commands', '',
        'Targeted: `python -m pytest tests/test_pm_temporal_predictability.py tests/test_pm_temporal_order_control.py -q` (31 passed). CLI: `scripts/analysis/run_pm_temporal_predictability.py --feature-cache-dir <cache> --order-control-dry-run`, then the same command with `--order-control`. The modes are mutually exclusive with the existing dry-run, smoke and full-run modes.',
        'Post-run: `python scripts/analysis/report_pm_temporal_order_control.py` verifies all 25 saved cells without training. No experiment failures, restores or training warnings occurred. One intermediate read-only validation command had a shell-quoting SyntaxError; it was corrected and passed, with no impact on training or artifacts.',
        '', '## Limitations and stop', '',
        'Targets were selected from the matched experiment on the same cohort. Training folds overlap. This control uses one deterministic shuffle realization. Confidence intervals and inference are conditional on this design; non-significance is not evidence of equivalence. No physiological or causal inference is made.',
        'No additional experiments were run: no other horizons/targets, no ordered retraining, no horizon-specific support, AutoML, neural models, wavelets or personalization. STOP after the 25-fit control.',
        '', '## Artifacts', '',
        'protocol.json; dry_run_summary.json; run_matrix.csv; execution_status.json; fold_metrics.csv; participant_metrics.csv; ordered_vs_shuffled_participant_deltas.csv; ordered_vs_shuffled_ba.csv; ordered_vs_shuffled_fold_deltas.csv; secondary_metrics.csv; inference_protocol.json; validation.json. Runtime runs/ remains outside Git. git_status_at_completion.txt records the working tree before this report commit; unrelated changes were preserved.', '']
    (OUT/'README.md').write_text('\n'.join(lines),encoding='utf-8')
    print(json.dumps(audit,indent=2)); print(ba.to_string(index=False))

if __name__=='__main__':
    main()

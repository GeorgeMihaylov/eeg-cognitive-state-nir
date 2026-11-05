from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from bench.experiments import pm_temporal_predictability as history
from bench.experiments import pm_low_high_q3_extremes_confirmatory as low

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "experiments/pm_diagnostics/pm_temporal_predictability_v1.json"


def frame():
    rows = []
    for fold in range(1, 6):
        for w in range(40):
            rows.append({"sample_id": f"s{fold}_{w}", "record_id": f"r{fold}", "subject_id": f"s{fold}",
                         "outer_fold": fold, "t_start": -5+w*10., "t_end": 5+w*10.,
                         **{f"target_{pm}": float(w % 7) for pm in low.PM_NAMES}})
    return pd.DataFrame(rows)


def base_context(full):
    positions, masks, _ = history.history_positions(full)
    cohorts = {pm: history.cohort_for(full, masks[10], pm) for pm in low.PM_NAMES}
    transforms, audit = low.build_threshold_audit(full, cohorts, [1, 2, 3, 4, 5])
    return SimpleNamespace(full=full, matrix=np.zeros((len(full), 371), dtype=np.float32),
        feature_names=[f"spectral__f{i}" for i in range(371)], transforms=transforms, threshold_audit=audit,
        protocol={"fixed_fold_hash": history.FOLD_HASH, "protocol_hash": "synthetic"},
        cache_identity={"n_features": 371}, root=ROOT)


@pytest.mark.parametrize("h,dim", [(10,371),(20,742),(30,1113),(60,2226),(120,4452)])
def test_exact_history_and_dimensions(h, dim):
    f = frame()
    pos, masks, _ = history.history_positions(f)
    targets = np.flatnonzero(masks[h])
    matrix = np.broadcast_to(np.arange(len(f))[:,None], (len(f),371)).astype(np.float32)
    x = history.history_matrix(matrix, pos, targets, h)
    assert x.shape == (len(targets), dim)
    np.testing.assert_array_equal(x[:,-371:], matrix[pos[-1][targets]])
    for j, lag in enumerate(history.HORIZONS[h]):
        np.testing.assert_array_equal(x[:, j*371:(j+1)*371], matrix[pos[lag][targets]])
        assert np.all(f.t_end.to_numpy()[pos[lag][targets]] <= f.t_start.to_numpy()[targets])
    names = history.feature_names(["a","b"], h)
    assert names[-2:] == ['lag_m10__a','lag_m10__b']
    assert names[0] == f'lag_m{h}__a'


def test_gaps_nested_support_no_substitution_or_record_crossing():
    f = frame().drop(index=15).reset_index(drop=True)
    pos, masks, _ = history.history_positions(f)
    after = f.index[f.sample_id.eq('s1_16')][0]
    assert pos[-1][after] == -1
    assert not masks[10][after]
    assert not masks[120][after+5]
    for h, next_h in zip([10,20,30,60],[20,30,60,120]):
        assert not (masks[next_h] & ~masks[h]).any()
    starts = np.flatnonzero(f.t_start.eq(-5))
    assert (pos[-1][starts] == -1).all()
    with pytest.raises(ValueError, match='Missing exact'):
        history.history_matrix(np.zeros((len(f),371)), pos, np.array([after]), 10)


@pytest.mark.parametrize('key,value', [('subject_id','other'),('outer_fold',2),('t_start',1.2),('t_end',1.2)])
def test_reject_identity_or_timestamp_corruption(key,value):
    f=frame()
    f.loc[3,key]=value
    with pytest.raises((ValueError,RuntimeError)):
        history.history_positions(f)


def test_thresholds_frozen_before_filtering_and_primary_hash_equality():
    f=frame()
    config=history.load_config(CONFIG)
    base=base_context(f)
    ctx=history.prepare_from_base(base,config,ROOT/'unused')
    assert len(ctx.run_matrix)==350
    primary=ctx.run_matrix.query('support_mode == @history.PRIMARY')
    columns=['threshold_hash','train_label_hash','test_label_hash','train_sample_hash','test_sample_hash',
             'train_participant_hash','test_participant_hash','train_fold_hash','test_fold_hash']
    assert primary.groupby(['pm','outer_fold'])[columns].nunique().eq(1).all().all()
    expected=low.fit_extreme_thresholds(f.loc[f.outer_fold.ne(1),'target_attention'])
    assert base.transforms[(1,'attention')]==expected
    f.loc[f.outer_fold.eq(1),'target_attention']=1000
    assert base_context(f).transforms[(1,'attention')]==expected
    smoke=history.prepare_from_base(base,config,ROOT/'unused',smoke=True)
    again=history.prepare_from_base(base,config,ROOT/'unused',smoke=True)
    pd.testing.assert_frame_equal(smoke.run_matrix,again.run_matrix)
    assert smoke.protocol['params']['n_estimators']==10
    assert len(smoke.run_matrix)==5
    with pytest.raises(ValueError,match='Only the five-fit'):
        history.run_smoke(ctx)


def test_shuffle_canonical_preserved_and_chunk_independent():
    f=frame(); pos,masks,_=history.history_positions(f)
    targets=np.flatnonzero(masks[120])
    matrix=np.arange(len(f)*2).reshape(len(f),2)
    ids=f.iloc[targets].sample_id.tolist()
    x=history.history_matrix(matrix,pos,targets,120,shuffled=True,sample_ids=ids).reshape(-1,12,2)
    ordered=history.history_matrix(matrix,pos,targets,120).reshape(-1,12,2)
    np.testing.assert_array_equal(x[:,-1],ordered[:,-1])
    np.testing.assert_array_equal(np.sort(x[:,:,0],axis=1),np.sort(ordered[:,:,0],axis=1))
    assert not np.array_equal(x[:,:-1],ordered[:,:-1])
    chunk=history.history_matrix(matrix,pos,targets[:5],120,shuffled=True,sample_ids=ids[:5])
    np.testing.assert_array_equal(chunk,x[:5].reshape(5,-1))


def test_paired_results_bootstrap_participants_not_windows():
    rows = [{"support_mode": history.PRIMARY, "pm": "attention", "subject_id": f"s{i}",
             "outer_fold": 1, "horizon_s": h, "balanced_accuracy": .5 + (h==20)*.1,
             "macro_f1": .4 + (h==20)*.1, "roc_auc": .7, "pr_auc": .7}
            for i in range(4) for h in (10,20)]
    paired, intervals = history.paired_history_results(pd.DataFrame(rows), bootstrap=True)
    np.testing.assert_allclose(paired.loc[paired.horizon_s.eq(20),'delta_balanced_accuracy'],.1)
    assert intervals.n_subjects.eq(4).all()
    with pytest.raises(ValueError,match='Duplicate'):
        history.paired_history_results(pd.DataFrame(rows+rows))

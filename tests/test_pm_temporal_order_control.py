from types import SimpleNamespace
import numpy as np
import pandas as pd
import pytest
from bench.experiments import pm_temporal_predictability as h
from bench.experiments import pm_temporal_order_control as c
from test_pm_temporal_predictability import frame, base_context, CONFIG


def test_shuffle_preserves_blocks_and_sample_permutation():
    f = frame(); pos, masks, _ = h.history_positions(f)
    targets = np.flatnonzero(masks[120]); ids = f.iloc[targets].sample_id.tolist()
    matrix = np.arange(len(f)*3).reshape(-1,3)
    ordered = h.history_matrix(matrix,pos,targets,120).reshape(-1,12,3)
    shuffled = h.history_matrix(matrix,pos,targets,120,shuffled=True,sample_ids=ids).reshape(-1,12,3)
    np.testing.assert_array_equal(ordered[:,-1],shuffled[:,-1])
    for a,b in zip(ordered[:,:-1],shuffled[:,:-1]):
        assert sorted(map(tuple,a)) == sorted(map(tuple,b))
    assert np.mean(np.any(ordered[:,:-1] != shuffled[:,:-1],axis=(1,2))) > .95
    reverse = h.history_matrix(matrix,pos,targets[::-1],120,shuffled=True,sample_ids=ids[::-1])
    np.testing.assert_array_equal(reverse,shuffled[::-1].reshape(-1,36))


def fixture_runs():
    base = base_context(frame())
    ctx = h.prepare_from_base(base,h.load_config(CONFIG),base.root/'unused',full=True)
    runs = ctx.run_matrix.loc[ctx.run_matrix.pm.isin(c.PMS)&ctx.run_matrix.horizon_s.eq(120)].copy()
    return base,runs


@pytest.mark.parametrize('field', ['train_label_hash','test_sample_hash','train_participant_hash','test_fold_hash','threshold_hash','n_train','q_low'])
def test_identity_rejects_tampering(field):
    _, ordered = fixture_runs(); altered = ordered.copy()
    c.compare_reference_identity(altered,ordered)
    altered.loc[altered.index[0],field] = 'corrupt' if field.endswith('hash') else altered.iloc[0][field]+1
    with pytest.raises((ValueError,AssertionError)):
        c.compare_reference_identity(altered,ordered)


def test_train_test_runtime_identity():
    base,runs = fixture_runs(); spec = runs.iloc[0].to_dict()
    _,masks,_ = h.history_positions(base.full)
    cohort = h.cohort_for(base.full,masks[120],spec['pm'])
    labels = base.transforms[(spec['outer_fold'],spec['pm'])].transform(cohort.continuous_target.to_numpy())
    c.validate_cell(None,spec,cohort,labels)
    changed = labels.copy(); i = np.flatnonzero(np.isfinite(changed))[0]; changed[i] = 1-changed[i]
    with pytest.raises(ValueError,match='hash'):
        c.validate_cell(None,spec,cohort,changed)


@pytest.mark.parametrize('field,value',[('pm','attention'),('horizon_s',60),('support_mode','horizon_specific_support'),('history_order','ordered'),('outer_fold',6)])
def test_control_rejects_other_cells(field,value):
    _,runs = fixture_runs(); runs['history_order']='shuffled'; runs.loc[runs.index[0],field]=value
    protocol = dict(params=h.PARAMS,mode='order_control_h120',primary_planned_fits=25,history_order_control='shuffled_older_blocks_canonical_fixed')
    with pytest.raises(ValueError,match='exactly 25'):
        c.validate(SimpleNamespace(run_matrix=runs,protocol=protocol))


def test_aggregation_paired_sign_and_five_contrasts(tmp_path):
    ref = tmp_path/'ref'; ref.mkdir(); out=tmp_path/'out'; out.mkdir()
    rows=[]; folds=[]
    for pm in c.PMS:
        for fold in range(1,6):
            rows.append(dict(pm=pm,subject_id=str(fold),outer_fold=fold,horizon_s=120,n_samples=10,n_low=5,n_high=5,**{m:.8 for m in c.METRICS}))
            folds.append(dict(pm=pm,outer_fold=fold,horizon_s=120,participant_macro_balanced_accuracy=.8))
    ordered=pd.DataFrame(rows); ordered.to_csv(ref/'participant_metrics.csv',index=False)
    of=pd.DataFrame(folds); of.to_csv(ref/'fold_metrics.csv',index=False)
    shuffled=ordered.copy(); shuffled[list(c.METRICS)]=.7
    sf=of.copy(); sf['participant_macro_balanced_accuracy']=.7
    ctx=SimpleNamespace(base=SimpleNamespace(root=tmp_path),protocol={'ordered_reference':'ref'},output=out)
    c.aggregate_order_control(ctx,sf,shuffled)
    result=pd.read_csv(out/'ordered_vs_shuffled_ba.csv')
    assert len(result)==5
    np.testing.assert_allclose(result.delta_ba,.1)
    np.testing.assert_allclose(result.ci95_low,.1)
    assert result.participants_ordered_better.eq(5).all()
    assert result.folds_ordered_better.eq(5).all()
    assert result.bh_q_5.lt(.05).all()
    shuffled.loc[0,'subject_id']='missing'
    with pytest.raises(ValueError,match='support differs'):
        c.aggregate_order_control(ctx,sf,shuffled)

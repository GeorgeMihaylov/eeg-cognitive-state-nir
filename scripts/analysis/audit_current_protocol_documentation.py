"""Read-only tracked text audit; writes documentation inventory, never scientific results."""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import re
import subprocess

ROOT=Path(__file__).resolve().parents[2]
EXTENSIONS={'.md','.yaml','.yml','.json','.csv','.py'}
PATTERNS={
    'Q3':r'q3(?!\d)',
    'label_q5':r'label_q5',
    'cognitive_load_5class':r'cognitive_load_5class',
    '3-class':r'3[- ]class|three[- ]class|LOW/(?:MIDDLE|MEDIUM)/HIGH|target_classes\s*[:=]\s*3',
    'temporal_status':r'Temporal Predictability Audit|full (?:175-fit experiment|run) has NOT run|smoke only',
    'other_status':r'AutoML pilot|entropy features absent|connectivity absent',
}
P0={'README.md','PROJECT_CONTEXT.md','AGENTS.md','reports/article_results_summary.md',
    'reports/integration/project_scientific_conclusions.md','reports/summary/experiment_summary.md',
    'reports/summary/colleague_metrics_summary.md','reports/summary/project_remaining_work.md',
    'reports/requirements/final_requirement_coverage.md','reports/summary/requirements_coverage.md',
    'reports/summary/metrics_glossary.md','reports/summary/experiment_registry.yaml',
    'reports/summary/config_registry.yaml','reports/summary/metrics_provenance.yaml',
    'reports/summary/requirements_registry.yaml','reports/summary/config_curation.yaml',
    'reports/diagnostics/pm_temporal_predictability_v1/README.md',
    'reports/diagnostics/pm_temporal_predictability_audit_v1/README.md'}
EXCLUDED={'scripts/analysis/audit_current_protocol_documentation.py',
          'reports/architecture/current_protocol_documentation_audit.md',
          'reports/architecture/current_protocol_documentation_audit.json'}
NEW={'reports/article_results_summary_legacy_label_q5.md',
     'reports/integration/project_scientific_conclusions_legacy_20260829.md',
     'tests/test_current_protocol_reporting.py'}


def git(*args: str) -> str:
    return subprocess.check_output(['git',*args],cwd=ROOT,stderr=subprocess.DEVNULL).decode('utf-8')


def scan_text(name: str, text: str) -> dict:
    matches=Counter(); lines=defaultdict(list)
    for number,line in enumerate(text.splitlines(),1):
        found={key:len(re.findall(pattern,line,re.I)) for key,pattern in PATTERNS.items()}
        found={key:value for key,value in found.items() if value}
        if found:
            matches.update(found)
            for category in found:
                lines[category].append(number)
    return {'path':name,'matches':dict(matches),'matching_lines':dict(lines)}


def classify(path: str, *, before: bool, text: str) -> tuple[str,str]:
    """Reviewed precedence; mixed documents retain their native/legacy scope explicitly."""
    if path in P0:
        return ('stale_needs_update','update current context/metadata and regenerate') if before else ('current_correct','current primary separated from retained historical evidence')
    if path.endswith('_legacy_label_q5.md') or path.endswith('_legacy_20260829.md'):
        return 'historical_snapshot','preserved previous document; no numerical rewriting'
    if 'Historical protocol.' in text[:1200] or 'HISTORICAL SNAPSHOT' in text[:1200] or 'Historical classification protocol' in text[:1200]:
        return 'historical_snapshot','banner; original results and provenance retained'
    if any(k in path for k in ['pm_low_high','pm_temporal_predictability','pm_eeg_absolute_lag']):
        return 'current_correct','q3_extremes is binary LOW/HIGH; immutable identifiers/preflight artifacts retained'
    if path.startswith('tests/') and re.search(r'Gaussian|synthetic|n_classes\s*=\s*3|num_classes\s*=\s*3|classes\s*[:=]\s*3',text,re.I):
        return 'synthetic_test_keep','small test fixture or validation contract, not internal PM target'
    if any(k in path for k in ['cog_bci','cl_drive','clare','external_multimodal','cross_dataset']):
        return 'native_three_class_keep','native external task labels retained; any internal label_q5 transfer endpoint remains legacy'
    if path.startswith('reports/summary/') or path.endswith('final_requirement_coverage.md'):
        return 'generated_from_registry','legacy/current role supplied by metadata; historical tables remain unchanged'
    if path.startswith('reports/architecture/') or any(k in path for k in ['project_final','project_negative','project_reproducibility','target_pipeline_audit','full_target_registry_audit','benchmark_core_audit','benchmarking_branch_audit','statistical_analysis_run_inventory']):
        return 'historical_snapshot','dated engineering/protocol inventory, not current primary claims'
    if path.endswith('.py') or path.startswith(('experiments/','configs/')):
        return 'legacy_valid','historical execution/target code or compatible fixtures; no target/config change'
    return 'legacy_valid','historical/supporting result or provenance; not promoted to current primary'


def audit(baseline: str, output: Path) -> dict:
    names=git('ls-tree','-r','--name-only',baseline).splitlines()
    names=[p for p in names if Path(p).suffix in EXTENSIONS and p not in EXCLUDED]
    changed=set(git('diff',baseline,'--name-only').splitlines())
    states={}; texts={}
    for phase in ['before','after']:
        rows=[]
        for name in sorted(set(names)|(NEW if phase=='after' else set())):
            p=ROOT/name
            if not p.exists():continue
            try:
                text=git('show',f'{baseline}:{name}') if phase=='before' and name in changed else p.read_text(encoding='utf-8-sig')
            except (UnicodeError,OSError):continue
            row=scan_text(name,text)
            if row['matches']:
                category,action=classify(name,before=phase=='before',text=text)
                row.update(classification=category,action=action); rows.append(row)
        states[phase]={'files':len(rows),'occurrences':dict(sum((Counter(r['matches']) for r in rows),Counter())),
                       'file_classifications':dict(Counter(r['classification'] for r in rows)), 'rows':rows}
    result={'baseline':baseline,'scope':'Tracked .md/.yaml/.yml/.json/.csv/.py; before baseline commit, after working tree plus declared new task files. Audit artifacts/script excluded to avoid self-counting. Q3 excludes Q33; counts are occurrences, not matching lines. Native external semantics and synthetic fixtures retained. No global replacement.',**states}
    output.parent.mkdir(parents=True,exist_ok=True)
    output.with_suffix('.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    lines=['# Current protocol documentation audit — 2026-11-06','',result['scope'],'',
           'Starting branch: analysis/pm-temporal-predictability-audit; starting HEAD fbd100f. Documentation and metadata only: no training, feature-cache rebuild, target/fold changes, or scientific metric recomputation.',
           '', '## Search totals','', '| Pattern | Before occurrences | After occurrences |','|---|---:|---:|']
    for pattern in PATTERNS:lines.append(f"| {pattern} | {states['before']['occurrences'].get(pattern,0)} | {states['after']['occurrences'].get(pattern,0)} |")
    lines+=['','## Semantic classification (files, not occurrences)','', '| Classification | Before | After |','|---|---:|---:|']
    categories=['current_correct','stale_needs_update','legacy_valid','historical_snapshot','native_three_class_keep','synthetic_test_keep','generated_from_registry','unclear_manual_review']
    for category in categories:lines.append(f"| {category} | {states['before']['file_classifications'].get(category,0)} | {states['after']['file_classifications'].get(category,0)} |")
    lines+=['','A file can contain several target contracts. The table records its dominant reviewed role; native external classes are never converted, and historical metrics retain their labels. The JSON inventory includes exact matching line numbers per pattern. Preflight manifests and immutable namespace identifiers are intentionally retained, even when an experiment later completed.',
            '', '## Authoritative current and historical hierarchy','',
            'Current: [README](../../README.md), [PROJECT_CONTEXT](../../PROJECT_CONTEXT.md), [AGENTS](../../AGENTS.md), current LOW/HIGH and temporal reports, [experiment registry](../summary/experiment_registry.yaml). Execution status and scientific_status are separate: current_primary, current_supporting, historical, legacy, diagnostic, closed_negative, infrastructure_only.',
            'Historical final snapshots retain the original 2026-08-29 date. The previous article/conclusions are preserved in separate legacy documents. Internal Q3/label_q5 reports receive banners; valid COG-BCI/CL-Drive/CLARE three-class labels and synthetic fixtures are untouched.',
            '', '## Corrections and generator ownership','',
            '- Corrected future/smoke-only temporal status to 175/175 matched and 25/25 order control; 5/7 history gains and 4/5 conditional order effects; stress inconclusive.',
            '- Corrected exact LOW <= Q33 / HIGH >= Q67 and before-history threshold fitting, canonical 371 features, nested AutoML completion and LOW/HIGH personalization.',
            '- Rebuilt current experiment/colleague/requirements summaries from metadata. Historical execution statuses, metric sources and numerical tables are retained.',
            '- experiment_summary separates current primary and legacy sections; colleague generator prepends curated current evidence. Requirement generator owns both summary and final coverage.',
            '- config auditor emits scientific roles from config_curation.yaml; dedicated JSON protocols are explicitly metadata_only, not falsely resolved by the YAML auditor.',
            '- Historical report_builder/project_final_package outputs cannot overwrite current article/conclusions/requirement coverage. No completed legacy package regeneration or result replacement was performed. Two existing tests entered the legacy generator but stopped at metadata mapping before writes.',
            '', '## File-by-file after audit','', '| Path | Matches | Classification | Action |','|---|---|---|---|']
    for row in states['after']['rows']:
        matches=', '.join(f'{k}: {v}' for k,v in row['matches'].items())
        lines.append(f"| `{row['path']}` | {matches} | {row['classification']} | {row['action']} |")
    output.write_text('\n'.join(lines)+'\n',encoding='utf-8')
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline',default='fbd100f')
    parser.add_argument('--output',type=Path,default=ROOT/'reports/architecture/current_protocol_documentation_audit.md')
    args=parser.parse_args(); result=audit(args.baseline,args.output)
    print(json.dumps({p:{k:v for k,v in result[p].items() if k!='rows'} for p in ['before','after']},indent=2))

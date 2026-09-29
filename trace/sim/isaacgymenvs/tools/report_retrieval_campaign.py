"""Summarize only complete, trace-validated and initial-state-matched batches.

Scene-stratified bootstrap intervals condition on the fitted checkpoints.
Variation across independent fits is reported separately, never inflated by
counting shared scenes or perturbation replicas as independent scene samples.
"""
from pathlib import Path
import argparse
from collections import Counter,defaultdict
import json
import re
import time
import numpy as np
from run_evaluation_campaign import write_json


def interval(values,tiers,seed=20260907,replicates=2000):
    values=np.asarray(values,dtype=float);tiers=np.asarray(tiers)
    rng=np.random.default_rng(seed);total=np.zeros(replicates)
    for tier in np.unique(tiers):
        group=values[tiers==tier]
        total+=group[rng.integers(0,len(group),(replicates,len(group)))].sum(1)
    return (100*np.percentile(total/len(values),[2.5,97.5])).tolist()


def summarize(records):
    grouped=defaultdict(list);fit_scores=defaultdict(list);tier_scores=defaultdict(list)
    for r in records:
        grouped[r['scene']['sha256']].append(r)
        fit_scores[r['case_name']].append(r['success'])
        tier_scores[r['scene']['tier']].append(r['success'])
    means=[np.mean([r['success'] for r in rows]) for rows in grouped.values()]
    tiers=[rows[0]['scene']['tier'] for rows in grouped.values()]
    fit_means=[100*np.mean(v) for v in fit_scores.values()]
    initial=sum(r['success'] and r['terminal_step']==0 for r in records)
    successes=sum(r['success'] for r in records)
    nontrivial=len(records)-initial
    successful_steps=[r['terminal_step'] for r in records if r['success'] and r['terminal_step']>0]
    evidence=[r for r in records if r.get('visible_fraction') is not None]
    return dict(unique_scenes=len(grouped),attempts=len(records),successes=successes,
        success_pct=100*float(np.mean(means)),scene_bootstrap_95ci=interval(means,tiers),
        checkpoints=len(fit_scores),per_checkpoint_success_pct=dict(zip(fit_scores,fit_means)),
        checkpoint_sd_pct=float(np.std(fit_means,ddof=1)) if len(fit_means)>1 else None,
        perturbation_seeds=sorted(set(r['perturbation_seed'] for r in records)),
        reasons=dict(Counter(r['reason'] for r in records)),initial_successes=initial,
        premature_stops=sum(r['reason']=='premature_stop' for r in records),
        timeouts=sum(r['reason']=='timeout' for r in records),
        graspable_timeouts=sum(r['reason']=='timeout' and r['terminal_q']>.9 for r in records),
        initially_graspable=sum(r['initial_q']>.9 for r in records),
        success_excluding_initial_success_pct=100*(successes-initial)/nontrivial if nontrivial else None,
        median_decisions_success_after_action=float(np.median(successful_steps)) if successful_steps else None,
        mean_per_episode_visible_fraction=float(np.mean([r['visible_fraction'] for r in evidence])) if evidence else None,
        episodes_exposed_to_scheduled_blackout=sum(r['blackout_steps_exposed']>0 for r in records),
        tiers={t:dict(attempts=len(v),successes=sum(v),success_pct=100*float(np.mean(v))) for t,v in tier_scores.items()})


def report(campaign):
    root=Path(campaign).resolve();status=json.loads((root/'status.json').read_text())
    families=defaultdict(list);cases=defaultdict(list);accepted_batches=[];stopping_modes={}
    for paired in sorted(root.glob('s*_b*_paired_starts.json')):
        pairing=json.loads(paired.read_text())
        if not pairing['matched']: continue
        tag=paired.name.removesuffix('_paired_starts.json') if hasattr(str,'removesuffix') else paired.name[:-len('_paired_starts.json')]
        temporary=[]
        for name in pairing['differences']:
            directory=root/(tag+'_'+name)
            validation=json.loads((directory/'validation.json').read_text())
            result=json.loads((directory/(name+'.json')).read_text())
            assert validation['valid'] and result['complete']
            stopping_modes[name]=dict(mode=result.get('termination'),diagnostic_only=result.get('diagnostic_only',False),
                                      privileged_stopping=result.get('privileged_stopping',False))
            assert validation['cases'][name]['successes']==sum(r['success'] for r in result['rows'])
            for row in result['rows']:
                temporary.append(dict(row,case_name=name,perturbation_seed=result['seed']))
        accepted_batches.append(tag)
        for row in temporary:
            cases[row['case_name']].append(row)
            family=re.sub(r'_s\d+(?=_|$)','',row['case_name'])
            families[family].append(row)
    result=dict(campaign=str(root),state=status['state'],partial=status['state']!='complete',
        generated_unix=time.time(),accepted_batches=accepted_batches,
        uncertainty='95% percentile bootstrap within scene tiers, resampling unique scenes and retaining all fits/perturbations together; conditional on fitted checkpoints. Checkpoint SD reported separately.',
        groups={k:summarize(v) for k,v in families.items()},
        cases={k:summarize(v) for k,v in cases.items()},stopping_modes=stopping_modes,paired_differences={})
    comparisons=[('student','bc'),('dagger_matched','bc_matched'),('student','teacher'),('student','replay'),
                 ('student_d90','bc_d90'),('student_d100','bc_d100')]
    comparisons += [(name,'bc') for name in sorted(families) if re.fullmatch(r'dagger_r\d+',name)]
    for left,right in comparisons:
        if left not in families or right not in families: continue
        def by_scene(rows):
            out=defaultdict(list)
            for row in rows: out[row['scene']['sha256']].append(row)
            return out
        l,r=by_scene(families[left]),by_scene(families[right]);common=sorted(set(l)&set(r))
        diffs=[np.mean([x['success'] for x in l[h]])-np.mean([x['success'] for x in r[h]]) for h in common]
        tiers=[l[h][0]['scene']['tier'] for h in common]
        result['paired_differences'][left+' minus '+right]=dict(unique_scenes=len(common),
            percentage_points=100*float(np.mean(diffs)),scene_bootstrap_95ci=interval(diffs,tiers))
    write_json(root/'summary.json',result)
    lines=['# Simulation campaign summary','',f"State: **{status['state']}**. "+('Partial results; not paper estimates.' if result['partial'] else 'All scheduled batches completed.'),'',
        '| Method | Unique scenes | Checkpoints | Attempts | Success (%) | Scene bootstrap 95% CI | Fit SD (pp) |',
        '|---|---:|---:|---:|---:|---|---:|']
    for name,row in result['groups'].items():
        ci=row['scene_bootstrap_95ci'];sd=row['checkpoint_sd_pct']
        sd_text=f'{sd:.2f}' if sd is not None else '—'
        lines.append(f"| {name} | {row['unique_scenes']} | {row['checkpoints']} | {row['attempts']} | {row['success_pct']:.2f} | [{ci[0]:.2f}, {ci[1]:.2f}] | {sd_text} |")
    lines+=['',result['uncertainty'],'','Only complete batches with validated traces and matched actual starts are included. Full initial-graspability, failure, tier, visibility, and paired-difference details are in `summary.json`.','']
    if any(row['mode']=='oracle_graspability' for row in stopping_modes.values()):
        lines += ['Student oracle-stopping cases use the privileged regular simulator graspability predictor. '
                  'These are simulator diagnostics, not deployable stopping or physical grasp-and-lift results.','']
    (root/'SUMMARY.md').write_text('\n'.join(lines))
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('campaign');args=parser.parse_args()
    payload=report(args.campaign)
    print(json.dumps(dict(state=payload['state'],matched_batches=len(payload['accepted_batches']),
        groups={k:dict(unique_scenes=v['unique_scenes'],success_pct=v['success_pct']) for k,v in payload['groups'].items()}),indent=2))

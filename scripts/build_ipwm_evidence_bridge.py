"""Summarize all completed policy/architecture cells without sign filtering."""
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'src')]
import numpy as np,pandas as pd
OUT=ROOT/'runs/ipwm_targeted_advantage_20260911'
records=[];inventory={}
for policy,folder,expected in [('legacy','legacy_policy_100',42),('directional','directional_policy_100',24)]:
    files=list((OUT/folder).glob('*_s*.json'));inventory[policy]=dict(completed=len(files),expected=expected)
    for p in files:
        j=json.loads(p.read_text())
        for r in j['rows']:records.append(dict(policy=policy,family=j['family'],seed=j['seed'],domain=j['domain'],**r))
df=pd.DataFrame(records)
metric_names=['object_mse','object_xy_mse','object_velocity_mse','free_mse']
group=['policy','family','seed','domain','method']
agg=df.groupby(group)[metric_names].mean().pow(.5)
agg=agg.join(df.groupby(group)[['robot_carrier_max','lock_violation']].max()).reset_index()
agg.to_csv(OUT/'expanded_all_cells.csv',index=False)
comparisons=[]
for key,part in agg.groupby(group[:-1]):
    by=part.set_index('method')
    for base in ['carrier','full']+(['global','nominal'] if key[1]=='final' else []):
        for metric in metric_names:
            b=float(by.loc[base,metric]);s=float(by.loc['selective',metric])
            comparisons.append(dict(zip(group[:-1],key),baseline=base,metric=metric.replace('mse','rmse'),
                reduction_pct=100*(1-s/b),baseline_value=b,candidate_value=s))
cmp=pd.DataFrame(comparisons);cmp.to_csv(OUT/'expanded_comparisons.csv',index=False)
target=['D3__high_damping','D3__mixed_composition','D3__mixed_unseen'];summaries=[]
for (policy,family,base,metric),p in cmp.groupby(['policy','family','baseline','metric']):
    for label,domains in [('targeted_three',target),('nominal',['D3__nominal']),('all_reported',None)]:
        sub=p if domains is None else p[p.domain.isin(domains)]
        seedmeans=sub.groupby('seed').reduction_pct.mean()
        draws=np.random.default_rng(20260911).choice(seedmeans.values,(10000,len(seedmeans)),replace=True).mean(1)
        summaries.append(dict(policy=policy,family=family,baseline=base,metric=metric,stratum=label,
            mean_reduction_pct=float(seedmeans.mean()),seed_means={str(k):float(v) for k,v in seedmeans.items()},
            positive_seeds=int((seedmeans>0).sum()),positive_cells=int((sub.reduction_pct>0).sum()),cells=len(sub),
            descriptive_seed_ci95=np.quantile(draws,[.025,.975]).tolist()))
# A paired trajectory bootstrap, conditional on the three fixed trained models.
trajectory_ci=[]
for policy in ['legacy','directional']:
    for metric in metric_names[:3]:
        sub=df[(df.policy==policy)&(df.family=='historical')&df.domain.isin(target)&df.method.isin(['carrier','selective'])]
        if len(sub)==0:continue
        tab=sub.groupby(['seed','domain','method','trajectory'])[metric].mean()
        seeds=sorted(sub.seed.unique());rng=np.random.default_rng(20260911);values=[]
        for d in target:
            ids=rng.integers(0,100,(2000,100))
            for s in seeds:
                try:
                    base=tab.loc[s,d,'carrier'].values;candidate=tab.loc[s,d,'selective'].values
                except KeyError:continue
                b=np.sqrt(base[ids].mean(1));c=np.sqrt(candidate[ids].mean(1))
                values.append(100*(1-c/b))
        draws=np.mean(values,axis=0)
        trajectory_ci.append(dict(policy=policy,metric=metric.replace('mse','rmse'),
            conditional_paired_trajectory_ci95=np.quantile(draws,[.025,.975]).tolist(),
            note='Conditional on fixed training seeds and fixed three target physics. Resample whole trajectories, not windows.'))
selection=[]
for p in sorted((OUT/'fresh_selection_120').glob('seed*.json')):
    j=json.loads(p.read_text());r=pd.DataFrame(j['rows'])
    for base in ['nominal_original','nominal_projected','carrier','global']:
        b=r[r.method==base].set_index('episode');c=r[r.method=='selective'].set_index('episode')
        item=dict(seed=j['seed'],baseline=base)
        for metric in ['endpoint','regret']:
            item[metric+'_reduction_pct']=100*(1-c[metric].mean()/b[metric].mean())
            item[metric+'_difference_mm']=1000*(b[metric]-c[metric]).mean()
            rng=np.random.default_rng(20260911);diff=(b[metric]-c[metric]).values
            boot=diff[rng.integers(0,len(diff),(10000,len(diff)))].mean(1)*1000
            item[metric+'_paired_ci95_mm']=np.quantile(boot,[.025,.975]).tolist()
        item['success_gain_pp']=100*(c.success.mean()-b.success.mean());selection.append(item)
result=dict(inventory=inventory,summary=summaries,trajectory_bootstrap=trajectory_ci,selection=selection,
    units='object_rmse combines two metre position channels and two metre/second velocity channels, unnormalized. Reported for historical compatibility; not an SE2 error.',
    restriction='No stable position or velocity superiority inferred from composite RMSE. Three trained seeds; bootstrap is descriptive. Prediction-policy tests are not closed-loop task tests.')
(OUT/'evidence_bridge_summary.json').write_text(json.dumps(result,indent=2))
print(inventory)
for s in summaries:
    if s['stratum']=='targeted_three' and s['baseline']=='carrier':print(s)
if selection:print('SELECTION',json.dumps(selection,indent=2))
# Plot each trained seed so a mean never hides disagreement.
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size':9,'axes.spines.right':False,'axes.spines.top':False})
fig,axes=plt.subplots(2,3,figsize=(11,6),constrained_layout=True)
for i,policy in enumerate(['legacy','directional']):
    for k,(metric,title) in enumerate([('object_rmse','Composite object state'),('object_xy_rmse','Object position'),('object_velocity_rmse','Object velocity')]):
        ax=axes[i,k];sub=cmp[(cmp.policy==policy)&(cmp.family=='historical')&(cmp.baseline=='carrier')&(cmp.metric==metric)&cmp.domain.isin(target)]
        for seed,color in [(27,'#2266aa'),(37,'#cc6622'),(47,'#228866')]:
            sp=sub[sub.seed==seed].set_index('domain');vals=[sp.loc[d,'reduction_pct'] if d in sp.index else np.nan for d in target]
            ax.plot(range(3),vals,'o-',color=color,label=f'Seed {seed}',lw=1.3,ms=4)
        ax.axhline(0,color='black',lw=.8);ax.set_xticks(range(3),['Damping','Composition','Mixed unseen']);ax.set_title(title)
        ax.set_ylabel(('Legacy path' if i==0 else 'Changed path')+'\nRMSE reduction (%)')
axes[0,0].legend(frameon=False)
fig.savefig(OUT/'expanded_policy_tradeoffs.png',dpi=200);fig.savefig(OUT/'expanded_policy_tradeoffs.svg');plt.close(fig)

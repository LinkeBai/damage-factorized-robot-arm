"""Complete-spectrum tables and paired training-seed summaries, with all signs retained."""
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'src')]
import numpy as np
import pandas as pd
OUT=ROOT/'runs/ipwm_targeted_advantage_20260911'
rows=[];inventory=[]
for path in sorted(OUT.glob('*_q*_s*_D*.json')):
    j=json.loads(path.read_text()); inventory.append({k:j[k] for k in ['family','query','seed','domain']})
    for row in j['rows']:rows.append(dict(family=j['family'],query=j['query'],seed=j['seed'],**row))
df=pd.DataFrame(rows)
keys=['family','query','seed','domain','horizon','method']
metrics=['object_mse','object_xy_mse','object_velocity_mse','free_mse']
agg=df.groupby(keys)[metrics].mean().pow(.5).rename(columns=lambda x:x.replace('mse','rmse'))
agg=agg.join(df.groupby(keys)[['robot_carrier_max','lock_violation']].max()).reset_index()
agg.to_csv(OUT/'all_cells.csv',index=False)
comparisons=[]
for (family,query,seed,domain,h),group in agg.groupby(keys[:-1]):
    by=group.set_index('method')
    for baseline in (['nominal','carrier','global','full'] if family=='final' else ['carrier','full']):
        for candidate in (['selective'] if family=='final' else ['selective','routed']):
            for metric in [m.replace('mse','rmse') for m in metrics]:
                b=float(by.loc[baseline,metric]);s=float(by.loc[candidate,metric])
                comparisons.append(dict(family=family,query=query,seed=seed,domain=domain,horizon=h,
                    baseline=baseline,candidate=candidate,metric=metric,baseline_value=b,candidate_value=s,
                    reduction_pct=100*(b-s)/b if b>0 else None))
cmp=pd.DataFrame(comparisons);cmp.to_csv(OUT/'paired_comparisons.csv',index=False)
targeted=['high_damping','mixed_composition','mixed_unseen']
summary=[]
for (fam,base,cand,metric),part in cmp[cmp.horizon==50].groupby(['family','baseline','candidate','metric']):
    for topology in sorted({d.split('__')[0] for d in part.domain}):
        top=part[part.domain.str.startswith(topology+'__')]
        for label,phy in [('all_seven',None),('targeted_three',targeted),('nominal',['nominal'])]:
            sub=top if phy is None else top[top.domain.map(lambda d:d.split('__')[1] in phy)]
            seedmeans=sub.groupby('seed').reduction_pct.mean()
            rng=np.random.default_rng(20260911)
            draws=rng.choice(seedmeans.to_numpy(),(10000,len(seedmeans)),replace=True).mean(1)
            summary.append(dict(family=fam,baseline=base,candidate=cand,metric=metric,topology=topology,
                stratum=label,mean_reduction_pct=float(seedmeans.mean()),
                seed_means={str(k):float(v) for k,v in seedmeans.items()},positive_seeds=int((seedmeans>0).sum()),
                descriptive_seed_bootstrap95=np.quantile(draws,[.025,.975]).tolist(),
                positive_cells=int((sub.reduction_pct>0).sum()),cells=len(sub)))
result=dict(completed_files=len(inventory),expected_files=168,complete=len(inventory)==168,
    primary_horizon=50,summary=summary,
    note='Equal domain/query weighting followed by equal training-seed weighting. CI is descriptive with three trained seeds. Query seeds are not additional training seeds. All seven physics conditions retained.')
(OUT/'summary.json').write_text(json.dumps(result,indent=2))
print('files',len(inventory),'/168')
for row in summary:
    if row['baseline']=='carrier' and row['candidate']=='selective' and row['metric'] in ['object_rmse','object_xy_rmse'] and row['stratum']=='targeted_three':
        print(row)
try:
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,2,figsize=(10,3.5),constrained_layout=True)
    phys=['nominal','weak_motor','high_damping','delay_1','noisy_deadband','mixed_composition','mixed_unseen']
    for ax,family in zip(axes,['historical','final']):
        sub=cmp[(cmp.family==family)&(cmp.domain.str.startswith('D3__'))&(cmp.horizon==50)&(cmp.baseline=='carrier')&(cmp.candidate=='selective')]
        for metric,color,label,offset in [('object_xy_rmse','#2266aa','Object position',-.13),('object_velocity_rmse','#cc6622','Object velocity',.13)]:
            part=sub[sub.metric==metric].groupby(['domain','seed']).reduction_pct.mean()
            vals=[part.loc['D3__'+p].mean() if 'D3__'+p in part.index.get_level_values(0) else np.nan for p in phys]
            ax.barh(np.arange(7)+offset,vals,height=.24,color=color,label=label)
            for i,p in enumerate(phys):
                if 'D3__'+p in part.index.get_level_values(0):
                    ax.plot(part.loc['D3__'+p].values,np.repeat(i+offset,len(part.loc['D3__'+p])),'.',color='black',markersize=3)
        ax.axvline(0,color='black',lw=.8);ax.set_yticks(range(7),[p.replace('_',' ') for p in phys]);ax.invert_yaxis()
        ax.set_xlabel('RMSE reduction vs carrier (%)');ax.set_title('Historical coupled IPWM' if family=='historical' else 'Final deployment-model family')
    axes[0].legend(loc='best',frameon=False)
    fig.savefig(OUT/'physics_spectrum.png',dpi=200)
    fig.savefig(OUT/'physics_spectrum.svg')
    plt.close(fig)
except ImportError as e: print('Plot dependency unavailable:',e)

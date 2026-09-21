"""Render complete measured results as reviewable evidence and LaTeX tables."""
import html,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'runs/ipwm_scale_goal_20260911'
NAMES={'ipwm':'IPWM','carrier':'No residual head','global':'Global residual','nominal_projected':'Projected nominal'}

def table(headers,rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+
        ['| '+' | '.join(map(str,r))+' |' for r in rows])

def main():
    d=json.loads((OUT/'results-summary.json').read_text())
    assert len(d['training'])==18 and len(d['planning'])==36
    plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,2,figsize=(12,4.8),layout='constrained')
    for ax,family,title in zip(axes,['adapted','deployed'],['Adaptation study (6 fit seeds)','Existing models (3 seeds)']):
        for i,method in enumerate(NAMES):
            vals=[r['xy_rmse_m']*1000 for r in d['prediction'] if r['family']==family and r['method']==method and r['horizon']==50]
            ax.scatter(np.full(len(vals),i)+np.linspace(-.1,.1,len(vals)),vals,s=35)
            ax.plot([i-.2,i+.2],[np.mean(vals)]*2,color='black',lw=2)
        ax.set_xticks(range(4),['IPWM','No head','Global','Nominal'],rotation=20)
        ax.set_ylabel('Object xy RMSE (mm), H=50');ax.set_title(title);ax.grid(axis='y',alpha=.2)
    fig.savefig(OUT/'scale-prediction.png',dpi=180);fig.savefig(OUT/'scale-prediction.svg');plt.close(fig)
    rows=[r for r in d['paired_comparisons'] if r['family']=='deployed' and r['seed']==27]
    fig,ax=plt.subplots(figsize=(8,4),layout='constrained')
    for i,r in enumerate(rows):
        mean=r['endpoint_improvement_m']*1000;low=r['paired_problem_ci_low_m']*1000;high=r['paired_problem_ci_high_m']*1000
        ax.errorbar(mean,i,xerr=[[mean-low],[high-mean]],fmt='o',capsize=5)
    ax.axvline(0,color='gray',lw=1);ax.set_yticks(range(len(rows)),[NAMES[r['baseline']] for r in rows])
    ax.set_xlabel('Endpoint improvement of IPWM over baseline (mm)')
    ax.set_title('Deployed checkpoint: 1,200 problems, 6 planner repeats\nPaired problem 95% bootstrap intervals')
    fig.savefig(OUT/'scale-planning-comparisons.png',dpi=180);fig.savefig(OUT/'scale-planning-comparisons.svg');plt.close(fig)
    protocol=(
        'We evaluate contact-neighborhood pushing under five diagnosed joint locks and four physical settings. '
        'The simulated object is a box with two translational degrees of freedom. '
        'The dataset contains 50,000 independently reset trajectories for adaptation, 2,000 for validation, and 6,000 for testing. '
        'Each trajectory contains 50 simulation steps at 5 ms per step. Training, validation, and test reset seeds are disjoint. '
        'We fit each adaptation variant with six seeds from a common pretrained carrier and select checkpoints using validation object-position error. '
        'The existing deployment checkpoints are evaluated separately with their weights unchanged. '
        'Position RMSE averages squared errors over the x and y coordinates before taking the square root; terminal distance is the Euclidean distance to the goal. '
        'Planning uses 1,200 independently generated start-goal problems, 128 candidates per replan, a 50-step prediction horizon, and five replans that each execute ten steps. '
        'All methods share the same problems and candidate commands within each repetition. '
        'The deployed checkpoint receives six candidate-sampling repetitions; intervals resample independent problems after averaging repetitions within each problem.')
    lines=['# IPWM 大规模仿真实验证据','',
        '本报告汇总完整注册实验。仿真物体为固定朝向、具有两个平移自由度的方块。数据规模比较针对仿真样本数量；不代表性能超过 ActivePusher，也不将损伤条件数量视为物体多样性。','',
        '## 实验规模','',table(['项目','ActivePusher 仿真','IPWM 本次实验'],[
            ['候选训练数据池','36,000（4×9,000）','50,000（5×10,000）'],
            ['独立预测测试','4,000（4×1,000）','6,000（5×1,200）'],
            ['每次训练实际消费数据','最多100条','50,000条轨迹，每轮一次'],
            ['训练重复','5次','每变体6次适配拟合，共享预训练基础'],
            ['独立规划问题','每任务100个','每损伤与距离区间120个，共1,200个'],
            ['固定模型规划重复','5次','原部署检查点6次']]),'',
        'ActivePusher 数据来自用户提供原文 Section V。其物体与任务与本实验不同，以上为计数对照。','',
        '## 实验设置英文稿','',protocol,'',
        '## 预测误差','',
        '位置RMSE为x、y两坐标平方误差取平均后开方；规划终点误差为物体到目标的欧氏距离。位置和速度分别以米、米每秒报告。下面列出H50；H10与H25完整结果见附带CSV。新适配模型和已有模型分表，不合并成同一部署模型。','']
    for family in ['adapted','deployed']:
        lines += ['### '+('新适配模型' if family=='adapted' else '已有模型'),'']
        rs=[r for r in d['prediction'] if r['family']==family and r['horizon']==50]
        lines += [table(['种子','方法','对象xy RMSE (mm)','对象速度RMSE (m/s)','自由关节RMSE (rad)'],
            [[r['seed'],NAMES[r['method']],f"{r['xy_rmse_m']*1000:.3f}",f"{r['velocity_rmse_mps']:.4f}",f"{r['free_q_rmse_rad']:.4f}"] for r in rs]),'']
    lines += ['## 闭环规划','',
        '每次执行都使用该方法自己的新状态重新选动作。终点误差和30mm成功率按最终状态计算。下表的重复已在同一问题内平均，不增加独立问题数。','',
        table(['模型组','种子','方法','每问题重复','终点误差 (mm)','成功率 (%)'],
            [[r['family'],r['seed'],NAMES[r['method']],r['planner_repeats_per_problem'],f"{r['mean_endpoint_m']*1000:.3f}",f"{r['success_rate']*100:.2f}"] for r in d['planning']]),'',
        '## IPWM相对各基线的配对差值','',
        '正数表示IPWM终点误差较低，负数表示基线更好。区间为给定拟合模型下的问题抽样区间，不是跨训练种子的区间。所有种子和比较均保留。','',
        table(['模型组','种子','基线','终点改善 (mm)','95%区间 (mm)','成功率差 (百分点)'],
            [[r['family'],r['seed'],NAMES[r['baseline']],f"{r['endpoint_improvement_m']*1000:.3f}",
              f"[{r['paired_problem_ci_low_m']*1000:.3f}, {r['paired_problem_ci_high_m']*1000:.3f}]",f"{r['success_difference_pp']:.2f}"] for r in d['paired_comparisons']]),'',
        '## 训练与复现','',
        '三个新拟合变体共享预训练基础与数据、优化预算。去除残差头的对照保留对象模块适配；新global使用全状态归一化监督，其机器人输出收到直接梯度。原有global的机器人残差输出为零，按已有对象残差变体另行评估。','',
        table(['方法','种子','选中轮次','训练轨迹','可训练参数'],[[NAMES[r['method']],r['seed'],r['best_epoch'],r['actual_training_unique_trajectories'],r['trainable_parameters']] for r in d['training']]),'',
        '协议、原始轨迹、逐问题结果、完整CSV及哈希清单位于 runs/ipwm_scale_goal_20260911。没有新增真机实验；仿真使用精确状态反馈。','']
    md='\n'.join(lines)
    (ROOT/'paper/ipwm-scale-evidence-20260911.md').write_text(md,encoding='utf-8')
    body=[];in_table=False
    for line in md.splitlines():
        if line.startswith('|'):
            if not in_table:body.append('<table>');in_table=True
            cells=[c.strip() for c in line.strip('|').split('|')]
            if all(c=='---' for c in cells):continue
            body.append('<tr>'+''.join('<td>'+html.escape(c)+'</td>' for c in cells)+'</tr>')
        else:
            if in_table:body.append('</table>');in_table=False
            if line.startswith('#'):
                level=len(line)-len(line.lstrip('#'));body.append(f'<h{level}>{html.escape(line[level:].strip())}</h{level}>')
            elif line:body.append('<p>'+html.escape(line)+'</p>')
    if in_table:body.append('</table>')
    body.insert(3,'<img src="../runs/ipwm_scale_goal_20260911/scale-prediction.png"><img src="../runs/ipwm_scale_goal_20260911/scale-planning-comparisons.png">')
    (ROOT/'paper/ipwm-scale-evidence-20260911.html').write_text('<!doctype html><meta charset="utf-8"><title>IPWM 大规模仿真实验证据</title><style>body{max-width:1180px;margin:40px auto;padding:0 24px;font:16px/1.65 system-ui;color:#172337}table{border-collapse:collapse;width:100%;font-size:14px}td{border-bottom:1px solid #d7dee7;padding:8px}tr:first-child{font-weight:bold;background:#edf2f8}img{max-width:100%;display:block;margin:24px auto}h2{margin-top:40px}</style>'+''.join(body),encoding='utf-8')
    tex=['\\section{Large-scale simulation evaluation}',protocol,'',
         '\\begin{table}[t]','\\centering','\\caption{Closed-loop planning for the deployed checkpoint, averaged over six candidate-sampling repetitions on 1,200 independent problems.}',
         '\\begin{tabular}{lrr}','\\hline','Method & Endpoint (mm) & Success (\\%) \\\\','\\hline']
    for r in d['planning']:
        if r['family']=='deployed' and r['seed']==27:
            tex.append(f"{NAMES[r['method']]} & {r['mean_endpoint_m']*1000:.3f} & {r['success_rate']*100:.2f} \\\\")
    tex+=['\\hline','\\end{tabular}','\\end{table}']
    (ROOT/'paper/ipwm-scale-experiments-20260911.tex').write_text('\n'.join(tex)+'\n',encoding='utf-8')
    print('Full evidence report, plots and LaTeX experiment text written')

if __name__=='__main__':main()

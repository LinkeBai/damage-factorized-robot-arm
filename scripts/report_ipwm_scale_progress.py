"""Current artifact-backed progress; incomplete work stays explicitly incomplete."""
import json,html
from datetime import datetime
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'runs/ipwm_scale_goal_20260911'

def main():
    audit=json.loads((OUT/'dataset-audit.json').read_text())
    queue=json.loads((OUT/'queue.json').read_text())
    state=json.loads((OUT/'loop-state.json').read_text())
    trains=[json.loads(p.read_text()) for p in (OUT/'training').glob('*/seed*/complete.json')]
    predictions=list((OUT/'prediction').glob('*/seed*/complete.json'))
    planning=list((OUT/'planning').glob('*/seed*/repeat*/*/D*-B*/complete.json'))
    active=[k for k,v in state['jobs'].items() if v['status']=='running']
    counts={k:sum(v.values()) for k,v in audit['counts'].items()}
    rows=[['数据池独立轨迹',36000,counts['pool']],['独立预测测试轨迹',4000,counts['test']],
          ['验证轨迹（单列，不计入测试）','—',counts['validation']],
          ['完成的新模型拟合次数','训练重复5次；不能与模型总数混同',f'{len(trains)} / 18'],
          ['完成的预测评估批次','—',f'{len(predictions)} / 9'],
          ['完成的规划对照批次','—',f'{len(planning)} / 560']]
    text=['# IPWM 大规模仿真实验进度','',
          '当前为过程记录，尚不是完成的论文结果。计数来自已经落盘的实验文件。','',
          '| 项目 | ActivePusher 仿真参照 | 当前已完成 |','|---|---:|---:|']
    text += ['| '+' | '.join(map(str,r))+' |' for r in rows]
    text += ['', '数据核验：58,000 个唯一 reset ID，58,000 个唯一初始状态与动作序列指纹；训练池、验证、测试不重叠。',
             '', '规划方案：1,200 个独立起点与目标组合；5 种锁定 × 2 个距离区间 × 每格120个问题。方法和重复共享这些问题，不能把所有运行次数当作独立问题数。',
             '', '新模型：6 个适配训练种子，共享同一个已预训练的机器人预测基础；每种模型各6次拟合。它们不是6次从零开始的完整预训练，也没有新增真机验证。',
             '', '部署模型：保留旧权重；seed27另做6次规划重复，seed7和17做敏感性对照。',
             '', 'GPU采集验证：20种锁定与物理条件全部通过CPU对照；轨迹数据采用MuJoCo Warp，训练与模型预测采用GPU。',
             '', 'ActivePusher原文来源：用户提供的PDF，Section V；4物体×9,000数据池、4×1,000测试、每次实际训练最多100条、训练重复5次；每规划任务100问题、规划重复5次。不同物体与不同损伤条件不是同一多样性维度。',
             '', '当前循环记录中的运行任务：'+', '.join(active),
             '', '停止条件仍未全部满足：全部训练、基线、规划、置信区间、复现清单和最终报告完成后，才做完成核验。']
    dest=ROOT/'paper/ipwm-scale-progress-20260911.md'
    dest.write_text('\n'.join(text)+'\n',encoding='utf-8')
    (OUT/'progress-summary.json').write_text(json.dumps(dict(counts=counts,completed_training=len(trains),
        completed_prediction=len(predictions),completed_planning=len(planning),registered_jobs=len(queue),
        loop_record_running=active,goal_complete=False),indent=2))
    stages=[('数据采集与逐条核验',58000,58000),('模型拟合',len(trains),18),
            ('预测评估',len(predictions),9),('规划对照批次',len(planning),560)]
    cards=[]
    for name,n,total in stages:
        cards.append(f'<section><h2>{name}</h2><strong>{n:,} / {total:,}</strong><progress value="{n}" max="{total}"></progress></section>')
    running=[]
    for name in active:
        detail=name
        if name.startswith('train-'):
            _,method,seed=name.split('-')
            path=OUT/'training'/method/f'seed{seed[1:]}'/'progress.json'
            if path.exists():
                progress=json.loads(path.read_text())
                detail+=f"，最近完成第 {progress['history'][-1]['epoch']} / 10 轮"
        running.append(html.escape(detail))
    bundle=OUT/'planning-bundle-progress.json'
    if bundle.exists():
        record=json.loads(bundle.read_text())
        if record.get('current'):running.append(html.escape(record['current']))
    page='''<!doctype html><meta charset="utf-8"><meta http-equiv="refresh" content="30"><title>IPWM 实验进度</title>
    <style>body{max-width:1000px;margin:40px auto;padding:0 24px;font:17px/1.7 system-ui;color:#193047;background:#f5f7fa}.grid{display:grid;grid-template-columns:1fr 1fr;gap:18px}section{background:white;border:1px solid #dae2ea;border-radius:10px;padding:20px}h2{font-size:18px;margin:0 0 12px}strong{font-size:28px}progress{display:block;width:100%;height:20px;margin-top:16px}small{color:#596c80}</style>'''
    page+='<h1>IPWM 大规模仿真实验</h1><p>目标：超过参考论文的可比仿真数据规模，并完成公平对照、统计与论文证据。当前尚未完成全部验收。</p>'
    page+='<div class="grid">'+''.join(cards)+'</div>'
    page+='<h2>当前运行</h2><p>'+'<br>'.join(running)+'</p>'
    page+='<p>数据池 50,000 条；验证 2,000 条；独立测试 6,000 条。规划使用 1,200 个独立问题，方法和重复次数单独统计。</p>'
    page+='<p>训练结果与原部署权重分开记录；所有种子、基线及负面结果均保留。数据量达标不会替代其余验收。</p>'
    page+='<small>页面数据更新于 '+datetime.now().strftime('%Y-%m-%d %H:%M:%S')+'；每30秒重新载入已保存的进度。</small>'
    (ROOT/'paper/ipwm-scale-progress-20260911.html').write_text(page,encoding='utf-8')
    print(dest)

if __name__=='__main__':main()

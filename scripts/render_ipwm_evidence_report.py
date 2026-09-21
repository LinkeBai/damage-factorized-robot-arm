"""Render the human-readable evidence memo as a local HTML page."""
from pathlib import Path
import re
from markdown_it import MarkdownIt
ROOT=Path(__file__).resolve().parents[1]
source=ROOT/'paper/ipwm-experiment-evidence-20260911.md'
content=MarkdownIt('commonmark').enable('table').render(source.read_text(encoding='utf-8'))
content=content.replace('<table>','<div class="table"><table>').replace('</table>','</table></div>')
content=re.sub(r'<img src="([^"]+)"([^>]*)>',r'<a href="\1" target="_blank"><img src="\1"\2></a>',content)
css='''body{margin:0;background:#f4f6f8;color:#172333;font:17px/1.8 "Microsoft YaHei","Segoe UI",sans-serif}main{max-width:1120px;margin:36px auto;background:white;padding:44px 58px;border-radius:12px}h1{font-size:30px;line-height:1.4}h2{font-size:24px;margin-top:44px;border-top:1px solid #dde4ea;padding-top:24px}h3{font-size:20px}p{margin:16px 0}.table{overflow-x:auto}table{width:100%;border-collapse:collapse;font-size:15px;margin:20px 0}th,td{border-bottom:1px solid #dce4ec;text-align:left;vertical-align:top;padding:12px}th{background:#edf3f8;color:#183f67}strong{color:#164b78}blockquote{margin:20px 0;background:#f0f5fa;border-left:4px solid #356c9f;padding:12px 24px}img{max-width:100%;height:auto}code{font-size:13px;overflow-wrap:anywhere;background:#f0f2f5;padding:2px 4px}a{color:#175d9d}li{margin:8px 0}footer{color:#667;font-size:14px;margin-top:40px}@media(max-width:800px){main{margin:0;padding:22px}body{font-size:16px}h1{font-size:25px}}'''
html='<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>IPWM 实验证据与 ActivePusher 对照</title><style>'+css+'</style><main>'+content+'<footer>本地可审阅版本 · 2026-09-11 · 所有结果须按模型版本、指标与协议引用。</footer></main></html>'
(ROOT/'paper/ipwm-experiment-evidence-20260911.html').write_text(html,encoding='utf-8')
print('Rendered evidence report')

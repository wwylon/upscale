"""Copy plan/design HTML into ui/ and wire simulated run() to the FastAPI backend."""

from __future__ import annotations

import re
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DESIGN = ROOT / "plan" / "design"
UI = ROOT / "ui"

BRIDGE = """
<script src="/ui/backend-bridge.js"></script>
"""


def _copy_tree(src: Path, dst: Path) -> None:
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)


def _wire_advanced(text: str) -> str:
    # Replace simulated run / runBatch / startCmp definitions
    text = re.sub(
        r"runBatch\(\)\{clearInterval\(this\.timer\);.*?\},120\);\}",
        "runBatch(){ if(window.UpscaleBackend) window.UpscaleBackend.runBatch(this); }",
        text,
        count=1,
        flags=re.S,
    )
    text = re.sub(
        r"run=\(force\)=>\{ const s=this\.state;.*?\},120\); \};",
        "run=(force)=>{ const s=this.state; if(s.file&&s.status!=='running'&&force!==true&&this._worst==='fail'){this.setState({blocked:true});return;} if(s.files.length>1&&s.status!=='running'){this.runBatch();return;} if(!s.file||s.status==='running') return; if(window.UpscaleBackend) window.UpscaleBackend.runJob(this,{force:!!force}); };",
        text,
        count=1,
        flags=re.S,
    )
    text = re.sub(
        r"startCmp=\(\)=>\{const s=this\.state;if\(!s\.file\|\|!s\.cmpSel\.length\)return;clearInterval\(this\.ct\);.*?\},700\);\};",
        "startCmp=()=>{ if(window.UpscaleBackend) window.UpscaleBackend.compareModels(this); };",
        text,
        count=1,
        flags=re.S,
    )
    text = text.replace(
        "cancelRun:()=>{clearInterval(this.timer);this.setState(st=>({status:'idle',paused:false,pct:0,files:st.files.map(f=>({...f,st:'queued',pct:0}))}));}",
        "cancelRun:()=>{if(window.UpscaleBackend)window.UpscaleBackend.cancel(this);clearInterval(this.timer);this.setState(st=>({status:'idle',paused:false,pct:0,files:st.files.map(f=>({...f,st:'queued',pct:0}))}));}",
    )
    text = text.replace(
        "togglePause:set({paused:!s.paused})",
        "togglePause:()=>{const n=!s.paused;this.setState({paused:n});if(window.UpscaleBackend){if(n)window.UpscaleBackend.pause(this);else window.UpscaleBackend.resume(this);}}",
    )

    # Live model status from /api/models (via bridge), not the design prototype list
    text = re.sub(
        r"DOWNLOADED = \[[^\]]*\];",
        "DOWNLOADED = [];",
        text,
        count=1,
    )
    text = re.sub(
        r"ERR=\{[^}]*\};",
        "ERR={};",
        text,
        count=1,
    )
    text = re.sub(
        r"mstat\(n\)\{[^}]+\}",
        "mstat(n){if(n==='__custom')return'ok';"
        "const a=this.state.added.find(x=>x[0]===n);if(a)return a[3]==='file'?'ok':'dl';"
        "if(window.UpscaleUI)return window.UpscaleUI.modelStatus(n);"
        "return this.DOWNLOADED.includes(n)?'ok':'dl';}",
        text,
        count=1,
    )

    # Download now → real ensure_model API
    text = text.replace(
        "fixModel:()=>{if(s.fixing)return;const n=s.model;this.setState({fixing:n});setTimeout(()=>{delete this.ERR[n];if(!this.DOWNLOADED.includes(n))this.DOWNLOADED.push(n);this.setState({fixing:null});},1500);},",
        "fixModel:()=>{if(s.fixing)return;const n=s.model;this.setState({fixing:n});"
        "(window.UpscaleBackend?window.UpscaleBackend.downloadModel(n):Promise.resolve())"
        ".then(()=>{delete this.ERR[n];this.setState({fixing:null});})"
        ".catch(e=>{this.ERR[n]=String(e.message||e);this.setState({fixing:null});});},",
    )
    text = text.replace(
        "const mErr=this.ERR[s.model], down=!custom&&!(m[3]==='file')&&!(this.DOWNLOADED.includes(s.model));",
        "const mErr=this.ERR[s.model], down=!custom&&!(m[3]==='file')&&(this.mstat(s.model)==='dl');",
    )
    text = text.replace(
        "modelState:custom?'Local file':m[3]?(m[3]==='url'?'Downloads on first use':'Local file'):this.DOWNLOADED.includes(s.model)?'Downloaded':'Downloads on first use',",
        "modelState:custom?'Local file':m[3]?(m[3]==='url'?'Downloads on first use':'Local file'):this.mstat(s.model)==='ok'?'Downloaded':'Downloads on first use',",
    )

    # Before/after: after layer uses resultUrl when the job finishes
    text = text.replace(
        "const imgBase='width:100%;height:100%;background:url('+s.url+') center/contain no-repeat;user-select:none;pointer-events:none;';\n"
        "    const z=s.zoom, cv=s.test&&done&&s.w, s2=Math.min(sw,sh)/Math.max(ew,eh)*0.9, bgv=cv?'url('+s.url+') '+(sw/2-s.cx*s.w*s2)+'px '+(sh/2-s.cy*s.h*s2)+'px/'+s.w*s2+'px '+s.h*s2+'px no-repeat':'';",
        "const afterSrc=(done&&s.resultUrl)?s.resultUrl:s.url;\n"
        "    const imgBase='width:100%;height:100%;background:url('+s.url+') center/contain no-repeat;user-select:none;pointer-events:none;';\n"
        "    const imgAfter='width:100%;height:100%;background:url('+afterSrc+') center/contain no-repeat;user-select:none;pointer-events:none;';\n"
        "    const z=s.zoom, cv=s.test&&done&&s.w, s2=Math.min(sw,sh)/Math.max(ew,eh)*0.9,"
        " bgv=cv?'url('+s.url+') '+(sw/2-s.cx*s.w*s2)+'px '+(sh/2-s.cy*s.h*s2)+'px/'+s.w*s2+'px '+s.h*s2+'px no-repeat':'',"
        " bgva=cv?'url('+afterSrc+') '+(sw/2-s.cx*s.w*s2)+'px '+(sh/2-s.cy*s.h*s2)+'px/'+s.w*s2+'px '+s.h*s2+'px no-repeat':'';",
    )
    text = text.replace(
        "afterStyle:(cv?imgBase.replace(/background:[^;]+;/,'background:'+bgv+';'):imgBase)+'transform:translate('+s.px+'px,'+s.py+'px) scale('+z+');image-rendering:'+(!done&&z>=3?'pixelated':'auto'),\n"
        "      beforeStyle:(cv?imgBase.replace(/background:[^;]+;/,'background:'+bgv+';'):imgBase)+'transform:translate('+s.px+'px,'+s.py+'px) scale('+z+');image-rendering:pixelated',",
        "afterStyle:(cv?imgAfter.replace(/background:[^;]+;/,'background:'+bgva+';'):imgAfter)+'transform:translate('+s.px+'px,'+s.py+'px) scale('+z+');image-rendering:auto',\n"
        "      beforeStyle:(cv?imgBase.replace(/background:[^;]+;/,'background:'+bgv+';'):imgBase)+'transform:translate('+s.px+'px,'+s.py+'px) scale('+z+');image-rendering:pixelated',",
    )

    # Compare tiles: show each model's result when ready
    text = text.replace(
        "const inner=pix=>'position:absolute;left:50%;top:50%;width:'+(s.w/M*100*cz)+'%;aspect-ratio:'+s.w+'/'+s.h+';transform:translate('+(-s.cx*100+s.cpx)+'%,'+(-s.cy*100+s.cpy)+'%);background:url('+s.url+') 0 0/100% 100% no-repeat;image-rendering:'+(pix?'pixelated':'auto');",
        "const inner=(pix,u)=>'position:absolute;left:50%;top:50%;width:'+(s.w/M*100*cz)+'%;aspect-ratio:'+s.w+'/'+s.h+';transform:translate('+(-s.cx*100+s.cpx)+'%,'+(-s.cy*100+s.cpy)+'%);background:url('+(u||s.url)+') 0 0/100% 100% no-repeat;image-rendering:'+(pix?'pixelated':'auto');",
    )
    text = text.replace(
        "return [{box:'border:2px solid var(--color-text);display:flex;flex-direction:column',outer,...hd,inner:inner(true),pending:false,pendingText:'',name:'Original',meta:this.fmtDims(ew,eh)+' px, nearest-neighbour',canUse:false,use:null,useLabel:''}].concat(s.cmp.map((c,i)=>({box:'border:2px solid '+(s.model===c.name?'var(--color-accent)':'var(--color-text)')+';display:flex;flex-direction:column',outer,...hd,inner:inner(false),pending:!c.done,pendingText:i===first?'Processing…':'Queued',name:c.name,meta:c.done?this.fmtDims(ow,oh)+' px · '+Math.round(c.secs)+' s'+(dn.length>1&&c.secs===fast?' · fastest':''):'~'+Math.round(c.secs)+' s',canUse:c.done,use:set({model:c.name}),useLabel:s.model===c.name?'Selected':'Use this model'})));})(),",
        "return [{box:'border:2px solid var(--color-text);display:flex;flex-direction:column',outer,...hd,inner:inner(true),pending:false,pendingText:'',name:'Original',meta:this.fmtDims(ew,eh)+' px, nearest-neighbour',canUse:false,use:null,useLabel:''}].concat(s.cmp.map((c,i)=>({box:'border:2px solid '+(s.model===c.name?'var(--color-accent)':'var(--color-text)')+';display:flex;flex-direction:column',outer,...hd,inner:inner(false,c.url),pending:!c.done,pendingText:i===first?'Processing…':'Queued',name:c.name,meta:c.done?this.fmtDims(ow,oh)+' px · '+Math.round(c.secs)+' s'+(dn.length>1&&c.secs===fast?' · fastest':''):'~'+Math.round(c.secs)+' s',canUse:c.done,use:set({model:c.name}),useLabel:s.model===c.name?'Selected':'Use this model'})));})(),",
    )

    # Download button uses result
    text = text.replace(
        "download:()=>{(batch?s.files:[{url:s.url,name:s.name}]).forEach((it,i)=>setTimeout(()=>{const a=document.createElement('a');a.href=it.url;a.download=it.name.replace(/\\.[^.]+$/,'')+'_'+(+this.kFor({w:it.w||s.w}).toFixed(2))+'x.'+ext;a.click();},i*300));},",
        "download:()=>{(batch?s.files:[{url:s.resultUrl||s.url,name:s.name,resultUrl:s.resultUrl}]).forEach((it,i)=>setTimeout(()=>{const a=document.createElement('a');a.href=it.resultUrl||it.url;a.download=it.name.replace(/\\.[^.]+$/,'')+'_'+(+this.kFor({w:it.w||s.w}).toFixed(2))+'x.'+ext;a.click();},i*300));},",
    )
    return text


def _wire_simple(text: str) -> str:
    text = re.sub(
        r"run=\(\)=>\{ const s=this\.state;.*?\},120\); \};",
        "run=()=>{ if(window.UpscaleBackend){ window.UpscaleBackend.runSimple(this); return; } };",
        text,
        count=1,
        flags=re.S,
    )
    text = text.replace(
        "const lay='position:absolute;inset:0;background:url('+s.url+') center/contain no-repeat;pointer-events:none;transform:scale('+(done?3:1)+');';",
        "const afterSrc=(done&&s.resultUrl)?s.resultUrl:s.url;"
        "const layB='position:absolute;inset:0;background:url('+s.url+') center/contain no-repeat;pointer-events:none;transform:scale(1);';"
        "const layA='position:absolute;inset:0;background:url('+afterSrc+') center/contain no-repeat;pointer-events:none;transform:scale(1);';",
    )
    text = text.replace(
        "afterStyle:lay+'image-rendering:auto', beforeStyle:lay+'image-rendering:pixelated',",
        "afterStyle:layA+'image-rendering:auto', beforeStyle:layB+'image-rendering:pixelated',",
    )
    text = text.replace(
        "hint:done?'Drag to compare. Zoomed 3× to show the detail.':'',",
        "hint:done?'Drag to compare.':'',",
    )
    text = text.replace(
        "download:()=>{const a=document.createElement('a');a.href=s.url;a.download=s.name.replace(/\\.[^.]+$/,'')+'_'+s.scale+'x.'+(big?'tif':'png');a.click();},",
        "download:()=>{const a=document.createElement('a');a.href=s.resultUrl||s.url;a.download=s.name.replace(/\\.[^.]+$/,'')+'_'+s.scale+'x.'+(big?'tif':'png');a.click();},",
    )
    text = text.replace(
        "clear:()=>{clearInterval(this.t);this.setState({url:null,w:0,h:0,status:'idle',pct:0});},",
        "clear:()=>{clearInterval(this.t);this.setState({url:null,resultUrl:null,w:0,h:0,status:'idle',pct:0});},",
    )
    return text


def _wire_library(text: str) -> str:
    # Prefer API model list when bridge has hydrated
    text = text.replace(
        "const list = s.models.filter(m => s.filter === 'all' || m[2] === s.filter || (s.filter === 'External' && m[5] === 'External'));",
        "const api=window.__UPSCALE_MODELS__;"
        "const base=Array.isArray(api)&&api.length?api.map(m=>[m.name,m.description||'',(m.tags&&m.tags[0])||'General',m.scale||4,"
        "m.size_bytes?Math.max(1,Math.round(m.size_bytes/1e6)):0,m.source==='built-in'?'Built-in':'External',"
        "m.status==='installed'?'ok':m.status==='failed'?'err':'dl']):s.models;"
        "const list = base.filter(m => s.filter === 'all' || m[2] === s.filter || (s.filter === 'External' && m[5] === 'External'));",
    )
    text = text.replace(
        "fixAll: () => this.setState(st => ({ models: st.models.map(m => m[6] === 'ok' ? m : [...m.slice(0, 6), 'ok']), msg: 'All models are ready.' })),",
        "fixAll: async () => { const names=(window.__UPSCALE_MODELS__||[]).filter(m=>m.status!=='installed').map(m=>m.name);"
        "this.setState({msg:'Downloading…'});"
        "try{ for(const n of names){ if(window.UpscaleBackend) await window.UpscaleBackend.downloadModel(n); }"
        "this.setState({msg:'All models are ready.'}); }catch(e){ this.setState({msg:String(e.message||e)}); } },",
    )
    text = re.sub(
        r"act: \(\) => m\[5\] === 'External' && m\[6\] === 'ok' \? this\.setState\(st => \(\{ models: st\.models\.filter\(x => x !== m\) \}\)\) : this\.setState\(\{ msg: m\[0\] \+ ': ' \+ \(m\[6\] === 'ok' \? 'hash matches' : 'download queued'\) \}\) \}\)\),",
        "act: async () => { if(m[6]==='ok'){ this.setState({msg:m[0]+': installed'}); return; }"
        "try{ this.setState({msg:m[0]+': downloading…'}); if(window.UpscaleBackend) await window.UpscaleBackend.downloadModel(m[0]);"
        "this.setState({msg:m[0]+': ready'}); }catch(e){ this.setState({msg:m[0]+': '+String(e.message||e)}); } }})),",
        text,
        count=1,
    )
    return text


def _fix_html(text: str, *, simple: bool = False, library: bool = False) -> str:
    text = text.replace("./support.js", "/ui/support.js")
    text = text.replace('href="_ds/', 'href="/ui/_ds/')
    text = text.replace('src="_ds/', 'src="/ui/_ds/')
    text = text.replace("Upscaler.dc.html", "index.html")
    text = text.replace("Upscaler Simple.dc.html", "simple.html")
    text = text.replace("Model Library.dc.html", "library.html")
    text = text.replace("CLI Reference.dc.html", "cli.html")
    text = text.replace("Desktop App.dc.html", "index.html")
    text = text.replace("Overview.dc.html", "index.html")

    if "</body>" in text and "backend-bridge.js" not in text:
        text = text.replace("</body>", BRIDGE + "\n</body>")

    if simple:
        text = _wire_simple(text)
    elif library:
        text = _wire_library(text)
    else:
        text = _wire_advanced(text)

    return text


def main() -> None:
    UI.mkdir(parents=True, exist_ok=True)

    # Design system + runtime
    shutil.copy2(DESIGN / "support.js", UI / "support.js")
    ds_src = DESIGN / "_ds"
    if ds_src.is_dir():
        _copy_tree(ds_src, UI / "_ds")

    mapping = [
        ("Upscaler.dc.html", "index.html", False, False),
        ("Upscaler Simple.dc.html", "simple.html", True, False),
        ("Model Library.dc.html", "library.html", False, True),
        ("CLI Reference.dc.html", "cli.html", False, True),
    ]
    for src_name, dst_name, simple, library in mapping:
        src = DESIGN / src_name
        raw = src.read_text(encoding="utf-8")
        out = _fix_html(raw, simple=simple, library=library)
        (UI / dst_name).write_text(out, encoding="utf-8")
        print(f"wrote ui/{dst_name}")

    print("Design UI synced. Ensure ui/backend-bridge.js exists.")


if __name__ == "__main__":
    main()

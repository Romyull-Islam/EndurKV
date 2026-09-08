#!/usr/bin/env python3
"""
OnePlus 15 LIVE resource/energy dashboard (the Android 'Xcode Instruments' live view).

Polls the phone over adb at ~4 Hz and serves a browser dashboard with time-aligned
tracks: battery power (W), CPU/GPU/NPU/DDR/skin temperature, CPU utilization, and
per-cluster CPU frequency. Headless-friendly (no X needed) — open the printed URL.

  python3 live_dashboard.py                 # serve on http://127.0.0.1:8717
  python3 live_dashboard.py --port 9000 --hz 4 --csv run1.csv

Env: ADB="/path/to/adb -s <serial>" overrides the adb command.

Power note: a phone reads true draw only while DISCHARGING. Unplug USB (Wi-Fi adb) or
use ../energy_measure.sh to disable charging; the dashboard shows battery status so you
know when the wattage is real.
"""
import argparse, json, os, shlex, subprocess, sys, threading, time
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ADB = os.environ.get("ADB", "/home/mislam22/tools/platform-tools/adb -s 3C15B8003ZA00000").split()
DEV_SAMPLER = "/data/local/tmp/op15_sampler.sh"
DEV_STOP = "/data/local/tmp/op15_sampler.stop"

# ----------------------------------------------------------------------------- phone setup
def adb(*args, **kw):
    return subprocess.run(ADB + list(args), capture_output=True, text=True, **kw)

def resolve_zones():
    """Map thermal-zone types -> /sys temp file lists, by family."""
    out = adb("shell", "su -c "
              "'for z in /sys/class/thermal/thermal_zone*; do "
              "echo $(basename $z) $(cat $z/type 2>/dev/null); done'").stdout
    fam = {"cpu": [], "npu": [], "gpu": [], "ddr": [], "skin": []}
    for line in out.splitlines():
        p = line.split()
        if len(p) < 2:
            continue
        zone, typ = p[0], p[1]
        f = f"/sys/class/thermal/{zone}/temp"
        if "trip" in typ or "limit" in typ:   continue   # constant trip points, not sensors
        if typ.startswith("cpu-"):            fam["cpu"].append(f)
        elif typ.startswith(("nsphvx", "nsphmx")): fam["npu"].append(f)
        elif typ.startswith("gpuss"):         fam["gpu"].append(f)
        elif typ == "ddr":                    fam["ddr"].append(f)
        elif typ.startswith("sys-therm"):     fam["skin"].append(f)
    return fam

def build_sampler(fam, hz):
    period = max(0.05, 1.0 / hz)
    j = lambda fs: " ".join(fs) if fs else "/dev/null"
    freq = " ".join(f"/sys/devices/system/cpu/cpu{c}/cpufreq/scaling_cur_freq" for c in (0, 4, 7))
    return f"""#!/system/bin/sh
SF={DEV_STOP}; rm -f "$SF"; B=/sys/class/power_supply/battery
while [ ! -f "$SF" ]; do
  TS=$(date +%s%3N)
  PW=$(cat $B/power_now 2>/dev/null); CU=$(cat $B/current_now 2>/dev/null)
  VO=$(cat $B/voltage_now 2>/dev/null); ST=$(cat $B/status 2>/dev/null)
  CPU=$(cat {j(fam['cpu'])} 2>/dev/null | tr '\\n' ',')
  NPU=$(cat {j(fam['npu'])} 2>/dev/null | tr '\\n' ',')
  GPU=$(cat {j(fam['gpu'])} 2>/dev/null | tr '\\n' ',')
  DDR=$(cat {j(fam['ddr'])} 2>/dev/null | tr '\\n' ',')
  SK=$(cat {j(fam['skin'])} 2>/dev/null | tr '\\n' ',')
  FR=$(cat {freq} 2>/dev/null | tr '\\n' ',')
  CS=$(grep '^cpu ' /proc/stat)
  echo "$TS|$PW|$CU|$VO|$ST|$CPU|$NPU|$GPU|$DDR|$SK|$FR|$CS"
  sleep {period:.3f}
done
"""

# ----------------------------------------------------------------------------- sampling state
BUF = deque(maxlen=8192)          # parsed samples
LOCK = threading.Lock()
T0 = [None]
PREV_STAT = [None]

def _maxc(s):
    vals = [int(x) / 1000.0 for x in s.split(",") if x.strip().lstrip("-").isdigit()]
    return max(vals) if vals else None

def power_w(pw, cu, vo):
    v = (vo or 0) / 1e6
    if pw and pw > 0:
        return pw / 1e6
    a = abs(cu or 0)
    amps = a / 1e6 if a > 100000 else a / 1e3      # µA vs mA heuristic
    return amps * v

def parse(line):
    try:
        ts, pw, cu, vo, st, cpu, npu, gpu, ddr, sk, fr, cs = line.split("|")
    except ValueError:
        return None
    ts = int(ts)
    if T0[0] is None:
        T0[0] = ts
    geti = lambda x: int(x) if x.strip().lstrip("-").isdigit() else 0
    # CPU utilization from /proc/stat aggregate delta
    nums = [int(x) for x in cs.split()[1:] if x.isdigit()]
    util = None
    if nums:
        idle = nums[3] + (nums[4] if len(nums) > 4 else 0)
        total = sum(nums)
        if PREV_STAT[0]:
            pi, pt = PREV_STAT[0]
            dt = total - pt
            if dt > 0:
                util = 100.0 * (1.0 - (idle - pi) / dt)
        PREV_STAT[0] = (idle, total)
    fl = [int(x) / 1e6 for x in fr.split(",") if x.strip().isdigit()]  # GHz
    return {
        "t": round((ts - T0[0]) / 1000.0, 2),
        "power": round(power_w(geti(pw), geti(cu), geti(vo)), 3),
        "status": st.strip(),
        "cpu": _maxc(cpu), "npu": _maxc(npu), "gpu": _maxc(gpu),
        "ddr": _maxc(ddr), "skin": _maxc(sk),
        "util": round(util, 1) if util is not None else None,
        "f0": fl[0] if len(fl) > 0 else None,
        "f4": fl[1] if len(fl) > 1 else None,
        "f7": fl[2] if len(fl) > 2 else None,
    }

def sampler_thread(csv_path):
    proc = subprocess.Popen(ADB + ["shell", f"su -c 'sh {DEV_SAMPLER}'"],
                            stdout=subprocess.PIPE, text=True, bufsize=1)
    csv = open(csv_path, "w") if csv_path else None
    if csv:
        csv.write("t,power_w,status,cpu_c,npu_c,gpu_c,ddr_c,skin_c,util,f0,f4,f7\n")
    for line in proc.stdout:
        s = parse(line.strip())
        if not s:
            continue
        with LOCK:
            BUF.append(s)
        if csv:
            csv.write(",".join(str(s.get(k, "") if s.get(k) is not None else "")
                      for k in ["t","power","status","cpu","npu","gpu","ddr","skin","util","f0","f4","f7"]) + "\n")
            csv.flush()

# ----------------------------------------------------------------------------- web server
HTML = r"""<!doctype html><html><head><meta charset=utf-8><title>OnePlus 15 — live</title>
<style>
 body{margin:0;background:#0e1116;color:#cdd6e4;font:13px -apple-system,Segoe UI,Roboto,sans-serif}
 header{padding:8px 14px;background:#161b22;border-bottom:1px solid #30363d;display:flex;gap:18px;align-items:center;position:sticky;top:0}
 header b{font-size:15px;color:#fff} .pill{padding:2px 9px;border-radius:10px;font-weight:600}
 .grid{display:grid;grid-template-columns:1fr 1fr;gap:1px;background:#30363d}
 .panel{background:#0e1116;padding:6px 10px} .panel h3{margin:2px 0 0;font-size:12px;color:#8b949e;font-weight:600}
 .panel .now{float:right;font-size:15px;font-weight:700} canvas{width:100%;height:120px;display:block}
</style></head><body>
<header><b>OnePlus 15 · live resources</b>
 <span>status <span id=st class=pill>—</span></span>
 <span id=rate style="color:#8b949e"></span>
 <span style="margin-left:auto;color:#8b949e">SM8850 · Snapdragon 8 Elite Gen5 · 4 Hz</span></header>
<div class=grid id=grid></div>
<script>
const P=[
 {k:'power',t:'Battery power',u:'W',c:'#3fb950'},
 {k:'cpu',t:'CPU temp',u:'°C',c:'#f85149'},
 {k:'gpu',t:'GPU temp',u:'°C',c:'#d29922'},
 {k:'npu',t:'NPU temp (Hexagon)',u:'°C',c:'#a371f7'},
 {k:'ddr',t:'DDR temp',u:'°C',c:'#db61a2'},
 {k:'skin',t:'Skin temp',u:'°C',c:'#ff9b72'},
 {k:'util',t:'CPU utilization',u:'%',c:'#58a6ff'},
 {k:'freq',t:'CPU freq (c0/c4/c7)',u:'GHz',c:'#79c0ff'},
];
const grid=document.getElementById('grid');
P.forEach(p=>{grid.insertAdjacentHTML('beforeend',
 `<div class=panel><h3>${p.t} <span class=now id=now_${p.k}></span></h3><canvas id=c_${p.k}></canvas></div>`);});
function draw(cv,series,color,unit){
 const dpr=devicePixelRatio||1,w=cv.clientWidth,h=cv.clientHeight;
 cv.width=w*dpr;cv.height=h*dpr;const x=cv.getContext('2d');x.scale(dpr,dpr);
 x.clearRect(0,0,w,h);
 let all=[].concat(...series.map(s=>s.v.filter(v=>v!=null)));
 if(!all.length)return;let mn=Math.min(...all),mx=Math.max(...all);
 if(mx-mn<1e-6){mx+=1;mn-=1;}const pad=(mx-mn)*0.12;mn-=pad;mx+=pad;
 const ts=series[0].t,t0=ts[0],t1=ts[ts.length-1]||t0+1;
 const px=t=>(t-t0)/(t1-t0||1)*(w-44)+40, py=v=>h-14-(v-mn)/(mx-mn)*(h-22);
 x.strokeStyle='#21262d';x.fillStyle='#6e7681';x.font='10px sans-serif';
 for(let i=0;i<=2;i++){const v=mn+(mx-mn)*i/2,yy=py(v);
  x.beginPath();x.moveTo(40,yy);x.lineTo(w,yy);x.stroke();x.fillText(v.toFixed(1),2,yy+3);}
 series.forEach((s,si)=>{x.strokeStyle=s.c||color;x.lineWidth=1.6;x.beginPath();let st=false;
  for(let i=0;i<s.v.length;i++){if(s.v[i]==null){st=false;continue;}
   const X=px(s.t[i]),Y=py(s.v[i]);if(!st){x.moveTo(X,Y);st=true;}else x.lineTo(X,Y);}x.stroke();});
}
async function tick(){
 let d;try{d=await(await fetch('/data')).json();}catch(e){return;}
 document.getElementById('st').textContent=d.status||'—';
 const dis=(d.status||'').toLowerCase().includes('dis');
 document.getElementById('st').style.background=dis?'#3fb950':'#6e7681';
 document.getElementById('st').style.color='#fff';
 document.getElementById('rate').textContent=d.n+' samples · '+(d.t.slice(-1)[0]||0).toFixed(0)+'s';
 P.forEach(p=>{
  const cv=document.getElementById('c_'+p.k);
  let series,last;
  if(p.k==='freq'){
   series=[{t:d.t,v:d.f0,c:'#58a6ff'},{t:d.t,v:d.f4,c:'#79c0ff'},{t:d.t,v:d.f7,c:'#a5d6ff'}];
   last=d.f7.filter(v=>v!=null).slice(-1)[0];
  }else{series=[{t:d.t,v:d[p.k],c:p.c}];last=d[p.k].filter(v=>v!=null).slice(-1)[0];}
  draw(cv,series,p.c,p.u);
  document.getElementById('now_'+p.k).textContent=(last==null?'—':last.toFixed(p.k==='power'||p.k==='freq'?2:1))+' '+p.u;
 });
}
setInterval(tick,500);tick();
</script></body></html>"""

class H(BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def do_GET(self):
        if self.path.startswith("/data"):
            with LOCK:
                rows = list(BUF)
            keys = ["t","power","cpu","npu","gpu","ddr","skin","util","f0","f4","f7"]
            out = {k: [r.get(k) for r in rows] for k in keys}
            out["status"] = rows[-1]["status"] if rows else "—"
            out["n"] = len(rows)
            body = json.dumps(out).encode()
            self.send_response(200); self.send_header("Content-Type","application/json")
            self.send_header("Content-Length",str(len(body))); self.end_headers(); self.wfile.write(body)
        else:
            body = HTML.encode()
            self.send_response(200); self.send_header("Content-Type","text/html; charset=utf-8")
            self.send_header("Content-Length",str(len(body))); self.end_headers(); self.wfile.write(body)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8717)
    ap.add_argument("--hz", type=float, default=4.0)
    ap.add_argument("--csv", default=None)
    a = ap.parse_args()

    if adb("shell", "echo ok").stdout.strip() != "ok":
        sys.exit("ERROR: phone not reachable. Set ADB env or check `adb devices`.")
    print("[dash] resolving thermal zones…")
    fam = resolve_zones()
    print(f"[dash] zones: cpu={len(fam['cpu'])} npu={len(fam['npu'])} gpu={len(fam['gpu'])} "
          f"ddr={len(fam['ddr'])} skin={len(fam['skin'])}")
    script = build_sampler(fam, a.hz)
    adb("shell", f"cat > {DEV_SAMPLER} <<'EOF'\n{script}\nEOF")
    threading.Thread(target=sampler_thread, args=(a.csv,), daemon=True).start()

    srv = ThreadingHTTPServer(("127.0.0.1", a.port), H)
    print(f"\n[dash] ✅ open  http://127.0.0.1:{a.port}   (Ctrl+C to stop)")
    if a.csv: print(f"[dash] logging samples -> {a.csv}")
    print("[dash] reminder: wattage is real only while battery status = Discharging.\n")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        adb("shell", f"touch {DEV_STOP}")
        print("\n[dash] stopped, sampler flagged to exit.")

if __name__ == "__main__":
    main()

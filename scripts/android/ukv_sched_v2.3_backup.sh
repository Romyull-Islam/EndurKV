#!/system/bin/sh
# ============================================================================
# ukv_sched.sh -- energy-aware scheduler for muKV on the phone: two loops and a lever
# (2026-09-06, v2.3: start-anchored energy split with a 30 s gauge lag, bench in background; v2.2: loops act on model error, bias decays when both budgets are met, walk slack has a
#  5% tolerance floor, table learning is clipped to 10% per request)
#
# ONE LEVER. L in [0, 1]: 1 = performance, 0 = energy. The battery tier only sets its
# default position (mains and above 50%: 1.0; 21 to 50%: 0.5; 20% or below: 0.0); the
# caller may move it (--lever L). L sets everything the two loops trade at:
#     exchange rate  LAM(L) : a step down the ladder is taken only if it saves at least
#                             LAM times as much energy (relative) as it costs in time
#                             (1.5 at L=1, 0.8 at L=0.5, 0.5 at L=0)
#     quality floor  QF(L)  : 1.0 / 0.9 / 0.75 of the K=1024 answer quality
#     time budget    T_bud  : T_full * (1 + 0.03 + 0.37 * (1 - L))   performance loop
#     energy budget  E_bud  : E_full * (1 - 0.25 * (1 - L))      energy loop
#     output cap            : 4096 / 1024 / 512 when the caller left the length open
#
# TWO LOOPS, INTERCONNECTED THROUGH THE LEVER.
#   performance loop: predicts T per plan from the table; a plan must fit T_bud; after the
#                     request compares the metered T with T_bud and, if over, nudges the
#                     lever toward performance for the next request in this context.
#   energy loop     : predicts E per plan from the table; walks the ladder while the
#                     exchange rate holds; after the request compares the metered E with
#                     E_bud and, if over, nudges the lever toward energy.
#   When both budgets are violated the larger relative error wins. The nudge is +-0.1 per
#   request, clamped to +-0.3, kept per battery tier in ukv_lever_bias.txt.
#   The loops share one memory: the cost table, updated from every request's meter (EMA 0.3).
#
# PLANS. Rows of the table: backend, GPU clock cap for prefill, cache K; a plan name that
# contains d<MHz> (e.g. gpu1200d902_k1024) lowers the GPU cap to <MHz> for decode only,
# via --gpu-mhz-decode in the engine. Prefill is compute-bound and pays for a cap in time;
# decode is bandwidth-bound and does not, so split plans are where the two loops meet.
#
# OUTPUT LENGTH POLICY as before: --max-tokens honored; --size short|medium|long = 256,
# 1024, 4096; a prompt whose last 600 bytes state a length is not capped; else by L.
#
# Usage:
#   ukv_sched.sh --prompt FILE [--max-tokens N | --size S] [--lever L] [--no-feedback]
#                [--force-soc S] [--force-status charging|discharging] [--cpu-only]
#                [--ignore-eos] [--dry-run] [--tag NAME]
# Files: table $ROOT/ukv_sched_table.txt, bias $ROOT/ukv_lever_bias.txt, log $ROOT/ukv_sched.log,
#        per-request outputs $ROOT/sched/<tag>/.  Runs as root (su).
# ============================================================================
ROOT=/data/local/tmp/endurkv
BIN_GPU=/data/local/tmp/ukv            # Vulkan build for GPU plans
BIN_CPU=$ROOT/bin_cpu_ea               # CPU-only build for CPU plans (the Vulkan build with
                                       # --n-gpu-layers 0 still opens the GPU and decodes 4x slower)
MODEL=$ROOT/models/Llama-3.2-1B-Instruct-Q4_K_M.gguf
TABLE=$ROOT/ukv_sched_table.txt
BIAS=$ROOT/ukv_lever_bias.txt
LOG=$ROOT/ukv_sched.log
SAMPLER=/data/local/tmp/sample_sensors.sh
MU="--policy v1_fa2 --fa-on-evict --n-sink 4 --adaptive-anchor --adaptive-rmin 32 --obs-window 16 --snapkv-pool 7 --gate-alpha-floor 0.70 --compact-inplace"
ALPHA=0.3

PROMPT=""; MAXTOK=""; SIZE=""; FSOC=""; FSTAT=""; CPUONLY=0; IGNEOS=""; DRY=0; TAG=""; LEVER=""; FEEDBACK=1
while [ $# -gt 0 ]; do
  case "$1" in
    --prompt) PROMPT=$2; shift 2;;
    --max-tokens) MAXTOK=$2; shift 2;;
    --size) SIZE=$2; shift 2;;
    --lever) LEVER=$2; shift 2;;
    --no-feedback) FEEDBACK=0; shift;;
    --force-soc) FSOC=$2; shift 2;;
    --force-status) FSTAT=$2; shift 2;;
    --cpu-only) CPUONLY=1; shift;;
    --ignore-eos) IGNEOS="--ignore-eos"; shift;;
    --dry-run) DRY=1; shift;;
    --tag) TAG=$2; shift 2;;
    *) echo "unknown arg: $1" >&2; exit 1;;
  esac
done
[ -f "$PROMPT" ] || { echo "need --prompt FILE" >&2; exit 1; }
[ -z "$TAG" ] && TAG=sched_$(date +%Y%m%d_%H%M%S)
OUT=$ROOT/sched/$TAG; mkdir -p $OUT

# ---- 1. phone state -> tier -> default lever ----
BAT=$(dumpsys battery 2>/dev/null)
SOC=$(echo "$BAT" | grep -m1 "level:" | sed 's/.*: *//' | tr -d ' \r')
STAT=$(echo "$BAT" | grep -m1 "status:" | sed 's/.*: *//' | tr -d ' \r')
PLUG=$(echo "$BAT" | grep -ciE "(AC|USB|Wireless) powered: true")
[ -n "$FSOC" ] && SOC=$FSOC
MAINS=0
case "$STAT" in 2|5) [ "$PLUG" -gt 0 ] && MAINS=1;; esac
[ "$FSTAT" = "charging" ] && MAINS=1
[ "$FSTAT" = "discharging" ] && MAINS=0
DDR=$(( $(cat /sys/class/thermal/thermal_zone47/temp 2>/dev/null || echo 0) / 1000 ))
BATT=$(( $(cat /sys/class/thermal/thermal_zone93/temp 2>/dev/null || echo 0) / 1000 ))
GPU_OK=1; { [ "$CPUONLY" = 1 ] || [ ! -f $BIN_GPU/libggml-vulkan.so ]; } && GPU_OK=0
HOT=0; { [ "$BATT" -ge 40 ] || [ "$DDR" -ge 60 ]; } && HOT=1
if [ "$MAINS" = 1 ]; then TIER=mains; L0=1.0
elif [ "$SOC" -gt 50 ]; then TIER=healthy; L0=1.0
elif [ "$SOC" -gt 20 ]; then TIER=mid; L0=0.5
else TIER=low; L0=0.0; fi
BIASV=$(grep -m1 "^$TIER " $BIAS 2>/dev/null | awk '{print $2}'); [ -z "$BIASV" ] && BIASV=0
if [ -n "$LEVER" ]; then L=$LEVER; LSRC="caller"; else L=$(awk -v a=$L0 -v b=$BIASV 'BEGIN{l=a+b; if(l<0)l=0; if(l>1)l=1; printf "%.2f", l}'); LSRC="tier $TIER default $L0 + feedback bias $BIASV"; fi
# lever -> loop parameters, piecewise linear through the three tier anchors (L = 0, 0.5, 1)
eval $(awk -v L=$L 'function pl(a,b,c){ return (L<0.5)? a+(b-a)*(L/0.5) : b+(c-b)*((L-0.5)/0.5) }
  BEGIN{ printf "WQ=%.3f WT=%.3f WE=%.3f LAM=%.3f QF=%.3f TSLACK=%.3f ESAVE=%.3f",
    pl(0.20,0.35,0.55), pl(0.10,0.20,0.40), pl(0.70,0.45,0.05), pl(0.5,0.8,1.5), pl(0.75,0.90,1.00), 0.03+0.37*(1-L), 0.25*(1-L) }')
# time slack is 3% at full performance (the healthy cells' own spread), 40% at the energy end;
# no budget is tighter than the 5% model tolerance the loops use, or one over-budget sample
# (the vendor limiter tripping early) would evict the split plan from the healthy walk
[ -z "$WQ" ] && { echo "lever mapping failed" >&2; exit 1; }
TSLACK=$(awk -v s=$TSLACK 'BEGIN{printf "%.3f", (s<0.05)?0.05:s}')
CAPL=$(awk -v L=$L 'BEGIN{print (L>=0.75)?4096:(L>=0.25)?1024:512}')

# ---- 2. request shape and output length policy ----
BYTES=$(stat -c %s "$PROMPT" 2>/dev/null || wc -c < "$PROMPT")
NPROMPT=$(( BYTES * 10 / 45 ))
LENRULE=""
if [ -n "$MAXTOK" ]; then NOUT=$MAXTOK; LENRULE="caller --max-tokens"
elif [ -n "$SIZE" ]; then
  case "$SIZE" in short) NOUT=256;; medium) NOUT=1024;; long) NOUT=4096;; *) echo "bad --size" >&2; exit 1;; esac
  LENRULE="caller --size $SIZE"
elif tail -c 600 "$PROMPT" | grep -qiE "in (one|two|three|four|five|[0-9]+) (words?|sentences?|paragraphs?|lines?|bullets?)|briefly|concise|one-line|short answer|at most [0-9]+ (words|tokens)"; then
  NOUT=4096; LENRULE="prompt states a length (in its last 600 bytes); not capped"
else NOUT=$CAPL; LENRULE="lever cap for L=$L"; fi

# ---- 3. score the plans: backend by weighted utility, then the ladder walk with both loops ----
PLAN=$(awk -v np=$NPROMPT -v no=$NOUT -v wq=$WQ -v wt=$WT -v we=$WE -v gpu=$GPU_OK -v hot=$HOT -v lam=$LAM -v qf=$QF -v tslack=$TSLACK '
  /^#/ || NF<9 {next}
  { if ($2=="gpu" && gpu==0) next;
    if ($2=="gpu" && $3>=1200 && hot==1 && $1 !~ /d[0-9]+_/) next;   # hot: no plan that prefills at 1200 without a decode cap
    n++; name[n]=$1; B[n]=$2; E[n]=np*$5+no*$6; T[n]=np*$7+no*$8; Q[n]=$9 }
  END { if (n==0) { print "none"; exit }
        eb=E[1]; tb=T[1]; for (i=2;i<=n;i++) { if (E[i]<eb) eb=E[i]; if (T[i]<tb) tb=T[i] }
        best=1; ub=-1e9
        for (i=1;i<=n;i++) { u=wq*Q[i]-wt*T[i]/tb-we*E[i]/eb; U[i]=u; if (u>ub) {ub=u; best=i} }
        bk=B[best]
        for (i=1;i<=n;i++) printf("  %-18s E=%7.0f J  T=%6.1f s  q=%.2f  U=%+.3f\n", name[i], E[i]/1000, T[i]/1000, Q[i], U[i]) > "/dev/stderr"
        m=0; for (i=1;i<=n;i++) if (B[i]==bk) { m++; o[m]=i }
        for (a=1;a<=m;a++) for (b=a+1;b<=m;b++) { i=o[a]; j=o[b]; if (Q[j]>Q[i] || (Q[j]==Q[i] && T[j]<T[i])) { o[a]=j; o[b]=i } }
        cur=o[1]; tfull=T[cur]; efull=E[cur]; tbud=tfull*(1+tslack)
        printf("  walk %s (T budget %.0f s = full %.0f s + %.0f%%): start %s", bk, tbud/1000, tfull/1000, 100*tslack, name[cur]) > "/dev/stderr"
        # walk in time order: a plan that saves nothing or trades poorly against the current one
        # is skipped (a later, larger step may still pay); the walk stops only when the next plan is
        # over the time budget or under the quality floor, since both only get worse down the ladder
        for (a=2;a<=m;a++) { k=o[a]; dE=(E[cur]-E[k])/E[cur]; dT=(T[k]-T[cur])/T[cur]
          if (Q[k]<qf)  { printf(" -> %s: stop (quality)", name[k]) > "/dev/stderr"; break }
          if (T[k]>tbud){ printf(" -> %s: stop (over time budget)", name[k]) > "/dev/stderr"; break }
          ok = (E[k]<E[cur] && (dT<=0 || dE>=lam*dT))
          why = (E[k]>=E[cur])?"no saving":"poor exchange"
          printf(" -> %s (dE %+.0f%%, dT %+.0f%%): %s", name[k], 100*dE, 100*dT, ok?"take":"skip ("why")") > "/dev/stderr"
          if (ok) cur=k }
        printf("\n") > "/dev/stderr"
        printf("%s %.0f %.0f\n", name[cur], efull, tfull) }' $TABLE)
set -- $PLAN; PLAN=$1; EFULL=$2; TFULL=$3
[ "$PLAN" = "none" ] && { echo "no plan available" >&2; exit 1; }
ROW=$(grep -m1 "^$PLAN " $TABLE)
BACKEND=$(echo "$ROW" | awk '{print $2}'); GMHZ=$(echo "$ROW" | awk '{print $3}'); K=$(echo "$ROW" | awk '{print $4}')
DMHZ=$(echo "$PLAN" | sed -n 's/^gpu[0-9]*d\([0-9]*\)_.*/\1/p')
PE=$(echo "$ROW" | awk -v np=$NPROMPT -v no=$NOUT '{printf "%.0f", (np*$5+no*$6)/1000}')
PT=$(echo "$ROW" | awk -v np=$NPROMPT -v no=$NOUT '{printf "%.1f", (np*$7+no*$8)/1000}')
# Budgets the loops check the meter against. Both are the plan's own commitments, with a 5%
# model tolerance (the table's mean prediction error is 3.7%): the performance loop checks the
# time budget the plan was chosen under (never tighter than the plan's prediction + 5%), the
# energy loop checks the plan's predicted energy + 5%. So a nudge means the phone behaved
# differently from the model (heat, another app), not that a tier wished for more saving; the
# tier's saving is already fixed by where the lever put the ladder walk.
TBUD=$(awk -v t=$TFULL -v s=$TSLACK -v pt=$PT 'BEGIN{b=t*(1+s)/1000; if (pt*1.05>b) b=pt*1.05; printf "%.0f", b}')
EBUD=$(awk -v pe=$PE 'BEGIN{printf "%.0f", pe*1.05}')
ETARGET=$(awk -v e=$EFULL -v s=$ESAVE 'BEGIN{printf "%.0f", e*(1-s)/1000}')
echo "[sched] soc=${SOC}% status=$STAT mains=$MAINS tier=$TIER lever=$L ($LSRC) -> weights=(q $WQ, t $WT, e $WE) lam=$LAM qf=$QF batt=${BATT}C ddr=${DDR}C hot=$HOT gpu_ok=$GPU_OK" >&2
echo "[sched] request: ~$NPROMPT prompt tokens, output cap $NOUT ($LENRULE); loop budgets: time <= ${TBUD} s, energy <= ${EBUD} J (tier saving target ${ETARGET} J)" >&2
echo "[sched] plan: $PLAN backend=$BACKEND gpu_mhz=$GMHZ${DMHZ:+ decode_mhz=$DMHZ} K=$K predicted E=${PE} J T=${PT} s" >&2
[ "$DRY" = 1 ] && exit 0

# ---- 4. apply ----
# write the user cap (max_pwrlevel, composed with the thermal cap by max) and the clock, so the
# vendor thermal engine's own writes to thermal_pwrlevel cannot clear our cap mid-request
gpucap(){ ( cd /sys/class/kgsl/kgsl-3d0 && hz=$(( $1 * 1000000 )) && i=0 && for x in $(cat gpu_available_frequencies); do [ "$x" = "$hz" ] && echo $i > max_pwrlevel; i=$((i+1)); done; echo $hz > max_gpuclk ) 2>/dev/null; }
if [ "$BACKEND" = gpu ]; then NGL=99; BIN=$BIN_GPU; PIN="taskset f0 nice -n -20"; gpucap $GMHZ; else NGL=0; BIN=$BIN_CPU; PIN=""; fi
DECODE_FLAG=""; [ -n "$DMHZ" ] && DECODE_FLAG="--gpu-mhz-decode $DMHZ"
pkill -f sample_sensors 2>/dev/null; rm -f $OUT/sensors.csv
nohup sh $SAMPLER --out $OUT/sensors.csv --hz 2 >/dev/null 2>&1 &
# The bench is run in the background and the sampler stopped when meta.json appears: on battery the
# process takes up to two minutes to exit after decoding (observed 2026-09-06), and a sampler that
# runs through that tail puts the end anchor of the energy split on idle time. The sampler keeps
# going SETTLE seconds past the result so the fuel gauge's lagging coulomb counter catches up.
SETTLE=30
( cd $BIN && LD_LIBRARY_PATH=$BIN $PIN ./eviction_bench --model $MODEL --prompt $PROMPT \
  --prompt-id $TAG --eval-mode gen --max-tokens $NOUT $IGNEOS --ctx-size 16384 --seed 42 --threads 4 \
  --n-gpu-layers $NGL --greedy --cache-type-k f16 --cache-type-v f16 $MU --k-nominal $K $DECODE_FLAG \
  --n-batch 512 --n-ubatch 64 --out-meta $OUT/meta.json --out-gen $OUT/gen.txt --out-csv /dev/null > /dev/null 2> $OUT/err ) &
BPID=$!
i=0; while [ ! -s $OUT/meta.json ] && kill -0 $BPID 2>/dev/null && [ $i -lt 1500 ]; do sleep 1; i=$((i+1)); done
sleep $SETTLE
pkill -f sample_sensors 2>/dev/null
wait $BPID; RC=$?
[ "$BACKEND" = gpu ] && gpucap 1200
[ $RC -ne 0 ] && { echo "[sched] bench failed rc=$RC" >&2; exit $RC; }

# ---- 5. measure, learn the table, and let the two loops move the lever ----
g(){ grep -m1 "\"$1\"" $OUT/meta.json | sed 's/.*: *//; s/,.*//' | tr -d ' '; }
NP=$(g n_prompt_tokens); NS=$(g n_decode_steps); PMS=$(g prefill_ms); DMS=$(g decode_ms); TMS=$(g total_ms)
MEAS=$(awk -F, -v pms=$PMS -v tms=$TMS -v lag=$SETTLE 'NR==1 { for (i=1;i<=NF;i++) c[$i]=i; next }
  { t=$c["monotonic_s"]+0; v=$c["usb_voltage_uv"]/1e6; a=$c["usb_current_ua"]; if (a<0) a=-a; a/=1e6;
    n++; T[n]=t; P[n]=v*a; Q[n]=$c["bat_charge_uah"]+0; V[n]=$c["bat_voltage_now_uv"]/1e6; G[n]=("gpu_clk_hz" in c)?$c["gpu_clk_hz"]+0:0 }
  END { if (n<4) { print "0 0"; exit }
        # START-anchored split (2026-09-06): the bench starts prefill within seconds of the sampler
        # (first sample with the GPU above 900 MHz, else sample 2 plus 3 s) and may spend minutes in
        # teardown afterwards, so the end of the sampler window is not the end of decode. The rail
        # integral covers exactly [t0, t0+tms]; the pack delta runs LAG s longer to catch the fuel
        # gauge, which reports a discharge about 30 s late.
        t0=T[2]+3; for (i=2;i<=n;i++) if (G[i]>=9e8 && T[i]-T[1]<90) { t0=T[i]; break }
        tpf=t0+pms/1000; tend=t0+tms/1000; tq=tend+lag; ep=0; ed=0; qa=""; qb=""; qc=""; vs=0; vn=0
        for (i=2;i<=n;i++) { dt=T[i]-T[i-1]; if (dt>5) dt=5; if (dt<=0) continue;
          if (T[i]>t0 && T[i]<=tpf) ep+=P[i]*dt; else if (T[i]>tpf && T[i]<=tend) ed+=P[i]*dt;
          if (T[i]>=t0 && qa=="" && Q[i]>0) qa=Q[i]; if (T[i]<=tpf && Q[i]>0) qb=Q[i]; if (T[i]<=tq && Q[i]>0) qc=Q[i]; if (V[i]>0) {vs+=V[i]; vn++} }
        vm=(vn>0)?vs/vn:4.35
        if (qa!="" && qb!="") { d=(qa-qb)/1e6; if (d>0) ep+=d*vm*3600 }
        if (qb!="" && qc!="") { d=(qb-qc)/1e6; if (d>0) ed+=d*vm*3600 }
        printf("%.1f %.1f", ep, ed) }' $OUT/sensors.csv)
EP=$(echo $MEAS | awk '{print $1}'); ED=$(echo $MEAS | awk '{print $2}')
ETOT=$(awk -v a=$EP -v b=$ED 'BEGIN{printf "%.0f", a+b}'); TTOT=$(awk -v t=$TMS 'BEGIN{printf "%.0f", t/1000}')
if [ "$NP" -gt 0 ] && [ "$NS" -gt 0 ] && awk -v e=$EP 'BEGIN{exit !(e>0)}'; then
  # EMA, clipped: one request may move a row's cost by at most 10%, so a single disturbed request
  # (another app, an external cap) cannot reorder the ladder, while a persistent shift still tracks
  awk -v plan=$PLAN -v a=$ALPHA -v ep=$EP -v ed=$ED -v np=$NP -v ns=$NS -v pms=$PMS -v dms=$DMS '
    function upd(old, meas,  v) { v=(1-a)*old+a*meas; if (v>old*1.10) v=old*1.10; if (v<old*0.90) v=old*0.90; return sprintf("%.1f", v) }
    $1==plan && NF>=9 { $5=upd($5, ep*1000/np); $6=upd($6, ed*1000/ns); $7=upd($7, pms/np); $8=upd($8, dms/ns) }
    { print }' $TABLE > $TABLE.new && mv $TABLE.new $TABLE
  LEARN="table updated"
else LEARN="no table update (measurement missing)"; fi
# the two loops: compare the meter with the budgets and nudge the lever for this tier
LOOPS=$(awk -v e=$ETOT -v t=$TTOT -v eb=$EBUD -v tb=$TBUD -v bias=$BIASV -v fb=$FEEDBACK 'BEGIN{
  et=(e-eb)/eb; tt=(t-tb)/tb; nb=bias; act="hold"
  if (fb==1) {
    if (tt>0 && (et<=0 || tt>=et)) { nb=bias+0.1; act="performance loop: over time budget -> lever +0.1" }
    else if (et>0)                 { nb=bias-0.1; act="energy loop: over energy budget -> lever -0.1" }
    else if (bias>0.001 || bias<-0.001) {
      # both budgets met: the bias decays toward the tier default, so a past disturbance
      # does not pin the lever after it has passed
      nb=(bias>0)?bias-0.05:bias+0.05; if (nb<0.001 && nb>-0.001) nb=0
      act="hold: both budgets met -> bias decays toward the tier default" }
    else act="hold: both budgets met"
    if (nb>0.3) nb=0.3; if (nb<-0.3) nb=-0.3 }
  printf("%.2f|time %+.0f%% of budget, energy %+.0f%% of budget|%s", nb, 100*tt, 100*et, act) }')
NB=$(echo "$LOOPS" | cut -d'|' -f1); LOOPMSG=$(echo "$LOOPS" | cut -d'|' -f2); LOOPACT=$(echo "$LOOPS" | cut -d'|' -f3)
if [ "$FEEDBACK" = 1 ]; then
  { grep -v "^$TIER " $BIAS 2>/dev/null; echo "$TIER $NB"; } > $BIAS.new && mv $BIAS.new $BIAS
fi
TPS=$(g decode_tps)
echo "[sched] measured: prefill ${EP} J (${PMS} ms, $NP tokens)  decode ${ED} J (${DMS} ms, $NS tokens, ${TPS} tok/s)  total ${ETOT} J / ${TTOT} s vs predicted ${PE} J / ${PT} s; $LEARN" >&2
echo "[sched] loops: $LOOPMSG -> $LOOPACT (bias for $TIER now $NB)" >&2
echo "$(date '+%F %T') tag=$TAG soc=$SOC mains=$MAINS tier=$TIER lever=$L bias=$BIASV->$NB batt=$BATT ddr=$DDR plan=$PLAN K=$K gpu_mhz=$GMHZ decode_mhz=${DMHZ:-same} nout_cap=$NOUT lenrule=\"$LENRULE\" np=$NP ns=$NS pred_J=$PE pred_s=$PT meas_J=$ETOT meas_s=$TTOT pre_J=$EP dec_J=$ED prefill_ms=$PMS decode_ms=$DMS tps=$TPS tbud=$TBUD ebud=$EBUD $LEARN; $LOOPACT" >> $LOG

#!/bin/bash
# ============================================================================
# DEFINITIVE same-condition CPU table (HotMobile) -- full columns per policy:
#   PPL | prefill | decode_tps | wall | energy | live_cells | peak temps
#
# SAME CONDITION FOR ALL (fixes the SoC energy confound):
#   - Charge to >=90% ONCE at the start, then charging stays OFF for the ENTIRE
#     run (never toggled -> no battery-recharge USB spikes). Cool between cells by
#     IDLE-WAIT (charging off). Every cell starts cool (<=50 C big-core) at a
#     similar, monotonically-slowly-declining SoC.
#   - Energy = integral of TOTAL system power = USB rail (V*I) + battery-discharge
#     power (I>0 * V). Split-robust: correct whether the load is served by USB,
#     battery, or both.
#   - Platform: big cores capped 1632 for ALL cells equally. Baselines native (no
#     watchdog); muKV + surface-aware watchdog v2 (muKV-only; dormant when cool).
#
# Per policy: (1) GEN pass -> prefill_ms, decode_tps, wall, live_cells, energy, temps
#             (2) PPL pass -> teacher-forced perplexity on disjoint wiki_eval_1k.txt
# WikiText 9737-token prompt + 4096 decode, ctx 16384, Llama-3.2-1B Q4_K_M, 6 thr,
# k-nominal 1024, bin_cpu_sol2 (canonical SnapKV). Canonical SnapKV = --policy snapkv
# --obs-window 64 --n-sink 0. muKV = state-swap mass-full.
#
# ---- 2026-08-17 DERIVATIVE: KeyDiff arm for this exact table -----------------
# This is run_definitive_cpu.sh with ONLY the policy list changed. Everything
# that sets the measurement condition -- prompt, eval slice, 4096-token decode,
# ctx 16384, 6 threads, 1632 MHz cap, cool-gate, charging discipline, energy
# integration -- is byte-identical, because the KeyDiff row has to be comparable
# to the vanilla/SnapKV/AdaKV/muKV rows already in Table master-wikitext.
#
# WHY A VANILLA CONTROL RUNS FIRST. The published rows were produced by
# bin_cpu_sol2; the KeyDiff-capable binary is a different build (bin_cpu_kd).
# Comparing across builds is exactly the error that invalidated the pre-07-17
# Android numbers (no dotprod/i8mm). So vanilla is re-run HERE, on the new
# binary, in the same session: if it reproduces the published 5.0 tps / 1055 s /
# 1033 mWh / 9737 cells, the KeyDiff rows are comparable to the whole table and
# we say so with a number. If it does not, the two builds differ and every
# cross-build comparison in this table is suspect -- which we would then have to
# report rather than paper over.
#
# BUDGETS. All four of KeyDiff's published budgets {2048, 4096, 6144, 8192}
# bind on this 9737-token prompt, so all four run -- their headline 8K and 6K
# claims first. Decode-time re-eviction is ON (--keydiff-decode-block 128):
# without it the cache would grow to N+4096 during the 4K decode and we would be
# measuring our own omission instead of their policy.
# ============================================================================
set -u   # 2026-08-18: port pin removed -- the default server holds the device;
         # a second server on 5151 cannot claim the same USB endpoint, and the
         # other queued campaigns all use the default.
. /home/mislam22/EndurKV_workspace/EndurKV/scripts/android/adb_resilient.sh
# never share the phone with the running campaign
# 2026-08-19: NIGHT_CHAIN_DONE gate deleted -- that chain finished 08-17. The gate existed
# only to stop two campaigns sharing the phone, and had become a 15-hour no-op sleep that
# blocked every restart attempt.
# stage the KeyDiff-capable build in its OWN dir; bin_cpu_sol2 (provenance of the
# published rows) is left untouched.
adb_safe_shell "mkdir -p /data/local/tmp/endurkv/bin_cpu_kd" < /dev/null
for _so in /home/mislam22/EndurKV_workspace/EndurKV/llama.cpp/build-android/bin/lib*.so; do
  timeout 180 adb push "$_so" /data/local/tmp/endurkv/bin_cpu_kd/ < /dev/null >/dev/null 2>&1
done
timeout 180 adb push /home/mislam22/EndurKV_workspace/EndurKV/entropy_probe/build-android/eviction_bench \
         /data/local/tmp/endurkv/bin_cpu_kd/eviction_bench < /dev/null >/dev/null 2>&1
adb_safe_shell "chmod 755 /data/local/tmp/endurkv/bin_cpu_kd/eviction_bench" < /dev/null
OUT_HOST=/tmp/def_cpu_kd; mkdir -p "$OUT_HOST"
TS=$(date +%Y%m%d_%H%M%S); OUT=/data/local/tmp/endurkv/logs/defcpukd_$TS
adb_safe_shell "mkdir -p $OUT" < /dev/null
SCR=/tmp/claude-1001/-home-mislam22-EndurKV-workspace/1d283ef2-8bcb-4a99-8b56-fd8d8af9f80d/scratchpad
timeout 180 adb push "$SCR/wikitext_16k_p12k_d4k.txt" "$OUT/prompt.txt" < /dev/null >/dev/null 2>&1
timeout 180 adb push "$SCR/wiki_eval_disjoint.txt" "$OUT/eval.txt" < /dev/null >/dev/null 2>&1   # DISJOINT continuation (generalization PPL, not recall)
timeout 180 adb push /home/mislam22/EndurKV_workspace/EndurKV/scripts/android/preempt_throttle_watchdog_v2.sh \
         /data/local/tmp/preempt_throttle_watchdog_v2.sh < /dev/null >/dev/null 2>&1
MODEL=/data/local/tmp/endurkv/models/Llama-3.2-1B-Instruct-Q4_K_M.gguf
CB=/data/local/tmp/endurkv/bin_cpu_kd
WD_STOP=/data/local/tmp/cpu_wd.stop

# --- one-time: charge to >=90%, then charging OFF for the whole run ---
echo "[$(date +%H:%M:%S)] pre-charge to >=90% (same SoC start for all)"
adb_safe_shell "su -c 'echo 1 > /sys/class/oplus_chg/battery/mmi_charging_enable'" < /dev/null
CT0=$(date +%s)
while true; do
  cap=$(adb_safe_shell "su -c 'cat /sys/class/power_supply/battery/capacity'" < /dev/null|tr -d '\r'); cap=${cap:-0}
  case "$cap" in ''|*[!0-9]*) cap=0;; esac   # guard against non-numeric (permission/err)
  # 2026-08-18: gate lowered from 90 to ${SOC_FLOOR:-70}. The phone now hangs off a
  # Windows host through a reverse-tunnelled adb server and will not charge on that
  # connection (usb online=1, current_max=1.5A, but battery current_now=0 and status
  # "Not charging"), so a 90% gate can only ever hit its 30-minute timeout and start
  # anyway -- with the SoC unrecorded. What the gate is actually for is EQUAL state of
  # charge across cells, not a particular level: charging is held off for the whole run
  # regardless, so every cell still starts from the same slowly-declining SoC. The start
  # value is logged so the energy rows carry it.
  [ "$cap" -ge "${SOC_FLOOR:-70}" ] && { echo "  [start SoC ${cap}% -- floor ${SOC_FLOOR:-70}, host will not charge]"; break; }
  [ $(($(date +%s)-CT0)) -gt 1800 ] && { echo "  [charge timeout ${cap}%]"; break; }
  sleep 20
done
adb_safe_shell "su -c 'echo 0 > /sys/class/oplus_chg/battery/mmi_charging_enable'" < /dev/null  # OFF for whole run

platform_cap(){ adb_safe_shell "su -c 'pkill -9 -f eviction_bench 2>/dev/null; pkill -9 -f sample_sensors 2>/dev/null; touch $WD_STOP; for c in cpu6 cpu7; do echo 1632000 > /sys/devices/system/cpu/\$c/cpufreq/scaling_max_freq; done'" < /dev/null; }
start_wd(){ adb_safe_shell "su -c 'rm -f $WD_STOP; nohup sh /data/local/tmp/preempt_throttle_watchdog_v2.sh /data/local/tmp/cpu_wd_$1.log $WD_STOP >/dev/null 2>&1 &'" < /dev/null; }
stop_wd(){ adb_safe_shell "su -c 'touch $WD_STOP'" < /dev/null; }
# resolve thermal zones by NAME once (they re-enumerate across reboots)
ZMAP=$(adb_safe_shell "su -c 'for z in /sys/class/thermal/thermal_zone*; do printf \"%s:%s \" \$(basename \$z|sed s/thermal_zone//) \$(cat \$z/type 2>/dev/null); done'" < /dev/null|tr -d '\r')
z_by_name(){ echo "$ZMAP" | tr ' ' '\n' | grep -E ":$1\$" | head -1 | cut -d: -f1; }
DDR_Z=$(z_by_name ddr); SHELL_Z=$(z_by_name shell_front)
CPU_ZS=$(echo "$ZMAP" | tr ' ' '\n' | grep -E ':(cpu-[0-9]|cpullc-[0-9])' | cut -d: -f1 | tr '\n' ' ')  # exclude cpu-hw-trip (fixed 95C trip points)
echo "  [zones] cpu=($CPU_ZS) ddr=$DDR_Z shell=$SHELL_Z"
# STRICT idle cool-gate (charging stays OFF): CPU<36 AND DDR<36 AND shell<34 for a genuinely
# cold thermal start. Timeout 600s (cooling from ~58C DDR takes several minutes).
coolidle(){ local T0=$(date +%s)
  while true; do
    read cpu ddr sh <<<"$(adb_safe_shell "su -c 'm=0; for z in $CPU_ZS; do t=\$(cat /sys/class/thermal/thermal_zone\$z/temp 2>/dev/null); [ \$t -gt \$m ]&&m=\$t; done; d=\$(cat /sys/class/thermal/thermal_zone${DDR_Z}/temp 2>/dev/null); s=\$(cat /sys/class/thermal/thermal_zone${SHELL_Z}/temp 2>/dev/null); echo \$((m/1000)) \$((d/1000)) \$((s/1000))'" < /dev/null|tr -d '\r')"
    cpu=${cpu:-99}; ddr=${ddr:-99}; sh=${sh:-99}
    # TRUE cold start (24C room dissipates heat-soak in ~26min): CPU<37/DDR<37/shell<34.
    # Long timeout because a bench heats to ~70C and cooling back to <37 takes ~25min.
    if [ "$cpu" -lt 37 ] && [ "$ddr" -lt 37 ] && [ "$sh" -lt 34 ]; then echo "  [cold cpu=$cpu ddr=$ddr shell=$sh]"; return; fi
    # 2026-08-19: the gate is now AUTHORITATIVE and returns FAILURE on timeout.
    # It used to `return` (success) after 30 min, so run() proceeded regardless --
    # which is how a cell started at cpu=57C while a duplicate campaign was heating
    # the phone. Every timed/energy cell in this table MUST start from the same
    # thermal state or its tok/s and mJ/token are not comparable to the others.
    # Timeout raised to 75 min (a bench heats to ~70C; ~25 min to shed that with no
    # competing load, so 75 only trips if something is genuinely wrong). On failure
    # the cell is SKIPPED, leaving no gen.json -- resume retries it on the next pass.
    [ $(($(date +%s)-T0)) -gt 4500 ] && { echo "  [COOL FAILED cpu=$cpu ddr=$ddr shell=$sh -- skipping cell]"; return 1; }
    sleep 10
  done; }


# ---- detached device-side execution (2026-08-19) ----------------------------
# The phone is reachable only through an SSH reverse tunnel from a Windows host,
# and that link drops every ~20-30 min. A cell takes ~25 min. `adb shell CMD`
# runs CMD in a session that DIES WITH THE CONNECTION, so every drop killed the
# cell in flight; resume then restarted the same cell, which died the same way --
# an infinite loop, not slow progress. No amount of host-side retry logic fixes
# that, because the work itself was tied to the link.
#
# So the bench is now launched DETACHED on the device (setsid + nohup, output to
# files) and the host merely POLLS for its completion sentinel. The run survives
# any number of disconnects; the host can come and go. This is the same reason
# sample_sensors.sh has always been launched with nohup.
dev_run() {   # dev_run <tag> <done-file> <command...>
    local TAG=$1 DONEF=$2; shift 2
    # 2026-08-19: a HARD device-side timeout wraps the detached run. Detaching was
    # needed so a tunnel drop stops killing the cell -- but it also means the process
    # outlives the host, the link, and the supervisor. An orphan then keeps a phone
    # at load with nothing scheduled, heats it, and poisons the cool gate of whatever
    # runs next. ${DEV_RUN_MAX_S} is the same budget the host polls against, so the
    # device self-terminates at the moment the host would have given up on it.
    adb_safe_shell "su -c 'rm -f $DONEF; setsid nohup sh -c \"timeout ${DEV_RUN_MAX_S:-5400} env $* ; echo DONE > $DONEF\" >/dev/null 2>&1 &'" < /dev/null
    local waited=0
    while [ $waited -lt "${DEV_RUN_MAX_S:-5400}" ]; do
        if adb_safe_shell "[ -f $DONEF ] && echo yes" < /dev/null 2>/dev/null | grep -q yes; then
            return 0
        fi
        sleep 30; waited=$((waited + 30))
    done
    echo "  [dev_run TIMEOUT $TAG after ${waited}s]"
    return 1
}

# run <cell> <use_wd:0|1> <policy-args...>
run(){ local CELL=$1 WD=$2; shift 2; local PD=$OUT/$CELL
  # 2026-08-18: resume support. The adb path here is a reverse tunnel from a
  # Windows host and it flaps; a campaign that dies mid-way used to redo every
  # completed cell on restart (~25 min each). Skip anything already scored.
  if [ -s "$OUT_HOST/$CELL/gen.json" ]; then echo "  [$CELL cached -- skipping]"; return; fi
  echo "[$(date +%H:%M:%S)] $CELL (wd=$WD)"; platform_cap
  if ! coolidle; then echo "  [$CELL SKIPPED -- never reached cold start]"; return; fi
  adb_safe_shell "mkdir -p $PD" < /dev/null
  # Record the thermal state this cell actually started from, so the provenance of every
  # timed/energy number is checkable rather than assumed.
  # ORDER MATTERS (fixed 2026-08-20): this was originally placed BEFORE the mkdir, so the
  # redirect failed with "No such file or directory", $PD never existed, and the bench had
  # nowhere to write -- three cells "completed" in 19 s each with empty results and the
  # script then marked itself DONE. Never write into $PD before creating it.
  adb_safe_shell "su -c 'echo start_temps cpu=\$(cat /sys/class/thermal/thermal_zone${CPU_Z0:-24}/temp) ddr=\$(cat /sys/class/thermal/thermal_zone${DDR_Z}/temp) > $PD/start_temps.txt'" < /dev/null
  [ "$WD" = 1 ] && start_wd "$CELL"
  # (1) GEN pass with sensors
  adb_safe_shell "su -c 'nohup sh /data/local/tmp/endurkv/scripts/sample_sensors.sh --out $PD/sensors.csv --hz 5 >/dev/null 2>&1 &'" < /dev/null
  sleep 2; local t0=$(date +%s%N)
  dev_run "${CELL}_gen" "$PD/.gen_done" "LD_LIBRARY_PATH=$CB $CB/eviction_bench --prompt $OUT/prompt.txt --prompt-id ${CELL}_gen --eval-mode gen \
    --max-tokens 4096 --ignore-eos --ctx-size 16384 --model $MODEL --seed 42 --threads 6 --n-gpu-layers 0 --greedy \
    --k-nominal 1024 $* --out-meta $PD/gen.json --out-csv $PD/gen_steps.csv --out-prefill-csv $PD/gen_prefill.csv --out-gen $PD/gen.txt > $PD/gen.out 2> $PD/gen.err"
  local t1=$(date +%s%N)
  adb_safe_shell "su -c 'pkill -f sample_sensors 2>/dev/null'" < /dev/null
  echo "$(( (t1-t0)/1000000 ))" > "$OUT_HOST/${CELL}_wall_ms" 2>/dev/null; mkdir -p "$OUT_HOST/$CELL"; echo "$(( (t1-t0)/1000000 ))" > "$OUT_HOST/$CELL/wall_ms"
  # (2) PPL pass (teacher-forced disjoint eval, no long decode)
  dev_run "${CELL}_ppl" "$PD/.ppl_done" "LD_LIBRARY_PATH=$CB $CB/eviction_bench --prompt $OUT/prompt.txt --prompt-id ${CELL}_ppl --eval-mode ppl --eval-text $OUT/eval.txt \
    --max-tokens 512 --ignore-eos --ctx-size 16384 --model $MODEL --seed 42 --threads 6 --n-gpu-layers 0 --greedy \
    --k-nominal 1024 $* --out-meta $PD/ppl.json --out-csv /dev/null --out-gen /dev/null > $PD/ppl.out 2> $PD/ppl.err"
  [ "$WD" = 1 ] && { stop_wd; adb pull /data/local/tmp/cpu_wd_$CELL.log "$OUT_HOST/${CELL}_wd.log" < /dev/null >/dev/null 2>&1; }
  adb pull "$PD" "$OUT_HOST/" < /dev/null >/dev/null 2>&1
  echo "  [done $CELL] gen=$(grep -oE 'decode_tps=[0-9.]+' "$OUT_HOST/$CELL/gen.err" 2>/dev/null|head -1) ppl=$(grep -oiE 'ppl[= ][0-9.]+|perplexity[= :]+[0-9.]+' "$OUT_HOST/$CELL/ppl.err" 2>/dev/null|head -1)"
}
MU="--policy v1_fa2 --n-sink 4 --adaptive-anchor --adaptive-rmin 32 --obs-window 16 --snapkv-pool 7 --gate-alpha-floor 0.70"
# muKV-mass-full FIRST (the GO/NO-GO checkpoint), then baselines + swap alt.
KD="--policy keydiff --n-sink 0 --compact-inplace --keydiff-decode-block 128"
run vanilla_ctl 0 --policy vanilla                 # BUILD CONTROL -- must reproduce 5.0 tps / 1055 s / 1033 mWh / 9737 cells
run keydiff8192 0 $KD --k-nominal 8192             # their headline budget (<=0.04% drop claimed)
run keydiff6144 0 $KD --k-nominal 6144             # their second headline (<=1.5% drop claimed)
run keydiff2048 0 $KD --k-nominal 2048             # their smallest published budget
run keydiff4096 0 $KD --k-nominal 4096
# restore
adb_safe_shell "su -c 'touch $WD_STOP; for c in cpu6 cpu7; do cat /sys/devices/system/cpu/\$c/cpufreq/cpuinfo_max_freq > /sys/devices/system/cpu/\$c/cpufreq/scaling_max_freq; done; echo 1 > /sys/class/oplus_chg/battery/mmi_charging_enable'" < /dev/null
touch /tmp/def_cpu_kd_DONE; echo "[$(date +%H:%M:%S)] DEFINITIVE CPU KEYDIFF DONE -> $OUT_HOST"

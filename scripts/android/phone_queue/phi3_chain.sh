#!/system/bin/sh
# 2026-09-19, phone-resident, root, detached. Two jobs in sequence:
#  1. Phi-3-mini GPU repeats r2 and r3 for the five arms of the 6-page paper's Phi-3 block, with the
#     exact build, prompt and flags of run_phi3_gpu_complete.sh (bin_vk_cur, unpinned, f16 KV,
#     ctx 16384, 4096 generated), cooled per cell, charging off.
#  2. Recharge to 90%, then a second real discharge through the energy-aware scheduler with
#     Phi-3-mini (per-model cost table seeded from the Llama table). The USB input limit cannot be
#     written on this phone, so the battery carries the load only once the cable is pulled; until
#     then the loop runs one request per 20 min and flags each one usb_powered.
ROOT=/data/local/tmp/endurkv; RES=$ROOT/phi3_rep; mkdir -p $RES; LOG=$RES/chain.log
VK=$ROOT/bin_vk_cur; M=$ROOT/models/Phi-3-mini-128k-instruct-Q4_K_M.gguf; P=$ROOT/corpora/prompt_12k.txt
log(){ echo "$(date '+%F %T') $*" >> $LOG; }
MU="--policy v1_fa2 --fa-on-evict --compact-inplace --n-sink 4 --adaptive-anchor --adaptive-rmin 32 --obs-window 16 --snapkv-pool 7 --gate-alpha-floor 0.70"
settle(){
  a=0
  while [ $a -lt 3 ]; do
    a=$((a+1))
    . $ROOT/scripts/cool_gate.sh; cool_ddr36 >/dev/null 2>&1
    prev=999; same=0; i=0
    while [ $i -lt 90 ]; do
      d=$(( $(cat /sys/class/thermal/thermal_zone47/temp)/100 )); diff=$((d-prev)); [ $diff -lt 0 ] && diff=$((-diff))
      if [ $diff -le 3 ]; then same=$((same+1)); else same=0; fi
      [ $same -ge 3 ] && break; prev=$d; sleep 10; i=$((i+1))
    done
    sd=$(( $(cat /sys/class/thermal/thermal_zone47/temp)/1000 )); sb=$(( $(cat /sys/class/thermal/thermal_zone93/temp)/1000 ))
    log "  settle ddr=${sd}C batt=${sb}C"
    [ $sd -le 35 ] && [ $sb -le 33 ] && return 0
  done; return 1; }
cell(){
  TAG=$1; shift
  [ -s $RES/$TAG.json ] && { log "[$TAG] cached"; return; }
  log "[$TAG] cooling"; settle || log "[$TAG] hot after 3 settles, running anyway"
  rm -f $RES/$TAG.sensors.csv
  nohup sh /data/local/tmp/sample_sensors.sh --out $RES/$TAG.sensors.csv --hz 5 >/dev/null 2>&1 &
  log "[$TAG] running"
  cd $VK && LD_LIBRARY_PATH=$VK timeout 7200 ./eviction_bench --prompt $P --prompt-id $TAG --eval-mode gen \
    --max-tokens 4096 --ignore-eos --ctx-size 16384 --n-batch 512 --n-ubatch 64 --model $M --seed 42 --threads 4 \
    --n-gpu-layers 99 --greedy --cache-type-k f16 --cache-type-v f16 "$@" \
    --out-meta $RES/$TAG.json --out-gen $RES/$TAG.gen --out-csv /dev/null > /dev/null 2> $RES/$TAG.err
  pkill -f sample_sensors 2>/dev/null
  log "[$TAG] done $(grep -oE '"decode_tps": *[0-9.]+' $RES/$TAG.json 2>/dev/null)"
}
echo 0 > /sys/class/oplus_chg/battery/mmi_charging_enable
log "chain start"
for r in 2 3; do
  cell phi3_vanilla_r$r --policy vanilla
  cell phi3_mukv_r$r $MU --k-nominal 1024
  cell phi3_streamingllm_r$r --policy streamingllm --n-sink 4 --k-nominal 2000
  cell phi3_adakv_r$r --policy adakv --n-sink 0 --k-nominal 1024
  cell phi3_snapkv_r$r --policy snapkv --obs-window 16 --snapkv-kernel 5 --n-sink 0 --k-nominal 1024
done
log "REPEATS_DONE"
# ---- 2. recharge, then discharge with Phi-3 ----
echo 1 > /sys/class/oplus_chg/battery/mmi_charging_enable
t=0
while [ $t -lt 480 ]; do
  L=$(dumpsys battery | tr -d '\r' | awk '$1=="level:"{print $2}')
  [ "${L:-0}" -ge 90 ] && break
  sleep 60; t=$((t+1))
done
log "recharged to ${L}% after ${t} min"
rm -f $ROOT/tables/Phi-3-mini-128k-instruct-Q4_K_M.txt $ROOT/tables/Phi-3-mini-128k-instruct-Q4_K_M.bias.txt
OUTD=$ROOT/discharge_phi3 MODEL=Phi-3-mini-128k-instruct-Q4_K_M.gguf STOP_SOC=12 sh $ROOT/phone_discharge_loop.sh
log "CHAIN_DONE"

#!/system/bin/sh
# Waits for cpu_sllm_published.sh to finish, then runs muKV (frozen config, in-place compaction)
# on the same current CPU build, so muKV and StreamingLLM share one build and one full-cache base.
RES=/data/local/tmp/endurkv/qres_cpu; LOG=$RES/cpu.log
while ! grep -q CPU_DONE $LOG 2>/dev/null; do sleep 60; done
. /data/local/tmp/endurkv/cpu_sllm_published_funcs.sh
echo 0 > /sys/class/oplus_chg/battery/mmi_charging_enable
cell llama_mukv_cur /data/local/tmp/endurkv/models/Llama-3.2-1B-Instruct-Q4_K_M.gguf \
  --policy v1_fa2 --fa-on-evict --n-sink 4 --adaptive-anchor --adaptive-rmin 32 --obs-window 16 \
  --snapkv-pool 7 --gate-alpha-floor 0.70 --compact-inplace --k-nominal 1024
echo 1 > /sys/class/oplus_chg/battery/mmi_charging_enable
log "MUKV_DONE"

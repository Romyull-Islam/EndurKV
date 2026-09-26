#!/system/bin/sh
# muKV (frozen config, in-place compaction) on Bonsai-8B, current CPU build, same prompt and protocol as
# bonsai_vanilla_cur / bonsai_sllm_pub, so all three Bonsai arms share one build and one full-cache base.
. /data/local/tmp/endurkv/cpu_sllm_published_funcs.sh
echo 0 > /sys/class/oplus_chg/battery/mmi_charging_enable
cell bonsai_mukv_cur /data/local/tmp/endurkv/models/Bonsai-8B-Q1_0.gguf \
  --policy v1_fa2 --fa-on-evict --n-sink 4 --adaptive-anchor --adaptive-rmin 32 --obs-window 16 \
  --snapkv-pool 7 --gate-alpha-floor 0.70 --compact-inplace --k-nominal 1024
echo 1 > /sys/class/oplus_chg/battery/mmi_charging_enable
log "BONSAI_MUKV_DONE"

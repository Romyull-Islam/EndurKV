#!/bin/bash
# Push the phone-resident queue and start it detached. Needs the tunnel only for this moment.
export ANDROID_ADB_SERVER_PORT=${ANDROID_ADB_SERVER_PORT:-5160}
H=/home/mislam22/EndurKV_workspace/EndurKV/scripts/android
adb push $H/phone_queue/queue_gpu.sh /data/local/tmp/endurkv/queue_gpu.sh >/dev/null && \
adb push $H/phone_queue/queue_gpu.txt /data/local/tmp/endurkv/queue_gpu.txt >/dev/null && \
adb push $H/sample_sensors.sh /data/local/tmp/sample_sensors.sh >/dev/null && \
adb shell 'su -c "chmod 755 /data/local/tmp/endurkv/queue_gpu.sh; nohup sh /data/local/tmp/endurkv/queue_gpu.sh > /data/local/tmp/endurkv/queue_nohup.log 2>&1 &"' && \
sleep 3 && adb shell 'su -c "tail -3 /data/local/tmp/endurkv/qres/queue.log"'

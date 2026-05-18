#pragma once
#include <atomic>
#include <chrono>
#include <algorithm>
#include <stdio.h>

class ThermoKVMockHAL {
private:
    std::atomic<float> virtual_temp{35.0f}; 
    std::atomic<float> virtual_wear_gb{0.0f};
    
    const float TEMP_INCREASE_PER_MB = 0.05f; 
    const float TEMP_DECAY_PER_SEC = 1.2f;    

    std::chrono::time_point<std::chrono::steady_clock> last_update;

public:
    ThermoKVMockHAL() {
        last_update = std::chrono::steady_clock::now();
    }

    void record_io_write(float megabytes_written, float current_waf) {
        apply_thermal_decay(); 
        virtual_temp = virtual_temp + (megabytes_written * TEMP_INCREASE_PER_MB);
        virtual_wear_gb = virtual_wear_gb + ((megabytes_written * current_waf) / 1024.0f);
    }

    void apply_thermal_decay() {
        auto now = std::chrono::steady_clock::now();
        std::chrono::duration<float> elapsed = now - last_update;
        float decay = elapsed.count() * TEMP_DECAY_PER_SEC;
        virtual_temp = std::max(35.0f, virtual_temp.load() - decay);
        last_update = now;
    }

    float get_soc_temp() { apply_thermal_decay(); return virtual_temp; }
    float get_total_wear() { return virtual_wear_gb; }
};

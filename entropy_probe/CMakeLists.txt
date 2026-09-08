cmake_minimum_required(VERSION 3.14)
project(entropy_probe LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# --- Locate the pre-built llama.cpp ---
set(LLAMA_CPP_DIR  "${CMAKE_CURRENT_SOURCE_DIR}/../llama.cpp" CACHE PATH
    "Path to llama.cpp source tree (must already be built into <dir>/build)")
set(LLAMA_BUILD_DIR "${LLAMA_CPP_DIR}/build" CACHE PATH
    "Path to llama.cpp build directory")

if (NOT EXISTS "${LLAMA_CPP_DIR}/include/llama.h")
    message(FATAL_ERROR "llama.h not found at ${LLAMA_CPP_DIR}/include/llama.h. "
                        "Re-run cmake with -DLLAMA_CPP_DIR=<path-to-llama.cpp>.")
endif()

# Android NDK toolchain sets CMAKE_FIND_ROOT_PATH_MODE_LIBRARY=ONLY, which
# prevents find_library from looking outside the NDK sysroot — even with
# NO_DEFAULT_PATH + explicit PATHS. Relax to BOTH so the cross-built
# libllama.so / libggml*.so under $LLAMA_BUILD_DIR/bin can be located.
if (ANDROID)
    set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY BOTH)
endif()

find_library(LLAMA_LIBRARY
    NAMES llama
    PATHS "${LLAMA_BUILD_DIR}/bin"
    NO_DEFAULT_PATH
    REQUIRED
)
# We call ggml_backend_load_all() and ggml_time_us() directly, so we must
# link both libggml.so and libggml-base.so explicitly — otherwise the
# linker's default --as-needed drops these transitive deps of libllama.
find_library(GGML_LIBRARY
    NAMES ggml
    PATHS "${LLAMA_BUILD_DIR}/bin"
    NO_DEFAULT_PATH
    REQUIRED
)
find_library(GGML_BASE_LIBRARY
    NAMES ggml-base
    PATHS "${LLAMA_BUILD_DIR}/bin"
    NO_DEFAULT_PATH
    REQUIRED
)
# Modern llama.cpp ships backends as separate .so plugins (libggml-cpu.so,
# libggml-vulkan.so, libggml-cuda.so, ...). They must be NEEDED libs of the
# executable so the dynamic loader maps them up front; otherwise
# ggml_backend_load_all() at runtime has been observed to hang silently on
# Adreno (no stderr output, process stuck during init).
find_library(GGML_CPU_LIBRARY    NAMES ggml-cpu    PATHS "${LLAMA_BUILD_DIR}/bin" NO_DEFAULT_PATH)
find_library(GGML_VULKAN_LIBRARY NAMES ggml-vulkan PATHS "${LLAMA_BUILD_DIR}/bin" NO_DEFAULT_PATH)
find_library(GGML_CUDA_LIBRARY   NAMES ggml-cuda   PATHS "${LLAMA_BUILD_DIR}/bin" NO_DEFAULT_PATH)
set(GGML_BACKEND_LIBS "")
foreach(_lib ${GGML_CPU_LIBRARY} ${GGML_VULKAN_LIBRARY} ${GGML_CUDA_LIBRARY})
    if (_lib)
        list(APPEND GGML_BACKEND_LIBS ${_lib})
        message(STATUS "entropy_probe: linking backend ${_lib}")
    endif()
endforeach()
message(STATUS "entropy_probe: linking against ${LLAMA_LIBRARY}")
message(STATUS "entropy_probe: linking against ${GGML_LIBRARY}")
message(STATUS "entropy_probe: linking against ${GGML_BASE_LIBRARY}")

# --- Compile-time toggle for the per-step entropy/top-k computation ---
option(ENABLE_PROBE
    "Compile in per-step entropy / top-k computation and CSV output (off = decode-only)" ON)
message(STATUS "entropy_probe: ENABLE_PROBE=${ENABLE_PROBE}")

# --- Target ---
add_executable(entropy_probe    entropy_probe.cpp)
add_executable(attention_probe  attention_probe.cpp)
add_executable(prune_probe      prune_probe.cpp)
add_executable(controller_probe controller_probe.cpp)
add_executable(eviction_bench   eviction_bench.cpp)

# Per-target arch flags. On Android (cross-compile via NDK) -march=native is
# meaningless and the toolchain rejects it; pick an explicit aarch64 baseline
# instead. On a host build keep -march=native for the server study.
if (ANDROID)
    set(_arch_flag "-march=armv8.7-a")
else()
    set(_arch_flag "-march=native")
endif()

foreach(_t entropy_probe attention_probe prune_probe controller_probe eviction_bench)
    target_compile_options(${_t} PRIVATE
        -O3 -Wall -Wextra
        -ffast-math
        ${_arch_flag}
        -fopenmp-simd
        -funroll-loops
    )
    target_include_directories(${_t} PRIVATE
        "${LLAMA_CPP_DIR}/include"
        "${LLAMA_CPP_DIR}/ggml/include"
        "${CMAKE_CURRENT_SOURCE_DIR}"
    )
    target_link_libraries(${_t} PRIVATE
        ${LLAMA_LIBRARY} ${GGML_LIBRARY} ${GGML_BASE_LIBRARY} ${GGML_BACKEND_LIBS})
    # Force the linker to keep backend .so deps NEEDED even if no direct symbol
    # is referenced (their static constructors register the backend at load time).
    if (GGML_BACKEND_LIBS)
        target_link_options(${_t} PRIVATE "-Wl,--no-as-needed")
    endif()
    if (ANDROID)
        # On the phone the binary and its .so files will sit in the same
        # directory under /data/local/tmp; rpath $ORIGIN finds them without
        # requiring LD_LIBRARY_PATH at the shell.
        set_target_properties(${_t} PROPERTIES
            BUILD_RPATH   "\$ORIGIN"
            INSTALL_RPATH "\$ORIGIN"
        )
    else()
        set_target_properties(${_t} PROPERTIES
            BUILD_RPATH   "${LLAMA_BUILD_DIR}/bin"
            INSTALL_RPATH "${LLAMA_BUILD_DIR}/bin"
        )
    endif()
endforeach()

if (ENABLE_PROBE)
    # Only entropy_probe has an ENABLE_PROBE-gated path; attention_probe always probes.
    target_compile_definitions(entropy_probe PRIVATE ENABLE_PROBE=1)
endif()

# --- μKV×VL (2026-07-25): OPTIONAL multimodal support for eviction_bench ---
# If the llama.cpp build produced libmtmd (tools/mtmd), link it and enable the
# EVB_HAS_MTMD code path (--mmproj/--image). Builds without libmtmd (e.g. the
# phone build-android-vulkan tree) compile the exact same text-only binary.
# EVB_ENABLE_MTMD=OFF builds text-only (used for the PrismML fork, whose newer
# mtmd API differs: mtmd_helper_bitmap_init_from_file gained a 3rd parameter).
option(EVB_ENABLE_MTMD "Link libmtmd for --mmproj/--image (multimodal)" ON)
if (EVB_ENABLE_MTMD)
    find_library(MTMD_LIBRARY NAMES mtmd PATHS "${LLAMA_BUILD_DIR}/bin" NO_DEFAULT_PATH)
else()
    set(MTMD_LIBRARY "")
endif()
if (MTMD_LIBRARY)
    message(STATUS "entropy_probe: libmtmd found — eviction_bench gets --mmproj/--image (EVB_HAS_MTMD)")
    target_compile_definitions(eviction_bench PRIVATE EVB_HAS_MTMD=1)
    target_include_directories(eviction_bench PRIVATE "${LLAMA_CPP_DIR}/tools/mtmd")
    target_link_libraries(eviction_bench PRIVATE ${MTMD_LIBRARY})
else()
    message(STATUS "entropy_probe: libmtmd not found — eviction_bench is text-only")
endif()

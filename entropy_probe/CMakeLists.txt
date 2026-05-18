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
message(STATUS "entropy_probe: linking against ${LLAMA_LIBRARY}")
message(STATUS "entropy_probe: linking against ${GGML_LIBRARY}")
message(STATUS "entropy_probe: linking against ${GGML_BASE_LIBRARY}")

# --- Compile-time toggle for the per-step entropy/top-k computation ---
option(ENABLE_PROBE
    "Compile in per-step entropy / top-k computation and CSV output (off = decode-only)" ON)
message(STATUS "entropy_probe: ENABLE_PROBE=${ENABLE_PROBE}")

# --- Target ---
add_executable(entropy_probe   entropy_probe.cpp)
add_executable(attention_probe attention_probe.cpp)
add_executable(prune_probe     prune_probe.cpp)

# Per-target arch flags. On Android (cross-compile via NDK) -march=native is
# meaningless and the toolchain rejects it; pick an explicit aarch64 baseline
# instead. On a host build keep -march=native for the server study.
if (ANDROID)
    set(_arch_flag "-march=armv8.7-a")
else()
    set(_arch_flag "-march=native")
endif()

foreach(_t entropy_probe attention_probe prune_probe)
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
    target_link_libraries(${_t} PRIVATE ${LLAMA_LIBRARY} ${GGML_LIBRARY} ${GGML_BASE_LIBRARY})
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

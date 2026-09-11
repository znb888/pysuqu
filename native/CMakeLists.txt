cmake_minimum_required(VERSION 3.18)
project(pysuqu_native LANGUAGES CXX)

set(PYSUQU_NATIVE_MARCH_NATIVE "$ENV{PYSUQU_NATIVE_MARCH_NATIVE}" CACHE BOOL
    "Enable -march=native for GCC/Clang")
set(PYSUQU_NATIVE_PGO "$ENV{PYSUQU_NATIVE_PGO}" CACHE STRING
    "Profile guided optimization mode: OFF, GENERATE, or USE")
set_property(CACHE PYSUQU_NATIVE_PGO PROPERTY STRINGS OFF GENERATE USE)
string(TOUPPER "${PYSUQU_NATIVE_PGO}" PYSUQU_NATIVE_PGO_MODE)
if(PYSUQU_NATIVE_PGO_MODE STREQUAL "")
    set(PYSUQU_NATIVE_PGO_MODE OFF)
endif()
if(NOT PYSUQU_NATIVE_PGO_MODE MATCHES "^(OFF|GENERATE|USE)$")
    message(FATAL_ERROR "PYSUQU_NATIVE_PGO must be OFF, GENERATE, or USE")
endif()
set(PYSUQU_NATIVE_PGO_DIR "$ENV{PYSUQU_NATIVE_PGO_DIR}" CACHE PATH
    "Directory containing profile data")
if(PYSUQU_NATIVE_PGO_DIR STREQUAL "")
    set(PYSUQU_NATIVE_PGO_DIR "${CMAKE_BINARY_DIR}/pgo")
endif()

find_package(Python3 COMPONENTS Interpreter Development.Module REQUIRED)
find_package(Threads REQUIRED)
Python3_add_library(_dynamics MODULE WITH_SOABI dynamics.cpp)
target_compile_features(_dynamics PRIVATE cxx_std_17)
target_link_libraries(_dynamics PRIVATE Python3::Module Threads::Threads)

set(NATIVE_DETAIL_FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/detail/types.inc
    ${CMAKE_CURRENT_SOURCE_DIR}/detail/math_kernels.inc
    ${CMAKE_CURRENT_SOURCE_DIR}/detail/dopri.inc
    ${CMAKE_CURRENT_SOURCE_DIR}/detail/propagation.inc
    ${CMAKE_CURRENT_SOURCE_DIR}/detail/prepared_plan.inc
    ${CMAKE_CURRENT_SOURCE_DIR}/detail/python_api.inc)
set_property(SOURCE dynamics.cpp APPEND PROPERTY OBJECT_DEPENDS "${NATIVE_DETAIL_FILES}")

if(MSVC)
    if(PYSUQU_NATIVE_MARCH_NATIVE)
        message(FATAL_ERROR "PYSUQU_NATIVE_MARCH_NATIVE is unsupported by MSVC")
    endif()
    target_compile_options(_dynamics PRIVATE /O2 /EHsc)
    if(PYSUQU_NATIVE_PGO_MODE STREQUAL "GENERATE")
        target_compile_options(_dynamics PRIVATE /GL)
        target_link_options(_dynamics PRIVATE /LTCG:PGInstrument /GENPROFILE)
    elseif(PYSUQU_NATIVE_PGO_MODE STREQUAL "USE")
        target_compile_options(_dynamics PRIVATE /GL)
        target_link_options(_dynamics PRIVATE /LTCG:PGOptimize /USEPROFILE)
    endif()
elseif(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|AppleClang")
    target_compile_options(_dynamics PRIVATE -O3)
    if(PYSUQU_NATIVE_MARCH_NATIVE)
        target_compile_options(_dynamics PRIVATE -march=native)
    endif()
    if(PYSUQU_NATIVE_PGO_MODE STREQUAL "GENERATE")
        target_compile_options(_dynamics PRIVATE "-fprofile-generate=${PYSUQU_NATIVE_PGO_DIR}")
        target_link_options(_dynamics PRIVATE "-fprofile-generate=${PYSUQU_NATIVE_PGO_DIR}")
    elseif(PYSUQU_NATIVE_PGO_MODE STREQUAL "USE")
        target_compile_options(_dynamics PRIVATE "-fprofile-use=${PYSUQU_NATIVE_PGO_DIR}" -fprofile-correction)
        target_link_options(_dynamics PRIVATE "-fprofile-use=${PYSUQU_NATIVE_PGO_DIR}")
    endif()
else()
    if(PYSUQU_NATIVE_MARCH_NATIVE OR NOT PYSUQU_NATIVE_PGO_MODE STREQUAL "OFF")
        message(FATAL_ERROR "CPU tuning requires GCC/Clang or MSVC")
    endif()
    target_compile_options(_dynamics PRIVATE -O3)
endif()

install(TARGETS _dynamics
    LIBRARY DESTINATION pysuqu/_native
    RUNTIME DESTINATION pysuqu/_native
)

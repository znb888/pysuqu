cmake_minimum_required(VERSION 3.18)
project(pysuqu_native LANGUAGES CXX)

find_package(Python3 COMPONENTS Interpreter Development.Module REQUIRED)
find_package(Threads REQUIRED)
Python3_add_library(_dynamics MODULE WITH_SOABI dynamics.cpp)
target_compile_features(_dynamics PRIVATE cxx_std_17)
target_link_libraries(_dynamics PRIVATE Threads::Threads)

if(MSVC)
    target_compile_options(_dynamics PRIVATE /O2)
else()
    target_compile_options(_dynamics PRIVATE -O3)
endif()

install(TARGETS _dynamics
    LIBRARY DESTINATION pysuqu/_native
    RUNTIME DESTINATION pysuqu/_native
)

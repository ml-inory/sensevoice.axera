cmake_minimum_required(VERSION 3.13 FATAL_ERROR)
project(sensevoice)

set(CMAKE_CXX_STANDARD 17)

if (CMAKE_BUILD_TYPE MATCHES Debug)
    set(CMAKE_CXX_FLAGS "-fvisibility=hidden -g -O0")
elseif (CMAKE_BUILD_TYPE MATCHES Release)
    set(CMAKE_CXX_FLAGS "-fvisibility=hidden -O2")
endif()

include(cmake/msp_dependencies.cmake)
add_definitions(-DENV_HAS_STD_FILESYSTEM)
add_definitions(-DENV_HAS_POSIX_FILE_STAT)

include_directories(${MSP_INC_DIR})
link_directories(${MSP_LIB_DIR})

include_directories(src)
aux_source_directory(src SRC)
add_executable(sensevoice main.cpp ${SRC})
target_link_libraries(sensevoice ${MSP_LIBS})

install(TARGETS sensevoice
        RUNTIME
            DESTINATION .)
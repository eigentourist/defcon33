cmake_minimum_required(VERSION 3.10)
project(DefCon33 C)

set(CMAKE_C_STANDARD 99)

# --- matrix multiply demos ---
add_executable(matrix_mult_cpu src/matrix_mult_cpu.c)
add_executable(matrix_mult_opencl src/matrix_mult_opencl.c)

# --- game_of_life executable ---
add_executable(game_of_life src/main.c)

# --- list_platforms executable ---
add_executable(list_platforms src/list_platforms.c)

# -- Basic perceptrons, cpu and gpu version --
add_executable(perceptron_cpu src/perceptron_cpu.c)
add_executable(perceptron_opencl src/perceptron_opencl.c)

# -- Multi-layer perceptrons, cpu and gpu version --
add_executable(ml_perceptron_cpu src/ml_perceptron_cpu.c)
add_executable(ml_perceptron_opencl src/ml_perceptron_opencl.c)

# -- First convolutional neural nets, based on multi-player perceptrons --
add_executable(cnn_cpu src/cnn_cpu.c)
add_executable(cnn_opencl src/cnn_opencl.c)


# --- Platform-specific OpenCL setup ---
if(APPLE)
    message(STATUS "Detected macOS — using Homebrew OpenCL headers and system framework")

    # Intel Macs (default Homebrew path)
    target_include_directories(matrix_mult_opencl PRIVATE /usr/local/opt/opencl-headers/include)
    target_include_directories(game_of_life PRIVATE /usr/local/opt/opencl-headers/include)
    target_include_directories(list_platforms PRIVATE /usr/local/opt/opencl-headers/include)
    target_include_directories(perceptron_opencl PRIVATE /usr/local/opt/opencl-headers/include)
    target_include_directories(ml_perceptron_opencl PRIVATE /usr/local/opt/opencl-headers/include)
    target_include_directories(cnn_opencl PRIVATE /usr/local/opt/opencl-headers/include)

    # Apple Silicon (if needed):
    # target_include_directories(matrix_mult_opencl PRIVATE /opt/homebrew/opt/opencl-headers/include)
    # target_include_directories(game_of_life PRIVATE /opt/homebrew/opt/opencl-headers/include)
    # target_include_directories(list_platforms PRIVATE /opt/homebrew/opt/opencl-headers/include)
    # target_include_directories(perceptron_opencl PRIVATE /opt/homebrew/opt/opencl-headers/include)
    # target_include_directories(ml_perceptron_opencl PRIVATE /opt/homebrew/opt/opencl-headers/include)
    # target_include_directories(cnn_opencl PRIVATE /opt/homebrew/opt/opencl-headers/include)

    # Link against macOS OpenCL framework
    target_link_options(matrix_mult_opencl PRIVATE "-framework" "OpenCL")
    target_link_options(game_of_life PRIVATE "-framework" "OpenCL")
    target_link_options(list_platforms PRIVATE "-framework" "OpenCL")
    target_link_options(perceptron_opencl PRIVATE "-framework" "OpenCL")
    target_link_options(ml_perceptron_opencl PRIVATE "-framework" "OpenCL")
    target_link_options(cnn_opencl PRIVATE "-framework" "OpenCL")
else()
    # Linux or Windows
    find_package(OpenCL REQUIRED)
    target_link_libraries(matrix_mult_opencl OpenCL::OpenCL)
    target_link_libraries(game_of_life OpenCL::OpenCL)
    target_link_libraries(list_platforms OpenCL::OpenCL)
    target_link_libraries(perceptron_opencl OpenCL::OpenCL)
    target_link_libraries(ml_perceptron_opencl OpenCL::OpenCL)
    target_link_libraries(cnn_opencl OpenCL::OpenCL)
endif()

# --- ncurses (only needed by game_of_life) ---
find_package(Curses REQUIRED)
include_directories(${CURSES_INCLUDE_DIR})
target_link_libraries(game_of_life ${CURSES_LIBRARIES})

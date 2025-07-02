cmake_minimum_required(VERSION 3.10)
project(OpenCL_GameOfLife C)

add_executable(game_of_life src/main.c)

# --- OpenCL setup ---
if(APPLE)
    message(STATUS "Detected macOS — using Homebrew OpenCL headers and system framework")

    # Intel Macs (Homebrew installed under /usr/local)
    target_include_directories(game_of_life PRIVATE /usr/local/opt/opencl-headers/include)

    # Uncomment for Apple Silicon:
    # target_include_directories(game_of_life PRIVATE /opt/homebrew/opt/opencl-headers/include)

    # Link against macOS OpenCL framework
    target_link_options(game_of_life PRIVATE "-framework" "OpenCL")

else()
    # Linux or Windows — expect OpenCL via package manager or SDK
    find_package(OpenCL REQUIRED)
    target_link_libraries(game_of_life OpenCL::OpenCL)
endif()

# --- ncurses support ---
find_package(Curses REQUIRED)
include_directories(${CURSES_INCLUDE_DIR})
target_link_libraries(game_of_life ${CURSES_LIBRARIES})

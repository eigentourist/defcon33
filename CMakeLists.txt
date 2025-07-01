cmake_minimum_required(VERSION 3.10)
project(OpenCL_GameOfLife C)

add_executable(game_of_life src/main.c)

if(APPLE)
    message(STATUS "Detected macOS — using Homebrew OpenCL headers")

    # Homebrew location for OpenCL headers (Intel Macs)
    target_include_directories(game_of_life PRIVATE /usr/local/opt/opencl-headers/include)

    # (Optional: add this for Apple Silicon Homebrew if testing there later)
    # target_include_directories(game_of_life PRIVATE /opt/homebrew/opt/opencl-headers/include)

    # Link against system OpenCL framework
    target_link_options(game_of_life PRIVATE "-framework" "OpenCL")

else()
    # Linux or Windows
    find_package(OpenCL REQUIRED)
    target_link_libraries(game_of_life OpenCL::OpenCL)
endif()

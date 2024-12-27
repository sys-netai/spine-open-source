cmake_minimum_required (VERSION 3.20)
project (spine)
# generate compile_commands.json
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_compile_options(-std=c++17 -Wall -pedantic -Wextra -Weffc++ -g)

# settings
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_INCLUDE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/include)

# check c++ headers & generate config.h
include(CheckIncludeFileCXX)
CHECK_INCLUDE_FILE_CXX(experimental/filesystem HAVE_EXPERIMENTAL_FILESYSTEM)
CHECK_INCLUDE_FILE_CXX(filesystem HAVE_FILESYSTEM)
configure_file(config.h.in ${CMAKE_INCLUDE_OUTPUT_DIRECTORY}/config.h)

include(FetchContent)

FetchContent_Declare(json URL https://github.com/nlohmann/json/releases/download/v3.11.3/json.tar.xz)
FetchContent_MakeAvailable(json)

# include directory 
include_directories(./net ${CMAKE_INCLUDE_OUTPUT_DIRECTORY} ${json_SOURCE_DIR}/include)
add_subdirectory(net)

# target
add_executable(server server.cc)
add_executable(client_spine client_spine.cc)

# link libraries
target_link_libraries(server net pthread)
target_link_libraries(client_spine PRIVATE nlohmann_json::nlohmann_json net pthread stdc++fs)

# enable test
enable_testing()
add_subdirectory(test)

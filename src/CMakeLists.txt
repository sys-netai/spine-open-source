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

# check json 
set(JSON_DIR ${CMAKE_SOURCE_DIR}/../../third_party/json/)
set(nlohmann_json_DIR ${JSON_DIR}/build/)
find_package(nlohmann_json 3.2.0 REQUIRED)

# include directory 
include_directories(./net ${JSON_DIR}/single_include/nlohmann ${CMAKE_INCLUDE_OUTPUT_DIRECTORY})
add_subdirectory(net)

# target
add_executable(client client.cc) 
add_executable(server server.cc)
add_executable(passive_client passive_client.cc)
# test client for evaluation 
add_executable(client_eval client_eval.cc)
add_executable(client_spine client_spine.cc)

# link libraries
target_link_libraries(server net pthread)
target_link_libraries(client PRIVATE nlohmann_json::nlohmann_json net pthread stdc++fs)
target_link_libraries(passive_client PRIVATE nlohmann_json::nlohmann_json net pthread stdc++fs)
target_link_libraries(client_eval PRIVATE nlohmann_json::nlohmann_json net pthread stdc++fs)
target_link_libraries(client_spine PRIVATE nlohmann_json::nlohmann_json net pthread stdc++fs)

# enable test
enable_testing()
add_subdirectory(test)

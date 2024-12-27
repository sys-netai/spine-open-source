project(test)
cmake_minimum_required(VERSION 3.12)

# we don't need them here 
# set(JSON_DIR ${CMAKE_SOURCE_DIR}/../../third_party/json/)
# set(nlohmann_json_DIR ${JSON_DIR}/build/)
# find_package(nlohmann_json 3.2.0 REQUIRED )

add_executable(test_ipc test_ipc.cc)
add_executable(file_sender file_sender.cc)
add_executable(file_receiver file_receiver.cc)
add_executable(file_raw_sock_sender file_raw_sock_sender.cc)
target_link_libraries(test_ipc PRIVATE nlohmann_json::nlohmann_json net pthread stdc++fs)
target_link_libraries(file_sender PRIVATE nlohmann_json::nlohmann_json net pthread stdc++fs)
target_link_libraries(file_receiver net stdc++fs)
target_link_libraries(file_raw_sock_sender PRIVATE nlohmann_json::nlohmann_json net pthread stdc++fs)
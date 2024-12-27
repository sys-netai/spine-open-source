# add_compile_options(-fPIC -std=c++17 -Wall -pedantic -Wextra -Weffc++)

# aux_source_directory(. DIR_SOURCE)
file(GLOB LIB_SOURCE ./*.cc)
file(GLOB LIB_HEADERS ./*.hh)
# message("Source:" ${LIB_SOURCE})
add_library(net SHARED ${LIB_SOURCE} ${LIB_HEADERS})
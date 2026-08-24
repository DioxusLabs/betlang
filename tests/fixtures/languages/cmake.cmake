cmake_minimum_required(VERSION 3.16)

project(foo VERSION 1.2.0 LANGUAGES C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

add_library(foo src/foo.c src/util.c)
target_include_directories(foo PUBLIC include)

if(BUILD_TESTING)
  enable_testing()
  add_subdirectory(tests)
endif()

install(TARGETS foo DESTINATION lib)
install(DIRECTORY include/ DESTINATION include)

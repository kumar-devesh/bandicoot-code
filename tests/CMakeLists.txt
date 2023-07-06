# Armadillo is a dependency of the tests (since we compare with Armadillo's results).
if(${CMAKE_VERSION} VERSION_LESS 3.18)
  message(STATUS "*** WARNING: CMake < 3.18 cannot detect Armadillo versions configured without ARMA_USE_WRAPPER")
endif()

find_package(Armadillo REQUIRED)
include_directories(${ARMADILLO_INCLUDE_DIRS})

# The CMake documentation says this is a bad idea, and I don't care.
# Your build system should have a way to just include all the files in a directory as build dependencies.
# Note that my don't-care attitude here means that if you add a new test file, you really need to reconfigure with CMake altogether.
file(GLOB TEST_SOURCES "*.cpp")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
add_executable(bandicoot_test ${TEST_SOURCES})
target_link_libraries(bandicoot_test PRIVATE bandicoot ${ARMADILLO_LIBRARIES})
enable_testing()

add_test(NAME bandicoot_test COMMAND bandicoot_test)

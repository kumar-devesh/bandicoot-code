# The CMake documentation says this is a bad idea, and I don't care.
# Your build system should have a way to just include all the files in a directory as build dependencies.
# Note that my don't-care attitude here means that if you add a new test file, you really need to reconfigure with CMake altogether.
file(GLOB TEST_SOURCES "*.cpp")
add_executable(bandicoot_test ${TEST_SOURCES})
target_link_libraries(bandicoot_test PRIVATE bandicoot)
enable_testing()

add_test(NAME bandicoot_test COMMAND bandicoot_test)

enable_testing()

set(TEST_EXECUTABLE soft354_mpi_tests)
add_executable(${TEST_EXECUTABLE} GreyscaleFilterTest.cpp GaussianFilterTest.cpp)
target_link_libraries(${TEST_EXECUTABLE} ${LIBRARY})
target_link_libraries(${TEST_EXECUTABLE} GTest::GTest GTest::Main)
target_link_libraries(${TEST_EXECUTABLE} ${LODEPNG_LIB})
add_test(AllTests ${TEST_EXECUTABLE})

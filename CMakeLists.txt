cmake_minimum_required (VERSION 3.1)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
project (Naive_Bayes)
add_executable(Naive_Bayes main.cpp classifier.cpp)

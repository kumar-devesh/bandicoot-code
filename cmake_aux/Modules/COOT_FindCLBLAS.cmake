# - Find the clBLAS library and includes
# This module defines
#  CLBLAS_LIBRARIES, the libraries needed to use clBLAS
#  CLBLAS_INCLUDE_DIRS, any include directories needed to include clBLAS
#  CLBLAS_FOUND, If false, do not try to use clBLAS

if(DEFINED ENV{CLBLAS_ROOT} AND NOT DEFINED CLBLAS_ROOT)
  set(CLBLAS_ROOT $ENV{MKLROOT})
endif()

find_path(CLBLAS_INCLUDE_DIRS
  NAMES clBLAS.h
  HINTS ${CLBLAS_ROOT}
  PATH_SUFFIXES include include/x86_64 include/x64
  DOC "clBLAS.h main include header")

find_library(CLBLAS_LIBRARIES
  NAMES clBLAS
  HINTS ${CLBLAS_ROOT}
  PATH_SUFFIXES lib lib64 lib/x86_64 lib/x64 lib/x86 lib/import lib64/import lib/Win32
  DOC "clBLAS library")

if(CLBLAS_LIBRARIES AND CLBLAS_INCLUDE_DIRS)
  set(CLBLAS_FOUND "YES")
else()
  set(CLBLAS_FOUND "NO")
endif()

if(CLBLAS_FOUND)
  if(NOT CLBLAS_FIND_QUIETLY)
    message(STATUS "Found clBLAS: ${CLBLAS_LIBRARIES}")
  endif()
else()
  if(CLBLAS_FIND_REQUIRED)
    message(FATAL_ERROR "Could not find clBLAS!")
  endif()
endif()

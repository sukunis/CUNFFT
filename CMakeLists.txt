#test cmake version
cmake_minimum_required(VERSION 2.6.2)


SET(CMAKE_VERBOSE_MAKEFILE ON)
SET(PROJ_NAME "cunfft")
SET(PROJ_PATH ${CMAKE_SOURCE_DIR})
SET(PROJ_BIN_PATH ${CMAKE_BINARY_DIR})
SET(NFFT_SOURCE_DIR "/home/math/sukunis/NFFT_REPO/nfft/lib/branches/nfft-3.3/.libs")
SET(TexDocument_DIR "${CMAKE_CURRENT_LIST_DIR}/doc/Tutorial/")


##############################################################
# Set Project sources for compilation
##############################################################
SET(PROJ_SOURCE_DIR "${PROJ_PATH}/src")
SET(PROJ_SOURCES_CPP ${PROJ_PATH}/src/cuda_util.cpp ${PROJ_PATH}/src/c_util.cpp  ${PROJ_PATH}/src/simpleTest.cpp)
SET(PROJ_SOURCES_CU ${PROJ_PATH}/src/cunfft.cu ${PROJ_PATH}/src/cunfft_kernel.cu ${PROJ_PATH}/src/cunfft_wrapper.cu)
SET(PROJ_SOURCES ${PROJ_SOURCES_CPP} ${PROJ_SOURCES_CU})

# Set Project Header for compilation


FILE(GLOB_RECURSE PROJ_HEADER_H src/*.h)
FILE(GLOB_RECURSE PROJ_HEADER_CUH src/*.cuh)
SET(PROJ_HEADERS ${PROJ_HEADER_H} ${PROJ_HEADER_CUH})


SET(PROJ_LIBRARY_DIR "${PROJ_PATH}/lib") 
SET(PROJ_INCLUDE_DIR "${PROJ_PATH}/src")
SET(PROJ_EXAMPLE_DIR "${PROJ_PATH}/src")
SET(PROJ_EXAMPLE_FILE "simpleTest")

PROJECT(${PROJ_NAME})
message("\nProject: ${PROJ_NAME}")

message("-- INFO: proj_path: ${PROJ_PATH}")
message("-- INFO: proj_lib_dir: ${PROJ_LIBRARY_DIR}")
message("-- INFO: proj_inc_dir: ${PROJ_INCLUDE_DIR}")
message("-- INFO: proj_example_dir: ${PROJ_EXAMPLE_DIR}")
message("-- INFO: proj_src_dir: ${PROJ_SOURCE_DIR}")
message("-- INFO: proj_out_path: ${PROJ_BIN_PATH}\n")
message("-- INFO: project sources: ${PROJ_SOURCES}")


# locate command available
#TODO ungewollter output
EXECUTE_PROCESS(COMMAND which locate OUTPUT_VARIABLE FILE_LIST)
if(FILE_LIST)
	message(STATUS "CHECK bash command: locate -- FOUND\n")
	SET(USE_LOCATE 1)
else(FILE_LIST)
	message(STATUS "CHECK bash command: locate -- NOT FOUND!\n")
	SET(USE_LOCATE 0)
endif(FILE_LIST)

# Version number
set(cunfft_VERSION_MAJOR 1)
set(cunfft_VERSION_MINOR 0)
set(cunfft_BUILD_VERSION 0)
SET(cunfft_VERSION "${cunfft_VERSION_MAJOR}.${cunfft_VERSION_MINOR}.${cunfft_BUILD_VERSION}" )

# modul path findcuda
set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/CMakeModule/cuda" ${CMAKE_MODULE_PATH})

message(STATUS "CHECK cuda installation, set paths...")
find_package(CUDA 1.3 REQUIRED)
#find_package(CUDA QUIET REQUIRED)
# path setting by findCUDA
message(STATUS "SET PATH CUDA_SDK_ROOT_DIR = ${CUDA_SDK_ROOT_DIR}")
message(STATUS "SET PATH CUDA_TOOLKIT_ROOT_DIR = ${CUDA_TOOLKIT_ROOT_DIR}")
message(STATUS "SET PATH CUDA_NVCC_EXECUTABLE = ${CUDA_NVCC_EXECUTABLE}")
message(STATUS "SET PATH CUDA_TOOLKIT_INCLUDE = ${CUDA_TOOLKIT_INCLUDE}\n")
message(STATUS "SET PATH CUDA_CUDA_LIBRARY = ${CUDA_CUDA_LIBRARY}")
message(STATUS "SET PATH CUDA_CUDART_LIBRARY = ${CUDA_CUDART_LIBRARY}")
message(STATUS "SET PATH CUDA_cufft_LIBRARY = ${CUDA_cufft_LIBRARY}")
message(STATUS "SET PATH CUDA_CUT_LIBRARY = ${CUDA_CUT_LIBRARIES}\n")

#############################################################
# test if all paths and libs found
#############################################################
if(NOT CUDA_SDK_ROOT_DIR)
	LIST(APPEND MISSING_PATHS "CUDA_SDK_ROOT_DIR")
endif()
if(NOT CUDA_TOOLKIT_ROOT_DIR)
	LIST(APPEND MISSING_PATHS "CUDA_TOOLKIT_ROOT_DIR")
endif()
if(NOT CUDA_NVCC_EXECUTABLE)
	LIST(APPEND MISSING_PATHS "CUDA_NVCC_EXECUTABLE")
endif()
if(NOT CUDA_TOOLKIT_INCLUDE)
	LIST(APPEND MISSING_PATHS "CUDA_TOOLKIT_INCLUDE")
endif()
if(NOT CUDA_CUDA_LIBRARY)
	LIST(APPEND MISSING_PATHS "CUDA_CUDA_LIBRARY")
endif()
if(NOT CUDA_CUDART_LIBRARY)
	LIST(APPEND MISSING_PATHS "CUDA_CUDART_LIBRARY")
endif()
if(NOT CUDA_cufft_LIBRARY)
	LIST(APPEND MISSING_PATHS "CUDA_cufft_LIBRARY")
endif()
if(NOT CUDA_CUT_LIBRARIES)
	LIST(APPEND MISSING_PATHS "CUDA_CUT_LIBRARIES")
endif()

if (CUDA_FOUND)
	list(LENGTH MISSING_PATHS LSIZE)
	
	if(LSIZE GREATER 0)
		message("ERROR CHECK cuda installation -- NOT COMPLETE!")
		message("ERROR: Can't set paths of cuda installation: ${MISSING_PATHS}\n")
		LOCATEPATH("${MISSING_PATHS}")
    	message(FATAL_ERROR "")
	else()
	   message(STATUS "FINISHED cuda installation -- OK\n")
	endif()
else()
	message(FATAL_ERROR "ERROR cuda installation or paths -- NOT FOUND\n")
endif()

#############################################################
# check compiler version
#############################################################

IF(CMAKE_COMPILER_IS_GNUCC)
    # get version of compiler
    EXECUTE_PROCESS(COMMAND ${CMAKE_C_COMPILER} -dumpversion
      OUTPUT_VARIABLE GCC_VERSION)
    IF(GCC_VERSION VERSION_LESS 4.5)
      # ramc: CUDA v4.0 RC2 is incompatible with gcc >= 4.5. Gcc
      # 4.4. is selected in the root gerardus CMakeLists.txt, but this
      # flag needs to be set by hand, because otherwise nvcc will read
      # the version of the default gcc in the system, that is v4.5
      SET(CUDA_NVCC_FLAGS "-D__GNUC_MINOR__=4")
      #message(STATUS "GCC Version <4.5")
    #else()
      #message(STATUS "GCC Version >=4.5")
    ENDIF(GCC_VERSION VERSION_LESS 4.5)
  ENDIF(CMAKE_COMPILER_IS_GNUCC)
message(STATUS "INFO: GCC COMPILER VERSION: ${GCC_VERSION}")



#NVCC FLAGS setzen
#The output of nvlink is then linked together with all the host objects by the host linker to form the final executable. The generation of relocatable vs executable device code is controlled by --relocatable-device-code={true,false} option
#set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-dc)
#set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-rdc=true)

#set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-dlink)



#############################################################
# definitions config.h
#############################################################
set(THREAD_DIM_X "16" CACHE STRING "Maximum numbers of threads per block in x direction for 2D launch grid")
set(THREAD_DIM_Y "16" CACHE STRING "Maximum numbers of threads per block in y direction for 2D launch grid")
set(THREAD_DIM "512" CACHE STRING "Maximum numbers of threads per block in x direction for 1D launch grid")

#depending on graphic card
set(MAX_NUM_THREADS "1024" CACHE STRING "Maximum numbers of threads per block")
set(MAX_BLOCK_DIM_X "1024" CACHE STRING "Maximum numbers of threads per block in x direction")
set(MAX_BLOCK_DIM_Y "1024" CACHE STRING "Maximum numbers of threads per block in y direction")
set(MAX_BLOCK_DIM_Z "64" CACHE STRING "Maximum numbers of threads per block in z direction")
set(MAX_GRID_DIM_X "65535" CACHE STRING "Maximum numbers of blocks per grid in x direction")
set(MAX_GRID_DIM_Y "65535" CACHE STRING "Maximum numbers of blocks per grid in y direction")
set(MAX_GRID_DIM_Z "1" CACHE STRING "Maximum numbers of blocks per grid in z direction")

set(CUT_OFF "6" CACHE STRING "cut-off parameter m ")

OPTION(LARGE_INPUT "use LONG INT instead of INT" ON)
OPTION(CUNFFT_DOUBLE_PRECISION "Enable double precision computation" ON)
OPTION(COM_FG_PSI "Enable fast gaussian gridding" OFF )
OPTION(MILLI_SEC "Time to stdout in ms" ON )
OPTION(MEASURED_TIMES "Enable measure time for several steps" OFF)
OPTION(PRINT_CONFIG "Launch configuration to stdout" OFF)
OPTION(DEBUG "Enable verification of cuda function calls" OFF) 

message(STATUS "WRITE definitions to config.h:")
message(STATUS "INFO config.h: CUT_OFF: " @CUT_OFF@)
message(STATUS "INFO config.h: LARGE_INPUT: " @LARGE_INPUT@)
message(STATUS "INFO config.h: CUNFFT_DOUBLE_PRECISION: " @CUNFFT_DOUBLE_PRECISION@)
message(STATUS "INFO config.h: COM_FG_PSI: " @COM_FG_PSI@)
message(STATUS "INFO config.h:  MILLI_SEC: " @MILLI_SEC@)
message(STATUS "INFO config.h: MEASURED_TIMES: " @MEASURED_TIMES@)
message(STATUS "INFO config.h: PRINT_CONFIG: " @PRINT_CONFIG@)
message(STATUS "INFO config.h: DEBUG: " @DEBUG@)

message(STATUS "INFO config.h: THREAD_DIM_X: " @THREAD_DIM_X@)
message(STATUS "INFO config.h: THREAD_DIM_Y: " @THREAD_DIM_Y@)
message(STATUS "INFO config.h: THREAD_DIM: " @THREAD_DIM@)
message(STATUS "INFO config.h: MAX_NUM_THREADS: " @MAX_NUM_THREADS@)
message(STATUS "INFO config.h: MAX_BLOCK_DIM_X: " @MAX_BLOCK_DIM_X@)
message(STATUS "INFO config.h: MAX_BLOCK_DIM_Y: " @MAX_BLOCK_DIM_Y@)
message(STATUS "INFO config.h: MAX_BLOCK_DIM_Z: " @MAX_BLOCK_DIM_Z@)
message(STATUS "INFO config.h: MAX_GRID_DIM_X: " @MAX_GRID_DIM_X@)
message(STATUS "INFO config.h: MAX_GRID_DIM_Y: " @MAX_GRID_DIM_Y@)
message(STATUS "INFO config.h: MAX_GRID_DIM_Z: " @MAX_GRID_DIM_Z@)

message(STATUS "WRITE config.h to ${PROJ_INCLUDE_DIR}\n")
configure_file(${PROJ_INCLUDE_DIR}/config.h.in ${PROJ_INCLUDE_DIR}/config.h)

#############################################################
#Adding src directory to the project
#############################################################
ADD_SUBDIRECTORY(src)

#############################################################
# add custom targets
#############################################################
ADD_CUSTOM_TARGET (distclean @echo cleaning for source distribution)
  SET(DISTCLEANED
   CMakeCache.txt
   *.cmake
   Makefile
   core core.*
   install_manifest.txt
   ${PROJ_EXAMPLE_DIR}/${PROJ_EXAMPLE_FILE}
  )
  
  ADD_CUSTOM_COMMAND(
    DEPENDS clean
    COMMENT "distribution clean"
    COMMAND rm
    ARGS    -Rf src CMakeFiles ${PROJ_LIBRARY_DIR} ${DISTCLEANED} Doxyfile
    TARGET  distclean
  )
  
  
# add a target to generate API documentation with Doxygen
find_package(Doxygen)
if(DOXYGEN_FOUND)
#configure_file(${CMAKE_CURRENT_SOURCE_DIR}/Doxyfile.in ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile @ONLY)  
configure_file(${CMAKE_CURRENT_LIST_DIR}/doc/Doxyfile.in ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile @ONLY)
add_custom_target(doc
	${DOXYGEN_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile
	WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
	COMMENT "Generating API documentation with Doxygen" VERBATIM
)


#find_package(LATEX)
#if(PDFLATEX_COMPILER)
#message(STATUS "PDF Compiler command: " @PDFLATEX_COMPILER@)
#    ADD_CUSTOM_COMMAND( 
#      #OUTPUT    ${TexDocument_DIR}/cunfftDoc.pdf
#      COMMAND   ${PDFLATEX_COMPILER}
#      ARGS      ${TexDocument_DIR}/cunfftDoc.tex
#      DEPENDS   ${TexDocument_DIR}/tree.tex
#      COMMENT   "Generating Tutorial with pdflatex"
#      TARGET	doc
#    )
#endif(PDFLATEX_COMPILER)
endif(DOXYGEN_FOUND)


#build a cpack driven installer package
#include(InstallRequiredSystemLibraries)
#set(CPACK_RESOURCE_FILE_LICENSE "${CMAKE_CURRENT_SOURCE_DIR}/License.txt")
#set(CPACK_PACKAGE_VERSION_MAJOR "${cunfft_VERSION_MAJOR}")
#set(CPACK_PACKAGE_VERSION_MINOR "${cunfft_VERSION_MINOR}")
#include(CPack)



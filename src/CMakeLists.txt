

#INCLUDE_DIRECTORIES("/home/math/sukunis/NFFT_REPO/nfft/lib/branches/nfft-3.3/include")
#LINK_DIRECTORIES("/home/math/sukunis/NFFT_REPO/nfft/lib/branches/nfft-3.3/.libs")


# check included c header can be found
FIND_FILE(STD_FILE stdio.h)
message(STATUS "CHECK PATH stdio.h: ${STD_FILE}")
FIND_FILE(COMPLEX_FILE complex.h)
message(STATUS "CHECK PATH complex.h: ${COMPLEX_FILE}")
FIND_FILE(TIME_FILE time.h)
message(STATUS "CHECK PATH time.h: ${TIME_FILE}")
FIND_FILE(MATH_FILE math.h)
message(STATUS "CHECK PATH math.h: ${MATH_FILE}")
FIND_FILE(STDLIB_FILE stdlib.h)
message(STATUS "CHECK PATH stdlib.h: ${STDLIB_FILE}\n")



# src inc header 
INCLUDE_DIRECTORIES(${PROJ_INCLUDE_DIR})
#CUDA_INCLUDE_DIRECTORIES("/usr/local/cuda-5.5/samples/common/inc/")

# check included cuda header can be found
set(CUFFT_INC_DIR "${CUFFT_INC_DIR}-NOTFOUND")
# cufft.h
FIND_PATH(CUFFT_INC_DIR 
	NAMES cufft.h 
	PATHS  
		${CUDA_SDK_ROOT_DIR}
	 	${CUDA_SDK_ROOT_DIR}/C/common/inc
		${CUDA_SDK_ROOT_DIR}/CUDALibraries/common/inc
    	DOC "cufft header location."
     )

if(CUFFT_INC_DIR)
	message(STATUS "CHECK PATH cufft.h: ${CUFFT_INC_DIR}/cufft.h")
else(CUFFT_INC_DIR)
	EXECUTE_PROCESS(COMMAND locate cufft.h OUTPUT_VARIABLE FILE_LIST )
	FIND_PATH(CUFFT_INC_DIR 
	NAMES cufft.h 
	PATHS  
		${FILE_LIST}/..
    	DOC "cufft header location."
     )
	message(STATUS "CHECK PATH cufft.h: ${CUFFT_INC_DIR}/cufft.h\n")
endif(CUFFT_INC_DIR)


# checking cuda libs can be found
if(CUDA_CUDA_LIBRARY)
	message(STATUS "CHECK PATH cuda library: ${CUDA_CUDA_LIBRARY}")
else(CUDA_CUDA_LIBRARY)
	LOCATE(libcuda libcuda. CUDA_CUDA_LIBRARY)
endif(CUDA_CUDA_LIBRARY)

if(CUDA_CUDART_LIBRARY)
	message(STATUS "CHECK PATH cudart library: ${CUDA_CUDART_LIBRARY}")
else(CUDA_CUDART_LIBRARY)
	LOCATE(libcudart libcudart. CUDA_CUDART_LIBRARY)
endif(CUDA_CUDART_LIBRARY)

if(CUDA_cufft_LIBRARY)
	message(STATUS "CHECK PATH cufft library: ${CUDA_cufft_LIBRARY}\n")
else(CUDA_cufft_LIBRARY)
	LOCATE(libcufft libcufft. CUDA_cufft_LIBRARY)
endif(CUDA_cufft_LIBRARY)


CUDA_ADD_LIBRARY(${PROJ_NAME} 
	SHARED 
	${PROJ_SOURCES} 
	${PROJ_HEADERS} 
	OPTIONS -arch sm_21 #--ptxas-options=-v 
	)

#target_link_libraries(${PROJ_NAME} nfft3 )

#Linking with libcufft (expected to be in 
CUDA_ADD_CUFFT_TO_TARGET(${PROJ_NAME})

SET_TARGET_PROPERTIES(${PROJ_NAME} 
	PROPERTIES
	VERSION ${cunfft_VERSION}
  	SOVERSION ${cunfft_VERSION_MAJOR}.${cunfft_VERSION_MINOR}
)

#create executable demo
CUDA_ADD_EXECUTABLE(${PROJ_EXAMPLE_FILE} ${PROJ_EXAMPLE_DIR}/${PROJ_EXAMPLE_FILE}.cpp)
TARGET_LINK_LIBRARIES(${PROJ_EXAMPLE_FILE} ${PROJ_NAME})



CUDA_BUILD_CLEAN_TARGET()


################################################################
## installation of library
################################################################

INSTALL(PROGRAMS ${CMAKE_CURRENT_BINARY_DIR}/${PROJ_EXAMPLE_FILE} DESTINATION ${PROJ_EXAMPLE_DIR})
INSTALL(TARGETS ${PROJ_NAME} LIBRARY DESTINATION ${PROJ_LIBRARY_DIR})






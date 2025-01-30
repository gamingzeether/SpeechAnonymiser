cmake_minimum_required(VERSION 3.16.0 FATAL_ERROR)
set(CMAKE_CXX_STANDARD 20)
set(CXX_STANDARD 20)
set(VCPKG_POLICY_SKIP_ABSOLUTE_PATHS_CHECK enabled)

if (POLICY CMP0141)
  cmake_policy(SET CMP0141 NEW)
  set(CMAKE_MSVC_DEBUG_INFORMATION_FORMAT "$<IF:$<AND:$<C_COMPILER_ID:MSVC>,$<CXX_COMPILER_ID:MSVC>>,$<$<CONFIG:Debug,RelWithDebInfo>:EditAndContinue>,$<$<CONFIG:Debug,RelWithDebInfo>:ProgramDatabase>>")
endif()

option(ASAN "Compile with Address Sanitizer." OFF)
option(OMP "Compile with OpenMP." OFF)
option(AUDIO "Compile with audio in/output." ON)
option(MKL "Compile with Intel MKL." OFF)
option(LTO "Compile with LTO on release builds." ON)
if (ASAN)
    add_compile_options(-fsanitize=address)
    add_link_options(-fsanitize=address)
endif()
if (AUDIO)
    list(APPEND VCPKG_MANIFEST_FEATURES "audio")
    add_definitions(-DAUDIO=${AUDIO})
endif()
if (MKL)
    list(APPEND VCPKG_MANIFEST_FEATURES "mkl")
endif()

set(CMAKE_TOOLCHAIN_FILE $ENV{VCPKG_ROOT}/scripts/buildsystems/vcpkg.cmake)

project(SpeechAnonymiser CXX)

################################################################################
# Set target arch type if empty. Visual studio solution generator provides it.
################################################################################
if (MSVC)
    if(NOT CMAKE_VS_PLATFORM_NAME)
        set(CMAKE_VS_PLATFORM_NAME "x64")
    endif()
    message("${CMAKE_VS_PLATFORM_NAME} architecture in use")
    
    if(NOT ("${CMAKE_VS_PLATFORM_NAME}" STREQUAL "x64"
        OR "${CMAKE_VS_PLATFORM_NAME}" STREQUAL "x86"))
        message(FATAL_ERROR "${CMAKE_VS_PLATFORM_NAME} arch is not supported!")
    endif()
else()
    message("${CMAKE_SYSTEM_PROCESSOR} architecture in use")
endif()

################################################################################
# Global configuration types
################################################################################
set(CMAKE_CONFIGURATION_TYPES
    "Debug"
    "Release"
    CACHE STRING "" FORCE
)

################################################################################
# Global compiler options
################################################################################
if(MSVC)
    # remove default flags provided with CMake for MSVC
    set(CMAKE_CXX_FLAGS "")
    set(CMAKE_CXX_FLAGS_DEBUG "")
    set(CMAKE_CXX_FLAGS_RELEASE "")
endif()

################################################################################
# Global linker options
################################################################################
if(MSVC)
    # remove default flags provided with CMake for MSVC
    set(CMAKE_EXE_LINKER_FLAGS "")
    set(CMAKE_MODULE_LINKER_FLAGS "")
    set(CMAKE_SHARED_LINKER_FLAGS "")
    set(CMAKE_STATIC_LINKER_FLAGS "")
    set(CMAKE_EXE_LINKER_FLAGS_DEBUG "${CMAKE_EXE_LINKER_FLAGS}")
    set(CMAKE_MODULE_LINKER_FLAGS_DEBUG "${CMAKE_MODULE_LINKER_FLAGS}")
    set(CMAKE_SHARED_LINKER_FLAGS_DEBUG "${CMAKE_SHARED_LINKER_FLAGS}")
    set(CMAKE_STATIC_LINKER_FLAGS_DEBUG "${CMAKE_STATIC_LINKER_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS_RELEASE "${CMAKE_EXE_LINKER_FLAGS}")
    set(CMAKE_MODULE_LINKER_FLAGS_RELEASE "${CMAKE_MODULE_LINKER_FLAGS}")
    set(CMAKE_SHARED_LINKER_FLAGS_RELEASE "${CMAKE_SHARED_LINKER_FLAGS}")
    set(CMAKE_STATIC_LINKER_FLAGS_RELEASE "${CMAKE_STATIC_LINKER_FLAGS}")
endif()

################################################################################
# Nuget packages function stub.
################################################################################
function(use_package TARGET PACKAGE VERSION)
    message(WARNING "No implementation of use_package. Create yours. "
                    "Package \"${PACKAGE}\" with version \"${VERSION}\" "
                    "for target \"${TARGET}\" is ignored!")
endfunction()

################################################################################
# Common utils
################################################################################
include(CMake/Utils.cmake)

################################################################################
# Additional Global Settings(add specific info there)
################################################################################
include(CMake/GlobalSettingsInclude.cmake OPTIONAL)

################################################################################
# Use solution folders feature
################################################################################
set_property(GLOBAL PROPERTY USE_FOLDERS ON)

################################################################################
# Sub-projects
################################################################################

set(PROJECT_NAME SpeechAnonymiser)

################################################################################
# Source groups
################################################################################
set (PROJECT_SOURCE_DIR ${CMAKE_SOURCE_DIR}/src)

set(Headers
    "${PROJECT_SOURCE_DIR}/common_inc.h"
    "${PROJECT_SOURCE_DIR}/define.h"
    "${PROJECT_SOURCE_DIR}/structs.h"
    "${PROJECT_SOURCE_DIR}/TSVReader.h"
    "${PROJECT_SOURCE_DIR}/Visualizer.h"
    "${PROJECT_SOURCE_DIR}/ModelSerializer.h"
    "${PROJECT_SOURCE_DIR}/PhonemeClassifier.h"
    "${PROJECT_SOURCE_DIR}/JSONHelper.h"
    "${PROJECT_SOURCE_DIR}/Dataset.h"
    "${PROJECT_SOURCE_DIR}/ClassifierHelper.h"
    "${PROJECT_SOURCE_DIR}/Logger.h"
    "${PROJECT_SOURCE_DIR}/SpeechEngine.h"
    "${PROJECT_SOURCE_DIR}/SpeechEngineArticulator.h"
    "${PROJECT_SOURCE_DIR}/SpeechEngineConcatenator.h"
    "${PROJECT_SOURCE_DIR}/SpeechEngineFormant.h"
    "${PROJECT_SOURCE_DIR}/Animator.h"
    "${PROJECT_SOURCE_DIR}/include_mlpack.h"
    "${PROJECT_SOURCE_DIR}/PhonemeModel.h"
    "${PROJECT_SOURCE_DIR}/Voicebank.h"
    "${PROJECT_SOURCE_DIR}/Config.h"
    "${PROJECT_SOURCE_DIR}/PhonemeSet.h"
    "${PROJECT_SOURCE_DIR}/Global.h"
    "${PROJECT_SOURCE_DIR}/Util.h"
    "${PROJECT_SOURCE_DIR}/ParallelWorker.h"
    "${PROJECT_SOURCE_DIR}/TimitIterator.h"
)
source_group("Headers" FILES ${Headers})

set(Sources
    "${PROJECT_SOURCE_DIR}/SpeechAnonymiser.cpp"
    "${PROJECT_SOURCE_DIR}/TSVReader.cpp"
    "${PROJECT_SOURCE_DIR}/Visualizer.cpp"
    "${PROJECT_SOURCE_DIR}/ModelSerializer.cpp"
    "${PROJECT_SOURCE_DIR}/PhonemeClassifier.cpp"
    "${PROJECT_SOURCE_DIR}/JSONHelper.cpp"
    "${PROJECT_SOURCE_DIR}/Dataset.cpp"
    "${PROJECT_SOURCE_DIR}/ClassifierHelper.cpp"
    "${PROJECT_SOURCE_DIR}/Logger.cpp"
    "${PROJECT_SOURCE_DIR}/SpeechEngineArticulator.cpp"
    "${PROJECT_SOURCE_DIR}/SpeechEngineConcatenator.cpp"
    "${PROJECT_SOURCE_DIR}/SpeechEngineFormant.cpp"
    "${PROJECT_SOURCE_DIR}/Animator.cpp"
    "${PROJECT_SOURCE_DIR}/PhonemeModel.cpp"
    "${PROJECT_SOURCE_DIR}/Voicebank.cpp"
    "${PROJECT_SOURCE_DIR}/Config.cpp"
    "${PROJECT_SOURCE_DIR}/PhonemeSet.cpp"
    "${PROJECT_SOURCE_DIR}/Global.cpp"
    "${PROJECT_SOURCE_DIR}/Util.cpp"
    "${PROJECT_SOURCE_DIR}/TimitIterator.cpp"
)
source_group("Sources" FILES ${Sources})

set(ALL_FILES
    ${Headers}
    ${Sources}
)

################################################################################
# Target
################################################################################

add_executable(${PROJECT_NAME} ${ALL_FILES})

if (NOT MSVC)
    target_link_libraries(${PROJECT_NAME} PRIVATE stdc++fs)
endif()

if (AUDIO)
    find_package(RtAudio CONFIG REQUIRED)
    target_link_libraries(${PROJECT_NAME} PRIVATE RtAudio::rtaudio)
endif()

if (MKL)
    find_package(MKL CONFIG REQUIRED)
    target_link_libraries(${PROJECT_NAME} PRIVATE MKL::MKL)
endif()

find_package(Cargs CONFIG REQUIRED)
target_link_libraries(${PROJECT_NAME} PRIVATE cargs)

find_path(DRLIBS_INCLUDE_DIRS "dr_mp3.h")
find_path(DRLIBS_INCLUDE_DIRS "dr_wav.h")
target_include_directories(${PROJECT_NAME} PRIVATE ${DRLIBS_INCLUDE_DIRS})

find_package(FFTW3 CONFIG REQUIRED)
find_package(FFTW3f CONFIG REQUIRED)
target_link_libraries(${PROJECT_NAME} PRIVATE FFTW3::fftw3)
target_link_libraries(${PROJECT_NAME} PRIVATE FFTW3::fftw3f)

find_package(glfw3 CONFIG REQUIRED)
target_link_libraries(${PROJECT_NAME} PRIVATE glfw)

find_package(glm CONFIG REQUIRED)
target_link_libraries(${PROJECT_NAME} PRIVATE glm::glm)

find_package(SampleRate CONFIG REQUIRED)
target_link_libraries(${PROJECT_NAME} PRIVATE SampleRate::samplerate)

find_package(Bandicoot CONFIG REQUIRED)
target_link_libraries(${PROJECT_NAME} PRIVATE bandicoot)

find_package(Armadillo CONFIG REQUIRED)
target_link_libraries(${PROJECT_NAME} PRIVATE armadillo)

find_path(MLPACK_INCLUDE_DIRS "mlpack.hpp")
target_include_directories(${PROJECT_NAME} PRIVATE ${MLPACK_INCLUDE_DIRS})

find_package(Vulkan REQUIRED)
target_link_libraries(${PROJECT_NAME} PRIVATE Vulkan::Vulkan)

find_package(yyjson CONFIG REQUIRED)
target_link_libraries(${PROJECT_NAME} PRIVATE yyjson::yyjson)

find_package(libzip CONFIG REQUIRED)
target_link_libraries(${PROJECT_NAME} PRIVATE libzip::zip)

use_props(${PROJECT_NAME} "${CMAKE_CONFIGURATION_TYPES}" "${DEFAULT_CXX_PROPS}")
set(ROOT_NAMESPACE SpeechAnonymiser)

set_target_properties(${PROJECT_NAME} PROPERTIES
    VS_GLOBAL_KEYWORD "Win32Proj"
)
if("${CMAKE_VS_PLATFORM_NAME}" STREQUAL "x64")
    set_target_properties(${PROJECT_NAME} PROPERTIES
        INTERPROCEDURAL_OPTIMIZATION_RELEASE "TRUE"
    )
elseif("${CMAKE_VS_PLATFORM_NAME}" STREQUAL "x86")
    set_target_properties(${PROJECT_NAME} PROPERTIES
        INTERPROCEDURAL_OPTIMIZATION_RELEASE "TRUE"
    )
endif()
################################################################################
# Compile definitions
################################################################################
if("${CMAKE_VS_PLATFORM_NAME}" STREQUAL "x64")
    target_compile_definitions(${PROJECT_NAME} PRIVATE
        "$<$<CONFIG:Debug>:"
            "_DEBUG"
        ">"
        "$<$<CONFIG:Release>:"
            "NDEBUG"
        ">"
        "_CRT_SECURE_NO_WARNINGS;"
        "_CONSOLE;"
        "UNICODE;"
        "_UNICODE"
    )
elseif("${CMAKE_VS_PLATFORM_NAME}" STREQUAL "x86")
    target_compile_definitions(${PROJECT_NAME} PRIVATE
        "$<$<CONFIG:Debug>:"
            "_DEBUG"
        ">"
        "$<$<CONFIG:Release>:"
            "NDEBUG"
        ">"
        "WIN32;"
        "_CONSOLE;"
        "UNICODE;"
        "_UNICODE"
    )
else()
    target_compile_definitions(${PROJECT_NAME} PRIVATE
        "$<$<CONFIG:Debug>:"
            "_DEBUG"
        ">"
        "$<$<CONFIG:Release>:"
            "NDEBUG"
        ">"
        "UNICODE;"
        "_UNICODE"
    )
endif()

################################################################################
# Compile and link options
################################################################################
if(MSVC)
    if("${CMAKE_VS_PLATFORM_NAME}" STREQUAL "x64")
        target_compile_options(${PROJECT_NAME} PRIVATE
            $<$<CONFIG:Debug>:
                /Od;
                /Oi-
            >
            $<$<CONFIG:Release>:
                /Oi;
                /Gy
            >
            /permissive-;
            /sdl;
            /W3;
            /bigobj;
            ${DEFAULT_CXX_DEBUG_INFORMATION_FORMAT};
            ${DEFAULT_CXX_EXCEPTION_HANDLING};
        )
    elseif("${CMAKE_VS_PLATFORM_NAME}" STREQUAL "x86")
        target_compile_options(${PROJECT_NAME} PRIVATE
            $<$<CONFIG:Release>:
                /Oi;
                /Gy;
            >
            /permissive-;
            /sdl;
            /W3;
            /bigobj;
            ${DEFAULT_CXX_DEBUG_INFORMATION_FORMAT};
            ${DEFAULT_CXX_EXCEPTION_HANDLING}
        )
    endif()
    if("${CMAKE_VS_PLATFORM_NAME}" STREQUAL "x64")
        target_link_options(${PROJECT_NAME} PRIVATE
            $<$<CONFIG:Release>:
                /OPT:REF;
                /OPT:ICF
            >
            /DEBUG;
            /SUBSYSTEM:CONSOLE
        )
    elseif("${CMAKE_VS_PLATFORM_NAME}" STREQUAL "x86")
        target_link_options(${PROJECT_NAME} PRIVATE
            $<$<CONFIG:Release>:
                /OPT:REF;
                /OPT:ICF
            >
            /DEBUG;
            /SUBSYSTEM:CONSOLE
        )
    endif()
else()
    target_compile_options(${PROJECT_NAME} PRIVATE
        $<$<CONFIG:Debug>:
            -O0
        >
        $<$<CONFIG:Release>:
            -ffunction-sections
            -O3
            -fno-signed-zeros
            -funsafe-math-optimizations
            -fno-trapping-math
        >
        -std=c++2a
        -march=native
        ${DEFAULT_CXX_DEBUG_INFORMATION_FORMAT}
        ${DEFAULT_CXX_EXCEPTION_HANDLING}
    )
    if (OMP)
        find_package(OpenMP REQUIRED)
        target_link_libraries(${PROJECT_NAME} PRIVATE OpenMP::OpenMP_CXX)
        target_compile_options(${PROJECT_NAME} PRIVATE
            -fopenmp
        )
    endif()
    if (LTO AND (CMAKE_BUILD_TYPE STREQUAL "Release"))
        target_compile_options(${PROJECT_NAME} PRIVATE
            -flto=auto
        )
    endif()
endif()

add_custom_command(
    TARGET ${PROJECT_NAME} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory "${CMAKE_SOURCE_DIR}/shaders" "$<TARGET_FILE_DIR:${PROJECT_NAME}>/shaders"
    COMMAND ${CMAKE_COMMAND} -E copy_directory "${CMAKE_SOURCE_DIR}/configs" "$<TARGET_FILE_DIR:${PROJECT_NAME}>/configs"
)

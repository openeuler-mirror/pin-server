cmake_minimum_required(VERSION 3.5.1)

set (CMAKE_CXX_STANDARD 17)

find_package(Threads REQUIRED)

macro(_CHECK)
    if (${ARGV0} STREQUAL "${ARGV1}")
        message("${BoldRed}ERROR: can not find " ${ARGV2} " program${ColourReset}")
        set(CHECKER_RESULT 1)
    else()
        message("--  found " ${ARGV2} " --- works")
    endif()
endmacro()

find_package(PkgConfig REQUIRED)

# check protobuf
pkg_check_modules(PC_PROTOBUF "protobuf>=3.1.0")
find_library(PROTOBUF_LIBRARY protobuf
    HINTS ${PC_PROTOBUF_LIBDIR} ${PC_PROTOBUF_LIBRARY_DIRS})
_CHECK(PROTOBUF_LIBRARY "PROTOBUF_LIBRARY-NOTFOUND" "libprotobuf.so")

find_program(CMD_PROTOC protoc)
_CHECK(CMD_PROTOC "CMD_PROTOC-NOTFOUND" "protoc")
find_program(CMD_GRPC_CPP_PLUGIN grpc_cpp_plugin)
_CHECK(CMD_GRPC_CPP_PLUGIN "CMD_GRPC_CPP_PLUGIN-NOTFOUND" "grpc_cpp_plugin")

# check grpc
find_path(GRPC_INCLUDE_DIR grpc/grpc.h)
_CHECK(GRPC_INCLUDE_DIR "GRPC_INCLUDE_DIR-NOTFOUND" "grpc/grpc.h")
find_library(GRPC_PP_REFLECTION_LIBRARY grpc++_reflection)
_CHECK(GRPC_PP_REFLECTION_LIBRARY "GRPC_PP_REFLECTION_LIBRARY-NOTFOUND" "libgrpc++_reflection.so")
find_library(GRPC_PP_LIBRARY grpc++)
_CHECK(GRPC_PP_LIBRARY "GRPC_PP_LIBRARY-NOTFOUND" "libgrpc++.so")
find_library(GRPC_LIBRARY grpc)
_CHECK(GRPC_LIBRARY "GRPC_LIBRARY-NOTFOUND" "libgrpc.so")
find_library(GPR_LIBRARY gpr)
_CHECK(GPR_LIBRARY "GPR_LIBRARY-NOTFOUND" "libgpr.so")
    
# check abseil_synchronization
find_library(ABSEIL_SYNC_LIBRARY absl_synchronization)
_CHECK(ABSEIL_SYNC_LIBRARY "ABSEIL_SYNC_LIBRARY-NOTFOUND" "libabsl_synchronization.so")
find_library(ABSEIL_CORD_LIBRARY absl_cord)
_CHECK(ABSEIL_CORD_LIBRARY "ABSEIL_CORD_LIBRARY-NOTFOUND" "libabsl_cord.so")
find_library(ABSEIL_CORDZ_INFO_LIBRARY absl_cordz_info)
_CHECK(ABSEIL_CORDZ_INFO_LIBRARY "ABSEIL_CORDZ_INFO_LIBRARY-NOTFOUND" "libabsl_cordz_info.so")
find_library(ABSEIL_CORDZ_FUNCTION_LIBRARY absl_cordz_functions)
_CHECK(ABSEIL_CORDZ_FUNCTION_LIBRARY "ABSEIL_CORDZ_FUNCTION_LIBRARY-NOTFOUND" "libabsl_cordz_functions.so")
find_library(ABSEIL_LOG_INTERNAL_CHECK_OP_LIBRARY absl_log_internal_check_op)
_CHECK(ABSEIL_LOG_INTERNAL_CHECK_OP_LIBRARY "ABSEIL_LOG_INTERNAL_CHECK_OP_LIBRARY-NOTFOUND" "libabsl_log_internal_check_op.so")
find_library(ABSEIL_LOG_INTERNAL_MESSAGE_LIBRARY absl_log_internal_message)
_CHECK(ABSEIL_LOG_INTERNAL_MESSAGE_LIBRARY "ABSEIL_LOG_INTERNAL_MESSAGE_LIBRARY-NOTFOUND" "libabsl_log_internal_message.so")

# check jsoncpp
find_library(JSONCPP_LIBRARY jsoncpp)
_CHECK(JSONCPP_LIBRARY "JSONCPP_LIBRARY-NOTFOUND" "libjsoncpp.so")


cmake_minimum_required(VERSION 3.5.1)

if(POLICY CMP0077)
  cmake_policy(SET CMP0077 NEW)
endif()

if(POLICY CMP0116)
  cmake_policy(SET CMP0116 OLD)
endif()

project(Plugin C CXX)
include(cmake/common.cmake)

#-------------------------------------------------------------------------------
# Options and settings
#-------------------------------------------------------------------------------
set(CMAKE_BUILD_WITH_INSTALL_NAME_DIR ON)

set(CMAKE_CXX_STANDARD 14 CACHE STRING "C++ standard to conform to")

find_package(MLIR REQUIRED CONFIG)
message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")

set(LLVM_RUNTIME_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/bin)
set(LLVM_LIBRARY_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/lib)
set(MLIR_BINARY_DIR ${CMAKE_BINARY_DIR})

#-------------------------------------------------------------------------------
# MLIR/LLVM Configuration
#-------------------------------------------------------------------------------
list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
include(TableGen)
include(AddLLVM)
include(AddMLIR)
#include(HandleLLVMOptions)

include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})
include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${PROJECT_BINARY_DIR}/include)
link_directories(${LLVM_BUILD_LIBRARY_DIR})
add_definitions(${LLVM_DEFINITIONS})

include_directories(include)

add_compile_options(-std=c++14 -Wall -fPIC)
# Proto file
get_filename_component(plg_proto "protos/plugin.proto" ABSOLUTE)
get_filename_component(plg_proto_path "${plg_proto}" PATH)

# Generated sources
set(plg_proto_srcs "${CMAKE_CURRENT_BINARY_DIR}/plugin.pb.cc")
set(plg_proto_hdrs "${CMAKE_CURRENT_BINARY_DIR}/plugin.pb.h")
set(plg_grpc_srcs "${CMAKE_CURRENT_BINARY_DIR}/plugin.grpc.pb.cc")
set(plg_grpc_hdrs "${CMAKE_CURRENT_BINARY_DIR}/plugin.grpc.pb.h")
add_custom_command(
      OUTPUT "${plg_proto_srcs}" "${plg_proto_hdrs}" "${plg_grpc_srcs}" "${plg_grpc_hdrs}"
      COMMAND ${CMD_PROTOC}
      ARGS --grpc_out "${CMAKE_CURRENT_BINARY_DIR}"
        --cpp_out "${CMAKE_CURRENT_BINARY_DIR}"
        -I "${plg_proto_path}"
        --plugin=protoc-gen-grpc="${CMD_GRPC_CPP_PLUGIN}"
        "${plg_proto}"
      DEPENDS "${plg_proto}")

# Include generated *.pb.h files
include_directories("${CMAKE_CURRENT_BINARY_DIR}")

# plg_grpc_proto
add_library(plg_grpc_proto STATIC
  ${plg_grpc_srcs}
  ${plg_grpc_hdrs}
  ${plg_proto_srcs}
  ${plg_proto_hdrs})
target_link_libraries(plg_grpc_proto
  ${GRPC_PP_REFLECTION_LIBRARY}
  ${GRPC_PP_LIBRARY}
  ${PROTOBUF_LIBRARY})

add_subdirectory(include)
add_subdirectory(lib)
add_library(pin_user SHARED
  "user/ArrayWidenPass.cpp"
  "user/InlineFunctionPass.cpp"
  "user/LocalVarSummeryPass.cpp"
  "user/user.cpp")

target_link_libraries(pin_user
  MLIRServerAPI
)
add_custom_command(TARGET pin_user COMMAND sha256sum libpin_user.so > libpin_user.sha256)

# pin_server
add_executable(pin_server 
  "lib/PluginServer/PluginServer.cpp"
  "lib/PluginAPI/ControlFlowAPI.cpp"
  "lib/PluginServer/PluginGrpc.cpp"
  "lib/PluginServer/PluginJson.cpp"
  "lib/PluginServer/PluginCom.cpp"
  "lib/PluginServer/PluginLog.cpp"
  "lib/PluginServer/main.cpp")
target_link_libraries(pin_server
  MLIRServerAPI
  pin_user
  ${JSONCPP_LIBRARY}
  plg_grpc_proto
  ${GRPC_PP_REFLECTION_LIBRARY}
  ${GRPC_PP_LIBRARY}
  ${PROTOBUF_LIBRARY}
  ${ABSEIL_SYNC_LIBRARY})

# install
install(TARGETS pin_server RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
install(TARGETS pin_user LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
include(GNUInstallDirs)
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/libpin_user.sha256" DESTINATION ${CMAKE_INSTALL_LIBDIR})

# uninstall target
if(NOT TARGET uninstall)
  configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake_uninstall.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/cmake_uninstall.cmake"
    IMMEDIATE @ONLY)

  add_custom_target(uninstall
    COMMAND ${CMAKE_COMMAND} -P ${CMAKE_CURRENT_BINARY_DIR}/cmake_uninstall.cmake)
endif()

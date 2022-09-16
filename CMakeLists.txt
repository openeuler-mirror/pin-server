
cmake_minimum_required(VERSION 3.5.1)

project(Plugin C CXX)

include(cmake/common.cmake)
include_directories(include)

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

add_library(pin_user SHARED "user.cpp")
add_custom_command(TARGET pin_user COMMAND sha256sum libpin_user.so > libpin_user.sha256)

# pin_server
add_executable(pin_server "lib/PluginServer/PluginServer.cpp" "lib/PluginAPI/PluginAPI_Server.cpp" "lib/PluginServer/PluginLog.cpp")
target_link_libraries(pin_server
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
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/libpin_user.sha256" DESTINATION lib)

# uninstall target
if(NOT TARGET uninstall)
  configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake_uninstall.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/cmake_uninstall.cmake"
    IMMEDIATE @ONLY)

  add_custom_target(uninstall
    COMMAND ${CMAKE_COMMAND} -P ${CMAKE_CURRENT_BINARY_DIR}/cmake_uninstall.cmake)
endif()

add_subdirectory(Dialect)

add_mlir_library(MLIRServerAPI
  PluginAPI/PluginServerAPI.cpp

  DEPENDS
  MLIRPluginOpsIncGen
  MLIRPluginServer

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRPluginServer
)
add_subdirectory(Dialect)

add_mlir_library(MLIRServerAPI
  PluginAPI/PluginServerAPI.cpp

  DEPENDS
  MLIRPluginOpsIncGen
  MLIRPlugin

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRPlugin
)
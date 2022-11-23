/* Copyright (c) Huawei Technologies Co., Ltd. 2022-2022. All rights reserved.

   Licensed under the Apache License, Version 2.0 (the "License"); you may
   not use this file except in compliance with the License. You may obtain
   a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
   WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
   License for the specific language governing permissions and limitations
   under the License.


*/
//===----------------------------------------------------------------------===//
//
// This is the top level file for the plugin dialect.
//
//===----------------------------------------------------------------------===//

#ifndef PLUGIN_DIALECT_TD
#define PLUGIN_DIALECT_TD

include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// Plugin Dialect Definition.
//===----------------------------------------------------------------------===//

// Provide a definition of the 'Plugin' dialect in the ODS framework so that we
// can define our operations.
def Plugin_Dialect : Dialect {
  let name = "Plugin";
  let summary = "The Plugin Dialect.";
  let description = [{}];
  let cppNamespace = "::mlir::Plugin";
}

// Base class for Plugin dialect operations. This operation inherits from the base
// `Op` class in OpBase.td, and provides:
//   * The parent dialect of the operation.
//   * The mnemonic for the operation, or the name without the dialect prefix.
//   * A list of traits for the operation.
class Plugin_Op<string mnemonic, list<OpTrait> traits = []> :
    Op<Plugin_Dialect, mnemonic, traits>;

#endif // PLUGIN_DIALECT_TD
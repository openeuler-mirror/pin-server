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

#ifndef PLUGIN_OPS_TD
#define PLUGIN_OPS_TD

include "PluginDialect.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/CallInterfaces.td"

def FunctionOp : Plugin_Op<"function", [NoSideEffect]> {
  let summary = "function with a region";
  let description = [{
    TODO.
  }];

  let arguments = (ins OptionalAttr<UI64Attr>:$id,
                       OptionalAttr<StrAttr>:$funcName,
                       OptionalAttr<BoolAttr>:$declaredInline);
  let regions = (region AnyRegion:$bodyRegion);

  // Add custom build methods for the operation. These method populates
  // the `state` that MLIR uses to create operations, i.e. these are used when
  // using `builder.create<Op>(...)`.
  let builders = [
    OpBuilderDAG<(ins "uint64_t":$id, "StringRef":$funcName, "bool":$declaredInline)>
  ];

  let extraClassDeclaration = [{
    std::vector<LoopOp> GetAllLoops();
  }];
}

def LocalDeclOp : Plugin_Op<"declaration", [NoSideEffect]> {
  let summary = "local variable declaration";
  let description = [{TODO}];
  let arguments = (ins OptionalAttr<UI64Attr>:$id,
                       OptionalAttr<StrAttr>:$symName,
                       OptionalAttr<I64Attr>:$typeID,
                       OptionalAttr<UI64Attr>:$typeWidth);
  let regions = (region AnyRegion:$bodyRegion);
  let builders = [
    OpBuilderDAG<(ins "uint64_t":$id, "StringRef":$symName,
                  "int64_t":$typeID, "uint64_t":$typeWidth)>
  ];
}

def LoopOp : Plugin_Op<"loop", [NoSideEffect]> {
  let summary = "loop operation";
  let description = [{
        TODO.
  }];
  let arguments = (ins OptionalAttr<UI64Attr>:$id,
                       OptionalAttr<UI32Attr>:$index,
                       OptionalAttr<UI64Attr>:$innerLoopId,
                       OptionalAttr<UI64Attr>:$outerLoopId,
                       OptionalAttr<UI32Attr>:$numBlock);
  let regions = (region AnyRegion:$bodyRegion);
  let builders = [
    OpBuilderDAG<(ins "uint64_t":$id, "uint32_t":$index,
                      "uint64_t":$innerLoopId, "uint64_t":$outerLoopId,
                      "uint32_t":$numBlock)>
  ];
  let extraClassDeclaration = [{
    uint64_t GetHeader();
    uint64_t GetLatch();
    std::pair<uint64_t, uint64_t> GetSingleExit();
    void Delete();
    LoopOp GetInnerLoop();
    LoopOp GetOuterLoop();
    bool IsBlockInside(uint64_t);
    std::vector<std::pair<uint64_t, uint64_t> > GetExitEdges();
    std::vector<uint64_t> GetLoopBody(); 
    void AddLoop(uint64_t, uint64_t);
  }];
}

#endif // PLUGIN_OPS_TD
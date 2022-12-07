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
// This file defines operations in the Plugin dialect.
//
//===----------------------------------------------------------------------===//

#include "PluginAPI/PluginServerAPI.h"
#include "Dialect/PluginDialect.h"
#include "Dialect/PluginOps.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpImplementation.h"

using namespace mlir;
using namespace mlir::Plugin;
using std::vector;
using std::pair;

void FunctionOp::build(mlir::OpBuilder &builder, mlir::OperationState &state,
                       uint64_t id, StringRef funcName, bool declaredInline) {
    FunctionOp::build(builder, state,
        builder.getI64IntegerAttr(id),
        builder.getStringAttr(funcName),
        builder.getBoolAttr(declaredInline));
}

vector<LoopOp> FunctionOp::GetAllLoops()
{
    PluginAPI::PluginServerAPI pluginAPI;
    uint64_t funcId = idAttr().getInt();
    return pluginAPI.GetLoopsFromFunc(funcId);
}

void LocalDeclOp::build(mlir::OpBuilder &builder, mlir::OperationState &state,
                        uint64_t id, StringRef symName,
                        int64_t typeID, uint64_t typeWidth) {
    LocalDeclOp::build(builder, state,
        builder.getI64IntegerAttr(id),
        builder.getStringAttr(symName),
        builder.getI64IntegerAttr(typeID),
        builder.getI64IntegerAttr(typeWidth));
}

void LoopOp::build(mlir::OpBuilder &builder, mlir::OperationState &state,
                   uint64_t id, uint32_t index, uint64_t innerLoopId,
                   uint64_t outerLoopId, uint32_t numBlock) {
    LoopOp::build(builder, state,
        builder.getI64IntegerAttr(id),
        builder.getI32IntegerAttr(index),
        builder.getI64IntegerAttr(innerLoopId),
        builder.getI64IntegerAttr(outerLoopId),
        builder.getI32IntegerAttr(numBlock));
}

// FIXME: use Block instead of uint64_t
uint64_t LoopOp::GetHeader()
{
    PluginAPI::PluginServerAPI pluginAPI;
    uint64_t loopId = idAttr().getInt();
    return pluginAPI.GetHeader(loopId);
}

// FIXME: use Block instead of uint64_t
uint64_t LoopOp::GetLatch()
{
    PluginAPI::PluginServerAPI pluginAPI;
    uint64_t loopId = idAttr().getInt();
    return pluginAPI.GetLatch(loopId);
}

vector<uint64_t> LoopOp::GetLoopBody()
{
    PluginAPI::PluginServerAPI pluginAPI;
    uint64_t loopId = idAttr().getInt();
    return pluginAPI.GetLoopBody(loopId);
}

pair<uint64_t, uint64_t> LoopOp::GetSingleExit()
{
    PluginAPI::PluginServerAPI pluginAPI;
    uint64_t loopId = idAttr().getInt();
    return pluginAPI.LoopSingleExit(loopId);
}

void LoopOp::Delete()
{
    PluginAPI::PluginServerAPI pluginAPI;
    uint64_t loopId = idAttr().getInt();
    pluginAPI.DeleteLoop(loopId);
}

LoopOp LoopOp::GetInnerLoop()
{
    PluginAPI::PluginServerAPI pluginAPI;
    uint64_t loopId = innerLoopIdAttr().getInt();
    return pluginAPI.GetLoopById(loopId);
}

LoopOp LoopOp::GetOuterLoop()
{
    PluginAPI::PluginServerAPI pluginAPI;
    uint64_t loopId = outerLoopIdAttr().getInt();
    return pluginAPI.GetLoopById(loopId);
}

// FIXME: 用Block替换uint64_t
bool LoopOp::IsBlockInside(uint64_t b)
{
    PluginAPI::PluginServerAPI pluginAPI;
    uint64_t loopId = idAttr().getInt();
    return pluginAPI.IsBlockInLoop(loopId, b);
}

vector<pair<uint64_t, uint64_t> > LoopOp::GetExitEdges()
{
    PluginAPI::PluginServerAPI pluginAPI;
    uint64_t loopId = idAttr().getInt();
    return pluginAPI.GetLoopExitEdges(loopId);
}

void LoopOp::AddLoop(uint64_t outerId, uint64_t funcId)
{
    PluginAPI::PluginServerAPI pluginAPI;
    uint64_t loopId = idAttr().getInt();
    return pluginAPI.AddLoop(loopId, outerId, funcId);
}

//===----------------------------------------------------------------------===//
// TableGen'd op method definitions
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "Dialect/PluginOps.cpp.inc"
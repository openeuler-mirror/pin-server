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
#include "PluginAPI/ControlFlowAPI.h"
#include "Dialect/PluginDialect.h"
#include "Dialect/PluginOps.h"
#include "Dialect/PluginTypes.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpImplementation.h"

using namespace mlir;
using namespace mlir::Plugin;
using std::vector;
using std::pair;

static uint64_t getBlockAddress(mlir::Block* b)
{
    if (mlir::Plugin::CondOp oops = dyn_cast<mlir::Plugin::CondOp>(b->back())) {
        return oops.addressAttr().getInt();
    } else if (mlir::Plugin::FallThroughOp oops = dyn_cast<mlir::Plugin::FallThroughOp>(b->back())) {
        return oops.addressAttr().getInt();
    } else if (mlir::Plugin::RetOp oops = dyn_cast<mlir::Plugin::RetOp>(b->back())) {
        return oops.addressAttr().getInt();
    } else {
        assert(false);
    }
}

void FunctionOp::build(OpBuilder &builder, OperationState &state,
                       uint64_t id, StringRef funcName, bool declaredInline)
{
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

LoopOp FunctionOp::AllocateNewLoop()
{
    PluginAPI::PluginServerAPI pluginAPI;
    uint64_t funcId = idAttr().getInt();
    return pluginAPI.AllocateNewLoop(funcId);
}

bool FunctionOp::IsDomInfoAvailable()
{
    PluginAPI::PluginServerAPI pluginAPI;
    return pluginAPI.IsDomInfoAvailable();
}

void LocalDeclOp::build(OpBuilder &builder, OperationState &state,
                        uint64_t id, StringRef symName,
                        int64_t typeID, uint64_t typeWidth)
{
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

Block* LoopOp::GetHeader()
{
    PluginAPI::PluginServerAPI pluginAPI;
    uint64_t loopId = idAttr().getInt();
    return pluginAPI.GetHeader(loopId);
}

Block* LoopOp::GetLatch()
{
    PluginAPI::PluginServerAPI pluginAPI;
    uint64_t loopId = idAttr().getInt();
    return pluginAPI.GetLatch(loopId);
}

void LoopOp::SetHeader(mlir::Block* b)
{
    PluginAPI::PluginServerAPI pluginAPI;
    uint64_t loopId = idAttr().getInt();
    uint64_t blockId = pluginAPI.FindBasicBlock(b);
    pluginAPI.SetHeader(loopId, blockId);
}

void LoopOp::SetLatch(mlir::Block* b)
{
    PluginAPI::PluginServerAPI pluginAPI;
    uint64_t loopId = idAttr().getInt();
    uint64_t blockId = pluginAPI.FindBasicBlock(b);
    pluginAPI.SetLatch(loopId, blockId);
}

vector<mlir::Block*> LoopOp::GetLoopBody()
{
    PluginAPI::PluginServerAPI pluginAPI;
    uint64_t loopId = idAttr().getInt();
    return pluginAPI.GetLoopBody(loopId);
}

pair<mlir::Block*, mlir::Block*> LoopOp::GetSingleExit()
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

bool LoopOp::IsBlockInside(mlir::Block* b)
{
    PluginAPI::PluginServerAPI pluginAPI;
    uint64_t loopId = idAttr().getInt();
    uint64_t blockId = pluginAPI.FindBasicBlock(b);
    return pluginAPI.IsBlockInLoop(loopId, blockId);
}

bool LoopOp::IsLoopFather(mlir::Block* b)
{
    PluginAPI::PluginServerAPI pluginAPI;
    uint64_t loopId = idAttr().getInt();
    uint64_t blockId = pluginAPI.FindBasicBlock(b);
    LoopOp loopFather = pluginAPI.GetBlockLoopFather(blockId);
    uint64_t id = loopFather.idAttr().getInt();
    return id == loopId;
}

vector<pair<mlir::Block*, mlir::Block*> > LoopOp::GetExitEdges()
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

void LoopOp::AddBlock(mlir::Block* block)
{
    PluginAPI::PluginServerAPI pluginAPI;
    uint64_t blockId = pluginAPI.FindBasicBlock(block);
    uint64_t loopId = idAttr().getInt();
    pluginAPI.AddBlockToLoop(blockId, loopId);
}

//===----------------------------------------------------------------------===//
// PlaceholderOp

void PlaceholderOp::build(OpBuilder &builder, OperationState &state,
                          uint64_t id, IDefineCode defCode, bool readOnly,
                          Type retType)
{
    state.addAttribute("id", builder.getI64IntegerAttr(id));
    state.addAttribute("defCode",
        builder.getI32IntegerAttr(static_cast<int32_t>(defCode)));
    state.addAttribute("readOnly", builder.getBoolAttr(readOnly));
    state.addTypes(retType);
}

//===----------------------------------------------------------------------===//
// MemOp

void MemOp::build(OpBuilder &builder, OperationState &state,
                  uint64_t id, IDefineCode defCode, bool readOnly,
                  Value addr, Value offset, Type retType)
{
    state.addAttribute("id", builder.getI64IntegerAttr(id));
    state.addAttribute("readOnly", builder.getBoolAttr(readOnly));
    state.addOperands({addr, offset});
    state.addAttribute("defCode",
                       builder.getI32IntegerAttr(static_cast<int32_t>(defCode)));
    if (retType) state.addTypes(retType);
}

//===----------------------------------------------------------------------===//
// SSAOp

void SSAOp::build(OpBuilder &builder, OperationState &state, uint64_t id,
                  IDefineCode defCode, bool readOnly, uint64_t nameVarId,
                  uint64_t ssaParmDecl, uint64_t version, uint64_t definingId,
                  Type retType)
{
    state.addAttribute("id", builder.getI64IntegerAttr(id));
    state.addAttribute("defCode",
                       builder.getI32IntegerAttr(static_cast<int32_t>(defCode)));
    state.addAttribute("readOnly", builder.getBoolAttr(readOnly));
    state.addAttribute("nameVarId", builder.getI64IntegerAttr(nameVarId));
    state.addAttribute("ssaParmDecl", builder.getI64IntegerAttr(ssaParmDecl));
    state.addAttribute("version", builder.getI64IntegerAttr(version));
    state.addAttribute("definingId", builder.getI64IntegerAttr(definingId));
    state.addTypes(retType);
}

Value SSAOp::MakeSSA(OpBuilder &builder, Type t)
{
    PluginAPI::PluginServerAPI pluginAPI;
    PinServer::PluginServer::GetInstance()->SetOpBuilder(builder);
    return pluginAPI.CreateSSAOp(t);
}

Value SSAOp::Copy()
{
    PluginAPI::PluginServerAPI pluginAPI;
    OpBuilder builder(this->getOperation());
    PinServer::PluginServer::GetInstance()->SetOpBuilder(builder);
    return pluginAPI.CopySSAOp(this->idAttr().getInt());
}

Value SSAOp::GetCurrentDef()
{
    PluginAPI::PluginServerAPI pluginAPI;
    OpBuilder builder(this->getOperation());
    PinServer::PluginServer::GetInstance()->SetOpBuilder(builder);
    return pluginAPI.GetCurrentDefFromSSA(this->idAttr().getInt());
}

bool SSAOp::SetCurrentDef(Value def)
{
    PlaceholderOp phOp = def.getDefiningOp<PlaceholderOp>();
    uint64_t defId = phOp.idAttr().getInt();
    PluginAPI::PluginServerAPI pluginAPI;
    if (pluginAPI.SetCurrentDefInSSA(this->idAttr().getInt(), defId)) {
        return true;
    }
    return false;
}

Operation* SSAOp::GetSSADefOperation()
{
    PluginAPI::PluginServerAPI pluginAPI;
    uint64_t definingId = definingIdAttr().getInt();
    return pluginAPI.GetSSADefOperation(definingId);
}

//===----------------------------------------------------------------------===//
// ConstOp

void ConstOp::build(OpBuilder &builder, OperationState &state, uint64_t id,
                    IDefineCode defCode, bool readOnly, Attribute init,
                    Type retType)
{
    state.addAttribute("id", builder.getI64IntegerAttr(id));
    state.addAttribute("defCode",
                       builder.getI32IntegerAttr(static_cast<int32_t>(defCode)));
    state.addAttribute("readOnly", builder.getBoolAttr(readOnly));
    state.addAttribute("init", init);
    if (retType) state.addTypes(retType);
}

Value ConstOp::CreateConst(OpBuilder &builder, Attribute value, Type retType)
{
    PluginAPI::PluginServerAPI pluginAPI;
    PinServer::PluginServer::GetInstance()->SetOpBuilder(builder);
    return pluginAPI.CreateConstOp(value, retType);
}

//===----------------------------------------------------------------------===//
// PointerOp

void PointerOp::build(OpBuilder &builder, OperationState &state, uint64_t id,
                      IDefineCode defCode, bool readOnly, Type retType,
                      bool pointeeReadOnly)
{
    state.addAttribute("id", builder.getI64IntegerAttr(id));
    state.addAttribute("defCode",
        builder.getI32IntegerAttr(static_cast<int32_t>(defCode)));
    state.addAttribute("readOnly", builder.getBoolAttr(readOnly));
    state.addTypes(retType);
    state.addAttribute("pointeeReadOnly", builder.getBoolAttr(pointeeReadOnly));
}

//===----------------------------------------------------------------------===//
// CallOp

void CallOp::build(OpBuilder &builder, OperationState &state,
                   int64_t id, StringRef callee,
                   ArrayRef<Value> arguments)
{
    state.addAttribute("id", builder.getI64IntegerAttr(id));
    state.addOperands(arguments);
    state.addAttribute("callee", builder.getSymbolRefAttr(callee));
}

/// Return the callee of the generic call operation, this is required by the
/// call interface.
CallInterfaceCallable CallOp::getCallableForCallee()
{
    return (*this)->getAttrOfType<SymbolRefAttr>("callee");
}

/// Get the argument operands to the called function, this is required by the
/// call interface.
Operation::operand_range CallOp::getArgOperands() { return inputs(); }

bool CallOp::SetLHS(Value lhs)
{
    PlaceholderOp phOp = lhs.getDefiningOp<PlaceholderOp>();
    uint64_t lhsId = phOp.idAttr().getInt();
    PluginAPI::PluginServerAPI pluginAPI;
    if (pluginAPI.SetLhsInCallOp(this->idAttr().getInt(), lhsId)) {
        (*this)->setOperand(0, lhs);
        return true;
    }
    return false;
}

void CallOp::build(OpBuilder &builder, OperationState &state,
                   Value func, ArrayRef<Value> arguments)
{
    PluginAPI::PluginServerAPI pluginAPI;
    PlaceholderOp funcOp = func.getDefiningOp<PlaceholderOp>();
    uint64_t funcId = funcOp.idAttr().getInt();
    vector<uint64_t> argIds;
    for (auto v : arguments) {
        PlaceholderOp argOp = v.getDefiningOp<PlaceholderOp>();
        uint64_t argId = argOp.idAttr().getInt();
        argIds.push_back(argId);
    }
    Block *buildBlock = builder.getBlock();
    uint64_t blockId = pluginAPI.FindBasicBlock(buildBlock);
    uint64_t id = pluginAPI.CreateCallOp(blockId, funcId, argIds);
    state.addAttribute("id", builder.getI64IntegerAttr(id));
    state.addOperands(arguments);
    // FIXME: DEF_BUILTIN.
    state.addAttribute("callee", builder.getSymbolRefAttr("ctzll"));
}

void CallOp::build(OpBuilder &builder, OperationState &state,
                   ArrayRef<Value> arguments)
{
    PluginAPI::PluginServerAPI pluginAPI;
    vector<uint64_t> argIds;
    for (auto v : arguments) {
        PlaceholderOp argOp = v.getDefiningOp<PlaceholderOp>();
        uint64_t argId = argOp.idAttr().getInt();
        argIds.push_back(argId);
    }
    Block *buildBlock = builder.getBlock();
    uint64_t blockId = pluginAPI.FindBasicBlock(buildBlock);
    uint64_t funcId = 0;
    uint64_t id = pluginAPI.CreateCallOp(blockId, funcId, argIds);
    state.addAttribute("id", builder.getI64IntegerAttr(id));
    state.addOperands(arguments);
    state.addAttribute("callee", builder.getSymbolRefAttr("ctzll"));
}

//===----------------------------------------------------------------------===//
// CondOp

void CondOp::build(OpBuilder &builder, OperationState &state,
                   uint64_t id, uint64_t address, IComparisonCode condCode,
                   Value lhs, Value rhs, Block* tb, Block* fb, uint64_t tbaddr,
                   uint64_t fbaddr, Value trueLabel, Value falseLabel) {
    state.addAttribute("id", builder.getI64IntegerAttr(id));
    state.addAttribute("address", builder.getI64IntegerAttr(address));
    state.addAttribute("tbaddr", builder.getI64IntegerAttr(tbaddr));
    state.addAttribute("fbaddr", builder.getI64IntegerAttr(fbaddr));
    state.addOperands({lhs, rhs});
    state.addSuccessors(tb);
    state.addSuccessors(fb);
    state.addAttribute("condCode",
        builder.getI32IntegerAttr(static_cast<int32_t>(condCode)));
    if (trueLabel != nullptr) state.addOperands(trueLabel);
    if (falseLabel != nullptr) state.addOperands(falseLabel);
}

void CondOp::build(OpBuilder &builder, OperationState &state,
                   IComparisonCode condCode, Value lhs, Value rhs, Block* tb,
                   Block* fb)
{
    PluginAPI::PluginServerAPI pluginAPI;
    PlaceholderOp lhsOp = lhs.getDefiningOp<PlaceholderOp>();
    uint64_t lhsId = lhsOp.idAttr().getInt();
    PlaceholderOp rhsOp = rhs.getDefiningOp<PlaceholderOp>();
    uint64_t rhsId = rhsOp.idAttr().getInt();
    Block *buildBlock = builder.getBlock();
    uint64_t blockId = pluginAPI.FindBasicBlock(buildBlock);
    uint64_t tbaddr = getBlockAddress(tb);
    uint64_t fbaddr = getBlockAddress(fb);
    uint64_t id = pluginAPI.CreateCondOp(blockId, condCode, lhsId, rhsId,
                                         tbaddr, fbaddr);
    state.addAttribute("id", builder.getI64IntegerAttr(id));
    state.addOperands({lhs, rhs});
    state.addAttribute("condCode",
            builder.getI32IntegerAttr(static_cast<int32_t>(condCode)));
    state.addSuccessors(tb);
    state.addSuccessors(fb);
    state.addAttribute("tbaddr", builder.getI64IntegerAttr(tbaddr));
    state.addAttribute("fbaddr", builder.getI64IntegerAttr(fbaddr));
}

//===----------------------------------------------------------------------===//
// PhiOp

void PhiOp::build(OpBuilder &builder, OperationState &state,
                   ArrayRef<Value> operands, uint64_t id,
                   uint32_t capacity, uint32_t nArgs, uint64_t defStmtId)
{
    state.addAttribute("id", builder.getI64IntegerAttr(id));
    state.addAttribute("capacity", builder.getI32IntegerAttr(capacity));
    state.addAttribute("nArgs", builder.getI32IntegerAttr(nArgs));
    state.addAttribute("defStmtId", builder.getI64IntegerAttr(defStmtId));
    state.addOperands(operands);
}

Value PhiOp::GetResult()
{
    PluginAPI::PluginServerAPI pluginAPI;
    OpBuilder builder(this->getOperation());
    PinServer::PluginServer::GetInstance()->SetOpBuilder(builder);
    return pluginAPI.GetResultFromPhi(this->idAttr().getInt());
}

PhiOp PhiOp::CreatePhi(Value arg, Block *block)
{
    uint64_t argId = 0;
    if (arg != nullptr) {
        PlaceholderOp phOp = arg.getDefiningOp<PlaceholderOp>();
        argId = phOp.idAttr().getInt();
    }
    PluginAPI::PluginServerAPI pluginAPI;
    uint64_t blockId = pluginAPI.FindBasicBlock(block);
    return pluginAPI.CreatePhiOp(argId, blockId);
}

bool PhiOp::AddArg(Value arg, Block *pred, Block *succ)
{
    PlaceholderOp phOp = arg.getDefiningOp<PlaceholderOp>();
    uint64_t argId = phOp.idAttr().getInt();
    PluginAPI::PluginServerAPI pluginAPI;
    uint64_t predId = pluginAPI.FindBasicBlock(pred);
    uint64_t succId = pluginAPI.FindBasicBlock(succ);
    if (pluginAPI.AddArgInPhiOp(this->idAttr().getInt(), argId, predId, succId)) {
        uint32_t nArg = this->nArgsAttr().getInt() + 1;
        OpBuilder builder(this->getOperation());
        (*this)->setAttr("nArgs", builder.getI32IntegerAttr(nArg));
        return true;
    }
    return false;
}

//===----------------------------------------------------------------------===//
// AssignOp

void AssignOp::build(OpBuilder &builder, OperationState &state,
                     ArrayRef<Value> operands,
                     uint64_t id, IExprCode exprCode,
                     uint64_t defStmtId)
{
    state.addAttribute("id", builder.getI64IntegerAttr(id));
    state.addAttribute("exprCode",
        builder.getI32IntegerAttr(static_cast<int32_t>(exprCode)));
    state.addAttribute("defStmtId", builder.getI64IntegerAttr(defStmtId));
    state.addOperands(operands);
}

void AssignOp::build(OpBuilder &builder, OperationState &state,
                     ArrayRef<Value> operands, IExprCode exprCode,
                     uint64_t defStmtId)
{
    PluginAPI::PluginServerAPI pluginAPI;
    vector<uint64_t> argIds;
    for (auto v : operands) {
        PlaceholderOp argOp = v.getDefiningOp<PlaceholderOp>();
        uint64_t argId = argOp.idAttr().getInt();
        argIds.push_back(argId);
    }
    Block *buildBlock = builder.getBlock();
    uint64_t blockId = pluginAPI.FindBasicBlock(buildBlock);
    uint64_t id = pluginAPI.CreateAssignOp(blockId, exprCode, argIds);
    state.addAttribute("id", builder.getI64IntegerAttr(id));
    state.addAttribute("exprCode",
        builder.getI32IntegerAttr(static_cast<int32_t>(exprCode)));
    state.addAttribute("defStmtId", builder.getI64IntegerAttr(defStmtId));
    state.addOperands(operands);
}

//===----------------------------------------------------------------------===//
// BaseOp

void BaseOp::build(OpBuilder &builder, OperationState &state,
                   uint64_t id, StringRef opCode)
{
    state.addAttribute("id", builder.getI64IntegerAttr(id));
    state.addAttribute("opCode", builder.getStringAttr(opCode));
}

//===----------------------------------------------------------------------===//
// FallThroughOp

void FallThroughOp::build(OpBuilder &builder, OperationState &state,
                          uint64_t address, Block* dest, uint64_t destaddr)
{
    state.addAttribute("address", builder.getI64IntegerAttr(address));
    state.addAttribute("destaddr", builder.getI64IntegerAttr(destaddr));
    state.addSuccessors(dest);
}

void FallThroughOp::build(OpBuilder &builder, OperationState &state,
                          uint64_t address, Block* dest)
{
    PluginAPI::PluginServerAPI pluginAPI;
    uint64_t destaddr = pluginAPI.FindBasicBlock(dest);

    PluginAPI::ControlFlowAPI cfgAPI;
    cfgAPI.CreateFallthroughOp(address, destaddr);
    state.addAttribute("address", builder.getI64IntegerAttr(address));
    state.addAttribute("destaddr", builder.getI64IntegerAttr(destaddr));
    state.addSuccessors(dest);
}

//===----------------------------------------------------------------------===//
// RetOp

void RetOp::build(OpBuilder &builder, OperationState &state, uint64_t address)
{
    state.addAttribute("address", builder.getI64IntegerAttr(address));
}

//===----------------------------------------------------------------------===//
// TableGen'd op method definitions
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "Dialect/PluginOps.cpp.inc"
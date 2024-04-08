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
// ===----------------------------------------------------------------------===//
//
// This file defines operations in the Plugin dialect.
//
// ===----------------------------------------------------------------------===//

#include "PluginAPI/PluginServerAPI.h"
#include "PluginAPI/ControlFlowAPI.h"
#include "PluginAPI/DataFlowAPI.h"
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

static uint64_t GetValueId(Value v)
{
    Operation *op = v.getDefiningOp();
    if (auto mOp = dyn_cast<MemOp>(op)) {
        return mOp.getId();
    } else if (auto ssaOp = dyn_cast<SSAOp>(op)) {
        return ssaOp.getId();
    } else if (auto cstOp = dyn_cast<ConstOp>(op)) {
        return cstOp.getId();
    } else if (auto treelistop = dyn_cast<ListOp>(op)) {
        return treelistop.getId();
    } else if (auto strop = dyn_cast<StrOp>(op)) {
        return strop.getId();
    } else if (auto arrayop = dyn_cast<ArrayOp>(op)) {
        return arrayop.getId();
    } else if (auto declop = dyn_cast<DeclBaseOp>(op)) {
        return declop.getId();
    } else if (auto fieldop = dyn_cast<FieldDeclOp>(op)) {
        return fieldop.getId();
    } else if (auto addressop = dyn_cast<AddressOp>(op)) {
        return addressop.getId();
    } else if (auto constructorop = dyn_cast<ConstructorOp>(op)) {
        return constructorop.getId();
    } else if (auto vecop = dyn_cast<VecOp>(op)) {
        return vecop.getId();
    } else if (auto blockop = dyn_cast<BlockOp>(op)) {
        return blockop.getId();
    } else if (auto compOp = dyn_cast<ComponentOp>(op)) {
        return compOp.getId();
    } else if (auto phOp = dyn_cast<PlaceholderOp>(op)) {
        return phOp.getId();
    }
    return 0;
}

static uint64_t getBlockAddress(mlir::Block* b)
{
    if (mlir::Plugin::CondOp oops = dyn_cast<mlir::Plugin::CondOp>(b->back())) {
        return oops.getAddressAttr().getInt();
    } else if (mlir::Plugin::FallThroughOp oops = dyn_cast<mlir::Plugin::FallThroughOp>(b->back())) {
        return oops.getAddressAttr().getInt();
    } else if (mlir::Plugin::RetOp oops = dyn_cast<mlir::Plugin::RetOp>(b->back())) {
        return oops.getAddressAttr().getInt();
    } else if (mlir::Plugin::GotoOp oops = dyn_cast<mlir::Plugin::GotoOp>(b->back())) {
        return oops.getAddressAttr().getInt();
    } else if (mlir::Plugin::TransactionOp oops = dyn_cast<mlir::Plugin::TransactionOp>(b->back())) {
        return oops.getAddressAttr().getInt();
    } else {
        assert(false);
    }
}

// ===----------------------------------------------------------------------===//
// CGnodeOp

void CGnodeOp::build(OpBuilder &builder, OperationState &state,
                     uint64_t id, StringRef symbolName, bool definition,
                     uint32_t order)
{
    state.addRegion();
    state.addAttribute("id", builder.getI64IntegerAttr(id));
    state.addAttribute("symbolName", builder.getStringAttr(symbolName));
    state.addAttribute("definition", builder.getBoolAttr(definition));
    state.addAttribute("order", builder.getI32IntegerAttr(order));
}

// Value CGnodeOp::GetDecl()
// {
//     PluginAPI::PluginServerAPI pluginAPI;
//     uint64_t nodeId = getIdAttr().getInt();
//     return pluginAPI.GetDeclFromCGnode(nodeId);
// }

bool CGnodeOp::IsRealSymbol()
{
    PluginAPI::PluginServerAPI pluginAPI;
    uint64_t nodeId = getIdAttr().getInt();
    return pluginAPI.IsRealSymbolOfCGnode(nodeId);
}

// ===----------------------------------------------------------------------===//

void FunctionOp::build(OpBuilder &builder, OperationState &state,
                       uint64_t id, StringRef funcName, bool declaredInline, Type type, bool validType)
{
    state.addRegion();
    state.addAttribute("id", builder.getI64IntegerAttr(id));
    state.addAttribute("funcName", builder.getStringAttr(funcName));
    state.addAttribute("declaredInline", builder.getBoolAttr(declaredInline));
    state.addAttribute("validType", builder.getBoolAttr(validType));
    if (type) state.addAttribute("type", TypeAttr::get(type));
}

void FunctionOp::build(OpBuilder &builder, OperationState &state,
                       uint64_t id, StringRef funcName, bool declaredInline, bool validType)
{
    state.addRegion();
    state.addAttribute("id", builder.getI64IntegerAttr(id));
    state.addAttribute("funcName", builder.getStringAttr(funcName));
    state.addAttribute("declaredInline", builder.getBoolAttr(declaredInline));
    state.addAttribute("validType", builder.getBoolAttr(validType));
}

Type FunctionOp::getResultType()
{
    PluginIR::PluginFunctionType resultType = getType().dyn_cast<PluginIR::PluginFunctionType>();
    return resultType;
}

vector<LoopOp> FunctionOp::GetAllLoops()
{
    PluginAPI::PluginServerAPI pluginAPI;
    uint64_t funcId = getIdAttr().getInt();
    return pluginAPI.GetLoopsFromFunc(funcId);
}

LoopOp FunctionOp::AllocateNewLoop()
{
    PluginAPI::PluginServerAPI pluginAPI;
    uint64_t funcId = getIdAttr().getInt();
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
                   uint64_t outerLoopId, uint32_t numBlock)
{
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
    uint64_t loopId = getIdAttr().getInt();
    return pluginAPI.GetHeader(loopId);
}

Block* LoopOp::GetLatch()
{
    PluginAPI::PluginServerAPI pluginAPI;
    uint64_t loopId = getIdAttr().getInt();
    return pluginAPI.GetLatch(loopId);
}

void LoopOp::SetHeader(mlir::Block* b)
{
    PluginAPI::PluginServerAPI pluginAPI;
    pluginAPI.SetHeader(this, b);
}

void LoopOp::SetLatch(mlir::Block* b)
{
    PluginAPI::PluginServerAPI pluginAPI;
    pluginAPI.SetLatch(this, b);
}

vector<mlir::Block*> LoopOp::GetLoopBody()
{
    PluginAPI::PluginServerAPI pluginAPI;
    uint64_t loopId = getIdAttr().getInt();
    return pluginAPI.GetLoopBody(loopId);
}

pair<mlir::Block*, mlir::Block*> LoopOp::GetSingleExit()
{
    PluginAPI::PluginServerAPI pluginAPI;
    uint64_t loopId = getIdAttr().getInt();
    return pluginAPI.LoopSingleExit(loopId);
}

void LoopOp::Delete()
{
    PluginAPI::PluginServerAPI pluginAPI;
    uint64_t loopId = getIdAttr().getInt();
    pluginAPI.DeleteLoop(loopId);
}

LoopOp LoopOp::GetInnerLoop()
{
    PluginAPI::PluginServerAPI pluginAPI;
    uint64_t loopId = getInnerLoopIdAttr().getInt();
    return pluginAPI.GetLoopById(loopId);
}

LoopOp LoopOp::GetOuterLoop()
{
    PluginAPI::PluginServerAPI pluginAPI;
    uint64_t loopId = getOuterLoopIdAttr().getInt();
    return pluginAPI.GetLoopById(loopId);
}

bool LoopOp::IsBlockInside(mlir::Block* b)
{
    PluginAPI::PluginServerAPI pluginAPI;
    uint64_t loopId = getIdAttr().getInt();
    uint64_t blockId = pluginAPI.FindBasicBlock(b);
    return pluginAPI.IsBlockInLoop(loopId, blockId);
}

bool LoopOp::IsLoopFather(mlir::Block* b)
{
    PluginAPI::PluginServerAPI pluginAPI;
    uint64_t loopId = getIdAttr().getInt();
    LoopOp loopFather = pluginAPI.GetBlockLoopFather(b);
    uint64_t id = loopFather.getIdAttr().getInt();
    return id == loopId;
}

vector<pair<mlir::Block*, mlir::Block*> > LoopOp::GetExitEdges()
{
    PluginAPI::PluginServerAPI pluginAPI;
    uint64_t loopId = getIdAttr().getInt();
    return pluginAPI.GetLoopExitEdges(loopId);
}

void LoopOp::AddLoop(LoopOp* outerLoop, FunctionOp* funcOp)
{
    PluginAPI::PluginServerAPI pluginAPI;
    uint64_t loopId = getIdAttr().getInt();
    return pluginAPI.AddLoop(loopId, outerLoop->getIdAttr().getInt(),
                             funcOp->getIdAttr().getInt());
}

void LoopOp::AddBlock(mlir::Block* block)
{
    PluginAPI::PluginServerAPI pluginAPI;
    pluginAPI.AddBlockToLoop(block, this);
}

// ===----------------------------------------------------------------------===//
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

// ===----------------------------------------------------------------------===//
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
// DeclBaseOp

void DeclBaseOp::build(OpBuilder &builder, OperationState &state, uint64_t id,
                      IDefineCode defCode, bool readOnly, bool addressable, bool used, int32_t uid, Value initial,
                      Value name, std::optional<uint64_t> chain, Type retType)
{
    state.addAttribute("id", builder.getI64IntegerAttr(id));
    state.addAttribute("defCode",
        builder.getI32IntegerAttr(static_cast<int32_t>(defCode)));
    state.addAttribute("readOnly", builder.getBoolAttr(readOnly));
    state.addAttribute("addressable", builder.getBoolAttr(addressable));
    state.addAttribute("used", builder.getBoolAttr(used));
    state.addAttribute("uid", builder.getI32IntegerAttr(uid));
    state.addOperands(initial);
    if(chain) {
        state.addAttribute("chain", builder.getI64IntegerAttr(chain.value()));
    }
    state.addOperands(name);
    state.addTypes(retType);
}

//===----------------------------------------------------------------------===//
// BlockOp

void BlockOp::build(OpBuilder &builder, OperationState &state, uint64_t id,
                      IDefineCode defCode, bool readOnly, std::optional<Value> vars, std::optional<uint64_t> supercontext,
                      std::optional<Value> subblocks, std::optional<Value> abstract_origin, std::optional<Value> chain, Type retType)
{
    state.addAttribute("id", builder.getI64IntegerAttr(id));
    state.addAttribute("defCode",
        builder.getI32IntegerAttr(static_cast<int32_t>(defCode)));
    state.addAttribute("readOnly", builder.getBoolAttr(readOnly));
    if(vars) {
        state.addOperands(vars.value());
    }
    if(supercontext) {
        state.addAttribute("supercontext", builder.getI64IntegerAttr(supercontext.value()));
    }
    if(subblocks) {
        state.addOperands(subblocks.value());
    }
    if(abstract_origin) {
        state.addOperands(abstract_origin.value());
    }
    if(chain) {
        state.addOperands(chain.value());
    }
    state.addTypes(retType);
}

//===----------------------------------------------------------------------===//
// ComponentOp

void ComponentOp::build(OpBuilder &builder, OperationState &state,
                  uint64_t id, IDefineCode defCode, bool readOnly,
                  Value component, Value field, Type retType)
{
    state.addAttribute("id", builder.getI64IntegerAttr(id));
    state.addAttribute("readOnly", builder.getBoolAttr(readOnly));
    state.addAttribute("defCode",
                       builder.getI32IntegerAttr(static_cast<int32_t>(defCode)));

    state.addOperands({component, field});
    if (retType) state.addTypes(retType);
}
//===----------------------------------------------------------------------===//
// VecOp

void VecOp::build(OpBuilder &builder, OperationState &state, uint64_t id,
                      IDefineCode defCode, bool readOnly, int32_t len, ArrayRef<Value> elements, Type retType)
{
    state.addAttribute("id", builder.getI64IntegerAttr(id));
    state.addAttribute("defCode",
        builder.getI32IntegerAttr(static_cast<int32_t>(defCode)));
    state.addAttribute("readOnly", builder.getBoolAttr(readOnly));
    state.addAttribute("len", builder.getI32IntegerAttr(len));
    state.addOperands(elements);
    state.addTypes(retType);
}

//===----------------------------------------------------------------------===//
// ConstructorOp

void ConstructorOp::build(OpBuilder &builder, OperationState &state, uint64_t id,
                      IDefineCode defCode, bool readOnly, int32_t len, ArrayRef<Value> idx,
                      ArrayRef<Value> val, Type retType)
{
    state.addAttribute("id", builder.getI64IntegerAttr(id));
    state.addAttribute("defCode",
        builder.getI32IntegerAttr(static_cast<int32_t>(defCode)));
    state.addAttribute("readOnly", builder.getBoolAttr(readOnly));
    state.addAttribute("len", builder.getI32IntegerAttr(len));
    state.addOperands(idx);
    state.addOperands(val);

    state.addTypes(retType);
}

//===----------------------------------------------------------------------===//
// FieldDeclOp

void FieldDeclOp::build(OpBuilder &builder, OperationState &state, uint64_t id,
                      IDefineCode defCode, bool readOnly, bool addressable, bool used, int32_t uid, Value initial,
                      Value name, uint64_t chain, Value fieldOffset, Value fieldBitOffset, Type retType)
{
    state.addAttribute("id", builder.getI64IntegerAttr(id));
    state.addAttribute("defCode",
        builder.getI32IntegerAttr(static_cast<int32_t>(defCode)));
    state.addAttribute("readOnly", builder.getBoolAttr(readOnly));
    state.addAttribute("addressable", builder.getBoolAttr(addressable));
    state.addAttribute("used", builder.getBoolAttr(used));
    state.addAttribute("uid", builder.getI32IntegerAttr(uid));
    state.addOperands(initial);
    if(chain) {
        state.addAttribute("chain", builder.getI64IntegerAttr(chain));
    }
    state.addOperands(name);
    state.addOperands({fieldOffset, fieldBitOffset});
    state.addTypes(retType);
}

void FieldDeclOp::SetName(FieldDeclOp field)
{
    PluginAPI::PluginServerAPI pluginAPI;
    uint64_t fieldId = field.getIdAttr().getInt();
    unsigned idx = 1;
    this->setOperand(idx ,field.GetName());
    return pluginAPI.SetDeclName(this->getIdAttr().getInt(), fieldId);
}

void FieldDeclOp::SetType(FieldDeclOp field)
{
    PluginAPI::PluginServerAPI pluginAPI;
    uint64_t fieldId = field.getIdAttr().getInt();
    return pluginAPI.SetDeclType(this->getIdAttr().getInt(), fieldId);
}

void FieldDeclOp::SetDeclAlign(FieldDeclOp field)
{
    PluginAPI::PluginServerAPI pluginAPI;
    uint64_t fieldId = field.getIdAttr().getInt();
    return pluginAPI.SetDeclAlign(this->getIdAttr().getInt(), fieldId);
}

void FieldDeclOp::SetUserAlign(FieldDeclOp field)
{
    PluginAPI::PluginServerAPI pluginAPI;
    uint64_t fieldId = field.getIdAttr().getInt();
    return pluginAPI.SetUserAlign(this->getIdAttr().getInt(), fieldId);
}

unsigned FieldDeclOp::GetTypeSize()
{
    PluginAPI::PluginServerAPI pluginAPI;
    return pluginAPI.GetDeclTypeSize(this->getIdAttr().getInt());
}

void FieldDeclOp::SetSourceLocation(FieldDeclOp field)
{
    PluginAPI::PluginServerAPI pluginAPI;
    uint64_t fieldId = field.getIdAttr().getInt();
    return pluginAPI.SetSourceLocation(this->getIdAttr().getInt(), fieldId);
}

void FieldDeclOp::SetAddressable(FieldDeclOp field)
{
    PluginAPI::PluginServerAPI pluginAPI;
    uint64_t fieldId = field.getIdAttr().getInt();
    return pluginAPI.SetAddressable(this->getIdAttr().getInt(), fieldId);
}

void FieldDeclOp::SetNonAddressablep(FieldDeclOp field)
{
    PluginAPI::PluginServerAPI pluginAPI;
    uint64_t fieldId = field.getIdAttr().getInt();
    return pluginAPI.SetNonAddressablep(this->getIdAttr().getInt(), fieldId);
}

void FieldDeclOp::SetVolatile(FieldDeclOp field)
{
    PluginAPI::PluginServerAPI pluginAPI;
    uint64_t fieldId = field.getIdAttr().getInt();
    return pluginAPI.SetVolatile(this->getIdAttr().getInt(), fieldId);
}

void FieldDeclOp::SetDeclContext(uint64_t declId)
{
    PluginAPI::PluginServerAPI pluginAPI;
    return pluginAPI.SetDeclContext(this->getIdAttr().getInt(), declId);
}

void FieldDeclOp::SetDeclChain(FieldDeclOp field)
{
    PluginAPI::PluginServerAPI pluginAPI;
    uint64_t fieldId = field.getIdAttr().getInt();
    return pluginAPI.SetDeclChain(this->getIdAttr().getInt(), fieldId);
}
//===----------------------------------------------------------------------===//
// AddressOp

void AddressOp::build(OpBuilder &builder, OperationState &state, uint64_t id,
                      IDefineCode defCode, bool readOnly, Value operand, Type retType)
{
    state.addAttribute("id", builder.getI64IntegerAttr(id));
    state.addAttribute("defCode",
        builder.getI32IntegerAttr(static_cast<int32_t>(defCode)));
    state.addAttribute("readOnly", builder.getBoolAttr(readOnly));
    state.addOperands(operand);
    state.addTypes(retType);
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
    PinServer::PluginServer::GetInstance()->SetOpBuilder(&builder);
    return pluginAPI.CreateSSAOp(t);
}

Value SSAOp::Copy()
{
    PluginAPI::PluginServerAPI pluginAPI;
    static OpBuilder builder(this->getOperation());
    PinServer::PluginServer::GetInstance()->SetOpBuilder(&builder);
    return pluginAPI.CopySSAOp(this->getIdAttr().getInt());
}

Value SSAOp::GetCurrentDef()
{
    PluginAPI::PluginServerAPI pluginAPI;
    static OpBuilder builder(this->getOperation());
    PinServer::PluginServer::GetInstance()->SetOpBuilder(&builder);
    return pluginAPI.GetCurrentDefFromSSA(this->getIdAttr().getInt());
}

bool SSAOp::SetCurrentDef(Value def)
{
    uint64_t defId = GetValueId(def);
    PluginAPI::PluginServerAPI pluginAPI;
    if (pluginAPI.SetCurrentDefInSSA(this->getIdAttr().getInt(), defId)) {
        return true;
    }
    return false;
}

Operation* SSAOp::GetSSADefOperation()
{
    PluginAPI::PluginServerAPI pluginAPI;
    uint64_t definingId = this->getDefiningIdAttr().getInt();
    return pluginAPI.GetSSADefOperation(definingId);
}

// ===----------------------------------------------------------------------===//
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
    PinServer::PluginServer::GetInstance()->SetOpBuilder(&builder);
    return pluginAPI.CreateConstOp(value, retType);
}

// ===----------------------------------------------------------------------===//
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
// ListOp

void ListOp::build(OpBuilder &builder, OperationState &state, uint64_t id,
                  IDefineCode defCode, bool readOnly, bool hasPurpose,
                ArrayRef<Value> operands, Type retType)
{
    state.addAttribute("id", builder.getI64IntegerAttr(id));
    state.addAttribute("defCode",
                       builder.getI32IntegerAttr(static_cast<int32_t>(defCode)));
    state.addAttribute("readOnly", builder.getBoolAttr(readOnly));
    state.addAttribute("hasPurpose", builder.getBoolAttr(hasPurpose));
    state.addOperands(operands);
    state.addTypes(retType);
}

//===----------------------------------------------------------------------===//
// StrOp
void StrOp::build(OpBuilder &builder, OperationState &state, uint64_t id,
                  IDefineCode defCode, bool readOnly, StringRef str, Type retType)
{
    state.addAttribute("id", builder.getI64IntegerAttr(id));
    state.addAttribute("defCode",
                       builder.getI32IntegerAttr(static_cast<int32_t>(defCode)));
    state.addAttribute("str", builder.getStringAttr(str));
    state.addAttribute("readOnly", builder.getBoolAttr(readOnly));
    state.addTypes(retType);
}

//===----------------------------------------------------------------------===//
// ArrayOp

void ArrayOp::build(OpBuilder &builder, OperationState &state,
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

void CallOp::build(OpBuilder &builder, OperationState &state,
                   int64_t id, StringRef callee,
                   ArrayRef<Value> arguments)
{
    state.addAttribute("id", builder.getI64IntegerAttr(id));
    state.addOperands(arguments);
    //state.addAttribute("callee", builder.getSymbolRefAttr(callee));
    state.addAttribute("callee",
                     mlir::SymbolRefAttr::get(builder.getContext(), callee));
}

void CallOp::build(OpBuilder &builder, OperationState &state,
                   int64_t id, ArrayRef<Value> arguments)
{
    state.addAttribute("id", builder.getI64IntegerAttr(id));
    state.addOperands(arguments);
}

/// Return the callee of the generic call operation, this is required by the
/// call interface.
CallInterfaceCallable CallOp::getCallableForCallee()
{
    return (*this)->getAttrOfType<SymbolRefAttr>("callee");
}

/// Set the callee for the generic call operation, this is required by the call
/// interface.
void CallOp::setCalleeFromCallable(CallInterfaceCallable callee) {
    (*this)->setAttr("callee", callee.get<SymbolRefAttr>());
}

/// Get the argument operands to the called function, this is required by the
/// call interface.
Operation::operand_range CallOp::getArgOperands() { return getInputs(); }

bool CallOp::SetLHS(Value lhs)
{
    uint64_t lhsId = GetValueId(lhs);
    PluginAPI::PluginServerAPI pluginAPI;
    if (pluginAPI.SetLhsInCallOp(this->getIdAttr().getInt(), lhsId)) {
        (*this)->setOperand(0, lhs);
        return true;
    }
    return false;
}

void CallOp::build(OpBuilder &builder, OperationState &state,
                   Value func, ArrayRef<Value> arguments)
{
    Block *insertionBlock = builder.getInsertionBlock();
    assert(insertionBlock && "No InsertPoint is set for the OpBuilder.");
    PluginAPI::PluginServerAPI pluginAPI;
    uint64_t blockId = pluginAPI.FindBasicBlock(insertionBlock);
    PlaceholderOp funcOp = func.getDefiningOp<PlaceholderOp>();
    uint64_t funcId = funcOp.getIdAttr().getInt();
    vector<uint64_t> argIds;
    for (auto v : arguments) {
        uint64_t argId = GetValueId(v);
        argIds.push_back(argId);
    }
    uint64_t id = pluginAPI.CreateCallOp(blockId, funcId, argIds);
    state.addAttribute("id", builder.getI64IntegerAttr(id));
    state.addOperands(arguments);
    // FIXME: DEF_BUILTIN.
    //state.addAttribute("callee", builder.getSymbolRefAttr("ctzll"));
    // state.addAttribute("callee",
    //                  mlir::SymbolRefAttr::get(builder.getContext(), ctzll));
}

void CallOp::build(OpBuilder &builder, OperationState &state,
                   ArrayRef<Value> arguments)
{
    Block *insertionBlock = builder.getInsertionBlock();
    assert(insertionBlock && "No InsertPoint is set for the OpBuilder.");
    PluginAPI::PluginServerAPI pluginAPI;
    uint64_t blockId = pluginAPI.FindBasicBlock(insertionBlock);
    vector<uint64_t> argIds;
    for (auto v : arguments) {
        uint64_t argId = GetValueId(v);
        argIds.push_back(argId);
    }
    uint64_t funcId = 0;
    uint64_t id = pluginAPI.CreateCallOp(blockId, funcId, argIds);
    state.addAttribute("id", builder.getI64IntegerAttr(id));
    state.addOperands(arguments);
    //state.addAttribute("callee", builder.getSymbolRefAttr("ctzll"));
    // state.addAttribute("ctzll",
    //                  mlir::SymbolRefAttr::get(builder.getContext(), callee));
}

// ===----------------------------------------------------------------------===//
// CondOp

void CondOp::build(OpBuilder &builder, OperationState &state,
                   uint64_t id, uint64_t address, IComparisonCode condCode,
                   Value lhs, Value rhs, Block* tb, Block* fb, uint64_t tbaddr,
                   uint64_t fbaddr, Value trueLabel, Value falseLabel)
{
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
    Block *insertionBlock = builder.getInsertionBlock();
    assert(insertionBlock && "No InsertPoint is set for the OpBuilder.");
    PluginAPI::PluginServerAPI pluginAPI;
    uint64_t blockId = pluginAPI.FindBasicBlock(insertionBlock);
    uint64_t lhsId = GetValueId(lhs);
    uint64_t rhsId = GetValueId(rhs);
    uint64_t tbaddr = pluginAPI.FindBasicBlock(tb);
    uint64_t fbaddr = pluginAPI.FindBasicBlock(fb);
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

// ===----------------------------------------------------------------------===//
// PhiOp

void PhiOp::build(OpBuilder &builder, OperationState &state,
    ArrayRef<Value> operands, uint64_t id, uint32_t capacity, uint32_t nArgs)
{
    state.addAttribute("id", builder.getI64IntegerAttr(id));
    state.addAttribute("capacity", builder.getI32IntegerAttr(capacity));
    state.addAttribute("nArgs", builder.getI32IntegerAttr(nArgs));
    state.addOperands(operands);
}

Value PhiOp::GetResult()
{
    PluginAPI::PluginServerAPI pluginAPI;
    static OpBuilder builder(this->getOperation());
    PinServer::PluginServer::GetInstance()->SetOpBuilder(&builder);
    return pluginAPI.GetResultFromPhi(this->getIdAttr().getInt());
}

PhiOp PhiOp::CreatePhi(Value arg, Block *block)
{
    uint64_t argId = 0;
    if (arg != nullptr) {
        argId = GetValueId(arg);
    }
    PluginAPI::PluginServerAPI pluginAPI;
    uint64_t blockId = pluginAPI.FindBasicBlock(block);
    return pluginAPI.CreatePhiOp(argId, blockId);
}

bool PhiOp::AddArg(Value arg, Block *pred, Block *succ)
{
    uint64_t argId = GetValueId(arg);
    PluginAPI::PluginServerAPI pluginAPI;
    uint64_t predId = pluginAPI.FindBasicBlock(pred);
    uint64_t succId = pluginAPI.FindBasicBlock(succ);
    uint32_t nArg = pluginAPI.AddArgInPhiOp(this->getIdAttr().getInt(), argId, predId, succId);
    OpBuilder builder(this->getOperation());
    (*this)->insertOperands((*this)->getNumOperands(), {arg});
    (*this)->setAttr("nArgs", builder.getI32IntegerAttr(nArg));
    return true;
}
Value PhiOp::GetArgDef(int i)
{
    if (i >= (*this)->getNumOperands()) {
        return nullptr;
    }
    return getOperand(i);
}
// ===----------------------------------------------------------------------===//
// AssignOp

void AssignOp::build(OpBuilder &builder, OperationState &state,
                     ArrayRef<Value> operands,
                     uint64_t id, IExprCode exprCode)
{
    state.addAttribute("id", builder.getI64IntegerAttr(id));
    state.addAttribute("exprCode",
        builder.getI32IntegerAttr(static_cast<int32_t>(exprCode)));
    state.addOperands(operands);
}

void AssignOp::build(OpBuilder &builder, OperationState &state,
                     ArrayRef<Value> operands, IExprCode exprCode)
{
    Block *insertionBlock = builder.getInsertionBlock();
    assert(insertionBlock && "No InsertPoint is set for the OpBuilder.");
    PluginAPI::PluginServerAPI pluginAPI;
    uint64_t blockId = pluginAPI.FindBasicBlock(insertionBlock);
    vector<uint64_t> argIds;
    for (auto v : operands) {
        uint64_t argId = GetValueId(v);
        argIds.push_back(argId);
    }
    uint64_t id = pluginAPI.CreateAssignOp(blockId, exprCode, argIds);
    state.addAttribute("id", builder.getI64IntegerAttr(id));
    state.addAttribute("exprCode",
        builder.getI32IntegerAttr(static_cast<int32_t>(exprCode)));
    state.addOperands(operands);
}

//===----------------------------------------------------------------------===//
// NopOp
void NopOp::build(OpBuilder &builder, OperationState &state, uint64_t id)
{
    state.addAttribute("id", builder.getI64IntegerAttr(id));
}

//===----------------------------------------------------------------------===//
// EHElseOp
void EHElseOp::build(OpBuilder &builder, OperationState &state, uint64_t id, ArrayRef<uint64_t> nBody,
                    ArrayRef<uint64_t> eBody)
{
    state.addAttribute("id", builder.getI64IntegerAttr(id));
    llvm::SmallVector<mlir::Attribute, 4> nbodyattrs, ebodyattrs;
    for (auto item : nBody) {
        nbodyattrs.push_back(builder.getI64IntegerAttr(item));
    }
    for (auto item : eBody) {
        ebodyattrs.push_back(builder.getI64IntegerAttr(item));
    }
    state.addAttribute("nBody", builder.getArrayAttr(nbodyattrs));
    state.addAttribute("eBody", builder.getArrayAttr(ebodyattrs));
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
// DebugOp

void DebugOp::build(OpBuilder &builder, OperationState &state,
                    uint64_t id)
{
    state.addAttribute("id", builder.getI64IntegerAttr(id));
}

// ===----------------------------------------------------------------------===//
// AsmOp
void AsmOp::build(OpBuilder &builder, OperationState &state,
                   uint64_t id, StringRef statement, uint32_t nInputs, uint32_t nOutputs,
                   uint32_t nClobbers, ArrayRef<Value> operands)
{
    state.addAttribute("id", builder.getI64IntegerAttr(id));
    //state.addAttribute("statement", builder.getSymbolRefAttr(statement));
    state.addAttribute("callee",
                     mlir::SymbolRefAttr::get(builder.getContext(), statement));
    state.addAttribute("nInputs", builder.getI32IntegerAttr(nInputs));
    state.addAttribute("nOutputs", builder.getI32IntegerAttr(nOutputs));
    state.addAttribute("nClobbers", builder.getI32IntegerAttr(nClobbers));
    state.addOperands(operands);
}

//===----------------------------------------------------------------------===//
// SwitchOp
void SwitchOp::build(OpBuilder &builder, OperationState &state, uint64_t id,
                     Value index, uint64_t address, Value defaultLabel,
                     ArrayRef<Value> args, Block* defaultDest,
                     uint64_t defaultaddr, ArrayRef<Block*> caseDest,
                     ArrayRef<uint64_t> caseaddr)
{
    state.addAttribute("id", builder.getI64IntegerAttr(id));
    state.addAttribute("address", builder.getI64IntegerAttr(address));
    state.addAttribute("defaultaddr", builder.getI64IntegerAttr(defaultaddr));
    llvm::SmallVector<mlir::Attribute, 4> attributes;
    for (size_t i = 0; i < caseaddr.size(); ++i) {
        attributes.push_back(builder.getI64IntegerAttr(caseaddr[i]));
    }
    state.addAttribute("caseaddrs", builder.getArrayAttr(attributes));
    state.addOperands(index);
    state.addOperands(defaultLabel);
    state.addOperands(args);
    state.addSuccessors(defaultDest);
    state.addSuccessors(caseDest);
}
// FallThroughOp

void FallThroughOp::build(OpBuilder &builder, OperationState &state,
                          uint64_t address, Block* dest, uint64_t destaddr)
{
    state.addAttribute("address", builder.getI64IntegerAttr(address));
    state.addAttribute("destaddr", builder.getI64IntegerAttr(destaddr));
    state.addSuccessors(dest);
}

void FallThroughOp::build(OpBuilder &builder, OperationState &state,
                          Block* src, Block* dest)
{
    PluginAPI::PluginServerAPI pluginAPI;
    uint64_t address = pluginAPI.FindBasicBlock(src);
    uint64_t destaddr = pluginAPI.FindBasicBlock(dest);

    PluginAPI::ControlFlowAPI cfgAPI;
    cfgAPI.CreateFallthroughOp(address, destaddr);
    state.addAttribute("address", builder.getI64IntegerAttr(address));
    state.addAttribute("destaddr", builder.getI64IntegerAttr(destaddr));
    state.addSuccessors(dest);
}

// ===----------------------------------------------------------------------===//
// RetOp

void RetOp::build(OpBuilder &builder, OperationState &state, uint64_t address)
{
    state.addAttribute("address", builder.getI64IntegerAttr(address));
}

//===----------------------------------------------------------------------===//
// GotoOp

void GotoOp::build(OpBuilder &builder, OperationState &state, uint64_t id, uint64_t address,
Value dest, Block* success, uint64_t successaddr)
{
    state.addAttribute("id", builder.getI64IntegerAttr(id));
    state.addAttribute("address", builder.getI64IntegerAttr(address));
    state.addAttribute("successaddr", builder.getI64IntegerAttr(successaddr));
    state.addOperands(dest);
    state.addSuccessors(success);
}

//===----------------------------------------------------------------------===//
// LabelOp
void LabelOp::build(OpBuilder &builder, OperationState &state, uint64_t id, Value label)
{
    state.addAttribute("id", builder.getI64IntegerAttr(id));
    state.addOperands(label);
}

//==-----------------------------------------------------------------------===//
// TransactionOp
void TransactionOp::build(OpBuilder &builder, OperationState &state, uint64_t id, uint64_t address,
                        ArrayRef<uint64_t> stmtaddr, Value labelNorm, Value labelUninst, Value labelOver, Block* fallthrough,
                        uint64_t fallthroughaddr, Block* abort, uint64_t abortaddr)
{
    state.addAttribute("id", builder.getI64IntegerAttr(id));
    state.addAttribute("address", builder.getI64IntegerAttr(address));
    llvm::SmallVector<mlir::Attribute, 4> attributes;
    for (size_t i = 0; i < stmtaddr.size(); ++i) {
        attributes.push_back(builder.getI64IntegerAttr(stmtaddr[i]));
    }
    state.addAttribute("stmtaddr", builder.getArrayAttr(attributes));
    state.addOperands({labelNorm, labelUninst, labelOver});
    state.addSuccessors(fallthrough);
    state.addAttribute("fallthroughaddr", builder.getI64IntegerAttr(fallthroughaddr));
    state.addSuccessors(abort);
    state.addAttribute("abortaddr", builder.getI64IntegerAttr(abortaddr));
}

//===----------------------------------------------------------------------===//
// ResxOp

void ResxOp::build(OpBuilder &builder, OperationState &state, uint64_t id, uint64_t address, uint64_t region)
{
    state.addAttribute("id", builder.getI64IntegerAttr(id));
    state.addAttribute("address", builder.getI64IntegerAttr(address));
    state.addAttribute("region", builder.getI64IntegerAttr(region));
}

//===----------------------------------------------------------------------===//
// EHMntOp
void EHMntOp::build(OpBuilder &builder, OperationState &state, uint64_t id, Value decl)
{
    state.addAttribute("id", builder.getI64IntegerAttr(id));
    state.addOperands(decl);
}

//===----------------------------------------------------------------------===//
// EHDispatchOp

void EHDispatchOp::build(OpBuilder &builder, OperationState &state, uint64_t id, uint64_t address, uint64_t region,
                        ArrayRef<Block*> ehHandlers, ArrayRef<uint64_t> ehHandlersaddrs)
{
    state.addAttribute("id", builder.getI64IntegerAttr(id));
    state.addAttribute("address", builder.getI64IntegerAttr(address));
    state.addAttribute("region", builder.getI64IntegerAttr(region));
    state.addSuccessors(ehHandlers);
    llvm::SmallVector<mlir::Attribute, 4> attributes;
    for (size_t i = 0; i < ehHandlersaddrs.size(); ++i) {
        attributes.push_back(builder.getI64IntegerAttr(ehHandlersaddrs[i]));
    }
    state.addAttribute("ehHandlersaddrs", builder.getArrayAttr(attributes));
}
//===----------------------------------------------------------------------===//
// BindOp

void BindOp::build(OpBuilder &builder, OperationState &state, uint64_t id, Value vars, ArrayRef<uint64_t> body,
                    Value block)
{
    state.addAttribute("id", builder.getI64IntegerAttr(id));
    state.addOperands({vars, block});
    llvm::SmallVector<mlir::Attribute, 4> attributes;
    for (size_t i = 0; i < body.size(); ++i) {
        attributes.push_back(builder.getI64IntegerAttr(body[i]));
    }
    state.addAttribute("body", builder.getArrayAttr(attributes));
}

//===----------------------------------------------------------------------===//
// TryOp

void TryOp::build(OpBuilder &builder, OperationState &state, uint64_t id, ArrayRef<uint64_t> eval,
                ArrayRef<uint64_t> cleanup, uint64_t kind)
{
    state.addAttribute("id", builder.getI64IntegerAttr(id));
    llvm::SmallVector<mlir::Attribute, 4> attributes;
    for (size_t i = 0; i < eval.size(); ++i) {
        attributes.push_back(builder.getI64IntegerAttr(eval[i]));
    }
    state.addAttribute("eval", builder.getArrayAttr(attributes));
    attributes.clear();
    for (size_t i = 0; i < cleanup.size(); ++i) {
        attributes.push_back(builder.getI64IntegerAttr(cleanup[i]));
    }
    state.addAttribute("cleanup", builder.getArrayAttr(attributes));
    state.addAttribute("kind", builder.getI64IntegerAttr(kind));
}

//===----------------------------------------------------------------------===//
// CatchOp

void CatchOp::build(OpBuilder &builder, OperationState &state, uint64_t id, Value types, ArrayRef<uint64_t> handler)
{
    state.addAttribute("id", builder.getI64IntegerAttr(id));
    state.addOperands(types);
    llvm::SmallVector<mlir::Attribute, 4> attributes;
    for (size_t i = 0; i < handler.size(); ++i) {
        attributes.push_back(builder.getI64IntegerAttr(handler[i]));
    }
    state.addAttribute("handler", builder.getArrayAttr(attributes));
}
//===----------------------------------------------------------------------===//
// TableGen'd op method definitions
// ===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "Dialect/PluginOps.cpp.inc"
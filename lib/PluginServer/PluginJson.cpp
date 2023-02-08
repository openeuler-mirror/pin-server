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

   Author: Mingchuan Wu and Yancheng Li
   Create: 2022-08-18
   Description:
    This file contains the implementation of the PluginJson class.
*/

#include "PluginServer/PluginJson.h"
#include "PluginServer/PluginServer.h"

namespace PinJson {
using namespace PinServer;
using namespace mlir::Plugin;

static uintptr_t GetID(Json::Value node)
{
    string id = node.asString();
    return atol(id.c_str());
}

static void JsonGetAttributes(Json::Value node, map<string, string>& attributes)
{
    Json::Value::Members attMember = node.getMemberNames();
    for (unsigned int i = 0; i < attMember.size(); i++) {
        string key = attMember[i];
        string value = node[key.c_str()].asString();
        attributes[key] = value;
    }
}

Json::Value PluginJson::TypeJsonSerialize (PluginIR::PluginTypeBase& type)
{
    Json::Value root;
    Json::Value operationObj;
    Json::Value item;

    uint64_t ReTypeId;
    uint64_t ReTypeWidth;

    ReTypeId = static_cast<uint64_t>(type.getPluginTypeID());
    item["id"] = std::to_string(ReTypeId);

    if (auto elemTy = type.dyn_cast<PluginIR::PluginPointerType>()) {
        auto baseTy = elemTy.getElementType().dyn_cast<PluginIR::PluginTypeBase>();
        item["elementType"] = TypeJsonSerialize(baseTy);
        if (elemTy.isReadOnlyElem()) {
            item["elemConst"] = "1";
        } else {
            item["elemConst"] = "0";
        }
    }

    if (type.getPluginIntOrFloatBitWidth() != 0) {
        ReTypeWidth = type.getPluginIntOrFloatBitWidth();
        item["width"] = std::to_string(ReTypeWidth);
    }

    if (type.isSignedPluginInteger()) {
        item["signed"] = "1";
    }

    if (type.isUnsignedPluginInteger()) {
        item["signed"] = "0";
    }

    root["type"] = item;
    return root;
}

mlir::Value PluginJson::ValueJsonDeSerialize(Json::Value valueJson)
{
    uint64_t opId = GetID(valueJson["id"]);
    if (PluginServer::GetInstance()->HaveValue(opId))
        return PluginServer::GetInstance()->GetValue(opId);

    IDefineCode defCode = IDefineCode(atoi(valueJson["defCode"].asString().c_str()));
    mlir::Type retType = TypeJsonDeSerialize(valueJson["retType"].toStyledString());
    bool readOnly = GetID(valueJson["readOnly"]);
    mlir::Value opValue;
    mlir::OpBuilder *opBuilder = PluginServer::GetInstance()->GetOpBuilder();
    switch (defCode) {
        case IDefineCode::MemRef : {
            opValue = MemRefDeSerialize(valueJson.toStyledString());
            break;
        }
        case IDefineCode::IntCST : {
            uint64_t init = GetID(valueJson["value"]);
            // FIXME : AnyAttr!
            mlir::Attribute initAttr = opBuilder->getI64IntegerAttr(init);
            opValue = opBuilder->create<ConstOp>(
                    opBuilder->getUnknownLoc(), opId, IDefineCode::IntCST,
                    readOnly, initAttr, retType);
            break;
        }
        case IDefineCode::SSA : {
            opValue = SSAOpJsonDeSerialize(valueJson.toStyledString());
            break;
        }
        default: {
            opValue = opBuilder->create<PlaceholderOp>(
                    opBuilder->getUnknownLoc(), opId, defCode, readOnly, retType);
            break;
        }
    }
    PluginServer::GetInstance()->InsertValue(opId, opValue);
    return opValue;
}

mlir::Value PluginJson::MemRefDeSerialize(const string& data)
{
    Json::Value root;
    Json::Reader reader;
    reader.parse(data, root);
    uint64_t id = GetID(root["id"]);
    bool readOnly = (bool)atoi(root["readOnly"].asString().c_str());
    mlir::Value base = ValueJsonDeSerialize(root["base"]);
    mlir::Value offset = ValueJsonDeSerialize(root["offset"]);
    mlir::Type retType = TypeJsonDeSerialize(root["retType"].toStyledString().c_str());
    mlir::OpBuilder *opBuilder = PluginServer::GetInstance()->GetOpBuilder();
    mlir::Value memRef = opBuilder->create<MemOp>(
        opBuilder->getUnknownLoc(), id, IDefineCode::MemRef,
        readOnly, base, offset, retType);
    return memRef;
}

mlir::Value PluginJson::SSAOpJsonDeSerialize(const string& data)
{
    Json::Value node;
    Json::Reader reader;
    reader.parse(data, node);

    uint64_t id = GetID(node["id"]);
    bool readOnly = (bool)atoi(node["readOnly"].asString().c_str());
    uint64_t nameVarId = GetID(node["nameVarId"]);
    uint64_t ssaParmDecl = GetID(node["ssaParmDecl"]);
    uint64_t version = GetID(node["version"]);
    uint64_t definingId = GetID(node["definingId"]);
    mlir::Type retType = TypeJsonDeSerialize(node["retType"].toStyledString().c_str());
    mlir::OpBuilder *opBuilder = PluginServer::GetInstance()->GetOpBuilder();
    mlir::Value ret = opBuilder->create<SSAOp>(opBuilder->getUnknownLoc(),
                                               id, IDefineCode::SSA, readOnly,
                                               nameVarId, ssaParmDecl, version,
                                               definingId, retType);
    return ret;
}

void PluginJson::GetAttributes(Json::Value node, map<string, string>& attributes)
{
    Json::Value::Members attMember = node.getMemberNames();
    for (unsigned int i = 0; i < attMember.size(); i++) {
        string key = attMember[i];
        string value = node[key.c_str()].asString();
        attributes[key] = value;
    }
}

bool PluginJson::ProcessBlock(mlir::Block* block, mlir::Region& rg, const Json::Value& blockJson)
{
    if (blockJson.isNull()) {
        return false;
    }
    // todo process func return type
    // todo isDeclaration

    // process each stmt
    mlir::OpBuilder *opBuilder = PluginServer::GetInstance()->GetOpBuilder();
    opBuilder->setInsertionPointToStart(block);
    Json::Value::Members opMember = blockJson.getMemberNames();
    for (size_t opIdx = 0; opIdx < opMember.size(); opIdx++) {
        string baseOpKey = "Operation" + std::to_string(opIdx);
        Json::Value opJson = blockJson[baseOpKey];
        if (opJson.isNull()) continue;
        string opCode = opJson["OperationName"].asString();
        if (opCode == PhiOp::getOperationName().str()) {
            PhiOpJsonDeSerialize(opJson.toStyledString());
        } else if (opCode == CallOp::getOperationName().str()) {
            CallOpJsonDeSerialize(opJson.toStyledString());
        } else if (opCode == AssignOp::getOperationName().str()) {
            AssignOpJsonDeSerialize(opJson.toStyledString());
        } else if (opCode == CondOp::getOperationName().str()) {
            CondOpJsonDeSerialize(opJson.toStyledString());
        } else if (opCode == RetOp::getOperationName().str()) {
            RetOpJsonDeSerialize(opJson.toStyledString());
        } else if (opCode == FallThroughOp::getOperationName().str()) {
            FallThroughOpJsonDeSerialize(opJson.toStyledString());
        } else if (opCode == BaseOp::getOperationName().str()) {
            uint64_t opID = GetID(opJson["id"]);
            opBuilder->create<BaseOp>(opBuilder->getUnknownLoc(), opID, opCode);
        }
    }
    return true;
}

void PluginJson::IntegerDeSerialize(const string& data, int64_t& result)
{
    Json::Value root;
    Json::Reader reader;
    reader.parse(data, root);

    result = root["integerData"].asInt64();
}

void PluginJson::StringDeSerialize(const string& data, string& result)
{
    Json::Value root;
    Json::Reader reader;
    reader.parse(data, root);

    result = root["stringData"].asString();
}

void PluginJson::FuncOpJsonDeSerialize(
    const string& data, vector<mlir::Plugin::FunctionOp>& funcOpData)
{
    Json::Value root;
    Json::Reader reader;
    Json::Value node;
    reader.parse(data, root);

    Json::Value::Members operation = root.getMemberNames();

    mlir::OpBuilder opBuilder(PluginServer::GetInstance()->GetContext());

    for (size_t iter = 0; iter < operation.size(); iter++) {
        string operationKey = "FunctionOp" + std::to_string(iter);
        node = root[operationKey];
        int64_t id = GetID(node["id"]);
        Json::Value attributes = node["attributes"];
        map<string, string> funcAttributes;
        JsonGetAttributes(attributes, funcAttributes);
        bool declaredInline = false;
        if (funcAttributes["declaredInline"] == "1") declaredInline = true;
        auto location = opBuilder.getUnknownLoc();
        FunctionOp fOp = opBuilder.create<FunctionOp>(
                location, id, funcAttributes["funcName"], declaredInline);
        mlir::Region &bodyRegion = fOp.bodyRegion();
        Json::Value regionJson = node["region"];
        Json::Value::Members bbMember = regionJson.getMemberNames();
        // We must create Blocks before process ops
        for (size_t bbIdx = 0; bbIdx < bbMember.size(); bbIdx++) {
            string blockKey = "block" + std::to_string(bbIdx);
            Json::Value blockJson = regionJson[blockKey];
            mlir::Block* block = opBuilder.createBlock(&bodyRegion);
            PluginServer::GetInstance()->InsertCreatedBlock(
                GetID(blockJson["address"]), block);
        }
        
        for (size_t bbIdx = 0; bbIdx < bbMember.size(); bbIdx++) {
            string blockKey = "block" + std::to_string(bbIdx);
            Json::Value blockJson = regionJson[blockKey];
            uint64_t bbAddress = GetID(blockJson["address"]);
            mlir::Block* block = PluginServer::GetInstance()->FindBlock(bbAddress);
            ProcessBlock(block, bodyRegion, blockJson["ops"]);
        }
        funcOpData.push_back(fOp);
        opBuilder.setInsertionPointAfter(fOp.getOperation());
    }
}

PluginIR::PluginTypeBase PluginJson::TypeJsonDeSerialize(const string& data)
{
    Json::Value root;
    Json::Reader reader;
    Json::Value node;
    reader.parse(data, root);
    PluginIR::PluginTypeBase baseType;
    Json::Value type = root["type"];
    uint64_t id = GetID(type["id"]);
    PluginIR::PluginTypeID PluginTypeId = static_cast<PluginIR::PluginTypeID>(id);
    if (type["signed"] && (id >= static_cast<uint64_t>(PluginIR::UIntegerTy1ID) &&
        id <= static_cast<uint64_t>(PluginIR::IntegerTy64ID))) {
        string s = type["signed"].asString();
        uint64_t width = GetID(type["width"]);
        if (s == "1") {
            baseType = PluginIR::PluginIntegerType::get(
                PluginServer::GetInstance()->GetContext(), width, PluginIR::PluginIntegerType::Signed);
        } else {
            baseType = PluginIR::PluginIntegerType::get(
                PluginServer::GetInstance()->GetContext(), width, PluginIR::PluginIntegerType::Unsigned);
        }
    } else if (type["width"] && (id == static_cast<uint64_t>(PluginIR::FloatTyID) ||
             id == static_cast<uint64_t>(PluginIR::DoubleTyID))) {
        uint64_t width = GetID(type["width"]);
        baseType = PluginIR::PluginFloatType::get(PluginServer::GetInstance()->GetContext(), width);
    } else if (id == static_cast<uint64_t>(PluginIR::PointerTyID)) {
        mlir::Type elemTy = TypeJsonDeSerialize(type["elementType"].toStyledString());
        baseType = PluginIR::PluginPointerType::get(
            PluginServer::GetInstance()->GetContext(), elemTy, type["elemConst"].asString() == "1" ? 1 : 0);
    } else {
        if (PluginTypeId == PluginIR::VoidTyID) {
            baseType = PluginIR::PluginVoidType::get(PluginServer::GetInstance()->GetContext());
        }
        if (PluginTypeId == PluginIR::BooleanTyID) {
            baseType = PluginIR::PluginBooleanType::get(PluginServer::GetInstance()->GetContext());
        }
        if (PluginTypeId == PluginIR::UndefTyID) {
            baseType = PluginIR::PluginUndefType::get(PluginServer::GetInstance()->GetContext());
        }
    }
    return baseType;
}

void PluginJson::LocalDeclOpJsonDeSerialize(
    const string& data, vector<mlir::Plugin::LocalDeclOp>& decls)
{
    Json::Value root;
    Json::Reader reader;
    Json::Value node;
    reader.parse(data, root);
    Json::Value::Members operation = root.getMemberNames();
    mlir::OpBuilder opBuilder(PluginServer::GetInstance()->GetContext());
    for (size_t iter = 0; iter < operation.size(); iter++) {
        string operationKey = "localDecl" + std::to_string(iter);
        node = root[operationKey];
        int64_t id = GetID(node["id"]);
        Json::Value attributes = node["attributes"];
        map<string, string> declAttributes;
        JsonGetAttributes(attributes, declAttributes);
        string symName = declAttributes["symName"];
        uint64_t typeID = atol(declAttributes["typeID"].c_str());
        uint64_t typeWidth = atol(declAttributes["typeWidth"].c_str());
        auto location = opBuilder.getUnknownLoc();
        LocalDeclOp op = opBuilder.create<LocalDeclOp>(
                location, id, symName, typeID, typeWidth);
        decls.push_back(op);
    }
}

void PluginJson::LoopOpsJsonDeSerialize(
    const string& data, vector<mlir::Plugin::LoopOp>& loops)
{
    Json::Value root;
    Json::Reader reader;
    Json::Value node;
    reader.parse(data, root);
    Json::Value::Members operation = root.getMemberNames();
    mlir::OpBuilder builder(PluginServer::GetInstance()->GetContext());
    for (size_t iter = 0; iter < operation.size(); iter++) {
        string operationKey = "loopOp" + std::to_string(iter);
        node = root[operationKey];
        int64_t id = GetID(node["id"]);
        Json::Value attributes = node["attributes"];
        map<string, string> loopAttributes;
        JsonGetAttributes(attributes, loopAttributes);
        uint32_t index = GetID(attributes["index"]);
        uint64_t innerId = atol(loopAttributes["innerLoopId"].c_str());
        uint64_t outerId = atol(loopAttributes["outerLoopId"].c_str());
        uint32_t numBlock = atoi(loopAttributes["numBlock"].c_str());
        auto location = builder.getUnknownLoc();
        LoopOp op = builder.create<LoopOp>(
                location, id, index, innerId, outerId, numBlock);
        loops.push_back(op);
    }
}

LoopOp PluginJson::LoopOpJsonDeSerialize(const string& data)
{
    Json::Value root;
    Json::Reader reader;
    reader.parse(data, root);
    mlir::OpBuilder builder(PluginServer::GetInstance()->GetContext());
    uint64_t id = GetID(root["id"]);
    Json::Value attributes = root["attributes"];
    uint32_t index = GetID(attributes["index"]);
    uint64_t innerLoopId = GetID(attributes["innerLoopId"]);
    uint64_t outerLoopId = GetID(attributes["outerLoopId"]);
    uint32_t numBlock = GetID(attributes["numBlock"]);
    auto location = builder.getUnknownLoc();
    return builder.create<LoopOp>(
            location, id, index, innerLoopId, outerLoopId, numBlock);
}

void PluginJson::EdgesJsonDeSerialize(
    const string& data, vector<std::pair<mlir::Block*, mlir::Block*>>& edges)
{
    Json::Value root;
    Json::Reader reader;
    Json::Value node;
    reader.parse(data, root);
    Json::Value::Members operation = root.getMemberNames();
    for (size_t iter = 0; iter < operation.size(); iter++) {
        string operationKey = "edge" + std::to_string(iter);
        node = root[operationKey];
        uint64_t src = GetID(node["src"]);
        uint64_t dest = GetID(node["dest"]);
        std::pair<mlir::Block*, mlir::Block*> e;
        if (PluginServer::GetInstance()->HaveBlock(src)) {
            e.first = PluginServer::GetInstance()->FindBlock(src);
        } else {
            e.first = nullptr;
        }

        if (PluginServer::GetInstance()->HaveBlock(dest)) {
            e.second = PluginServer::GetInstance()->FindBlock(dest);
        } else {
            e.second = nullptr;
        }

        edges.push_back(e);
    }
}

void PluginJson::EdgeJsonDeSerialize(
    const string& data, std::pair<mlir::Block*, mlir::Block*>& edge)
{
    Json::Value root;
    Json::Reader reader;
    reader.parse(data, root);
    uint64_t src = GetID(root["src"]);
    uint64_t dest = GetID(root["dest"]);
    if (PluginServer::GetInstance()->HaveBlock(src)) {
        edge.first = PluginServer::GetInstance()->FindBlock(src);
    } else {
        edge.first = nullptr;
    }

    if (PluginServer::GetInstance()->HaveBlock(dest)) {
        edge.second = PluginServer::GetInstance()->FindBlock(dest);
    } else {
        edge.second = nullptr;
    }
}

void PluginJson::IdsJsonDeSerialize(
    const string& data, vector<uint64_t>& idsResult)
{
    Json::Value root;
    Json::Reader reader;
    Json::Value node;
    reader.parse(data, root);
    Json::Value::Members operation = root.getMemberNames();
    for (size_t iter = 0; iter < operation.size(); iter++) {
        string operationKey = "block" + std::to_string(iter);
        node = root[operationKey];
        uint64_t id = GetID(node["id"]);
        idsResult.push_back(id);
    }
}

mlir::Operation *PluginJson::CallOpJsonDeSerialize(const string& data)
{
    Json::Value node;
    Json::Reader reader;
    reader.parse(data, node);
    Json::Value operandJson = node["operands"];
    Json::Value::Members operandMember = operandJson.getMemberNames();
    llvm::SmallVector<mlir::Value, 4> ops;
    for (size_t opIter = 0; opIter < operandMember.size(); opIter++) {
        string key = "input" + std::to_string(opIter);
        mlir::Value opValue = ValueJsonDeSerialize(operandJson[key.c_str()]);
        ops.push_back(opValue);
    }
    int64_t id = GetID(node["id"]);
    mlir::StringRef callName(node["callee"].asString());
    mlir::OpBuilder *opBuilder = PluginServer::GetInstance()->GetOpBuilder();
    CallOp op = opBuilder->create<CallOp>(opBuilder->getUnknownLoc(),
                                         id, callName, ops);
    return op.getOperation();
}

mlir::Operation *PluginJson::CondOpJsonDeSerialize(const string& data)
{
    Json::Value node;
    Json::Reader reader;
    reader.parse(data, node);
    mlir::Value LHS = ValueJsonDeSerialize(node["lhs"]);
    mlir::Value RHS = ValueJsonDeSerialize(node["rhs"]);
    mlir::Value trueLabel = nullptr;
    mlir::Value falseLabel = nullptr;
    int64_t id = GetID(node["id"]);
    int64_t address = GetID(node["address"]);
    int64_t tbaddr = GetID(node["tbaddr"]);
    int64_t fbaddr = GetID(node["fbaddr"]);
    mlir::Block* tb = PluginServer::GetInstance()->FindBlock(tbaddr);
    mlir::Block* fb = PluginServer::GetInstance()->FindBlock(fbaddr);
    IComparisonCode iCode = IComparisonCode(
        atoi(node["condCode"].asString().c_str()));
    mlir::OpBuilder *opBuilder = PluginServer::GetInstance()->GetOpBuilder();
    CondOp op = opBuilder->create<CondOp>(
        opBuilder->getUnknownLoc(), id, address, iCode, LHS,
        RHS, tb, fb, tbaddr, fbaddr, trueLabel, falseLabel);
    return op.getOperation();
}

mlir::Operation *PluginJson::RetOpJsonDeSerialize(const string& data)
{
    Json::Value node;
    Json::Reader reader;
    reader.parse(data, node);
    int64_t address = GetID(node["address"]);
    mlir::OpBuilder *opBuilder = PluginServer::GetInstance()->GetOpBuilder();
    RetOp op = opBuilder->create<RetOp>(opBuilder->getUnknownLoc(), address);
    return op.getOperation();
}

mlir::Operation *PluginJson::FallThroughOpJsonDeSerialize(const string& data)
{
    Json::Value node;
    Json::Reader reader;
    reader.parse(data, node);
    int64_t address = GetID(node["address"]);
    int64_t destaddr = GetID(node["destaddr"]);
    mlir::Block* succ = PluginServer::GetInstance()->FindBlock(destaddr);
    mlir::OpBuilder *opBuilder = PluginServer::GetInstance()->GetOpBuilder();
    FallThroughOp op = opBuilder->create<FallThroughOp>(opBuilder->getUnknownLoc(),
                                                        address, succ, destaddr);
    return op.getOperation();
}

mlir::Operation *PluginJson::AssignOpJsonDeSerialize(const string& data)
{
    Json::Value node;
    Json::Reader reader;
    reader.parse(data, node);
    Json::Value operandJson = node["operands"];
    Json::Value::Members operandMember = operandJson.getMemberNames();
    llvm::SmallVector<mlir::Value, 4> ops;
    for (size_t opIter = 0; opIter < operandMember.size(); opIter++) {
        string key = "input" + std::to_string(opIter);
        mlir::Value opValue = ValueJsonDeSerialize(operandJson[key.c_str()]);
        ops.push_back(opValue);
    }
    uint64_t id = GetID(node["id"]);
    IExprCode iCode = IExprCode(atoi(node["exprCode"].asString().c_str()));
    mlir::OpBuilder *opBuilder = PluginServer::GetInstance()->GetOpBuilder();
    AssignOp op = opBuilder->create<AssignOp>(opBuilder->getUnknownLoc(),
                                             ops, id, iCode);
    PluginServer::GetInstance()->InsertDefOperation(id, op.getOperation());
    return op.getOperation();
}

mlir::Operation *PluginJson::PhiOpJsonDeSerialize(const string& data)
{
    Json::Value node;
    Json::Reader reader;
    reader.parse(data, node);
    Json::Value operandJson = node["operands"];
    Json::Value::Members operandMember = operandJson.getMemberNames();
    llvm::SmallVector<mlir::Value, 4> ops;
    for (size_t opIter = 0; opIter < operandMember.size(); opIter++) {
        string key = "input" + std::to_string(opIter);
        mlir::Value opValue = ValueJsonDeSerialize(operandJson[key.c_str()]);
        ops.push_back(opValue);
    }
    uint64_t id = GetID(node["id"]);
    uint32_t capacity = GetID(node["capacity"]);
    uint32_t nArgs = GetID(node["nArgs"]);
    mlir::OpBuilder *opBuilder = PluginServer::GetInstance()->GetOpBuilder();
    PhiOp op = opBuilder->create<PhiOp>(opBuilder->getUnknownLoc(),
                                       ops, id, capacity, nArgs);
    
    PluginServer::GetInstance()->InsertDefOperation(id, op.getOperation());
    return op.getOperation();
}

void PluginJson::GetPhiOpsJsonDeSerialize(
    const string& data, vector<mlir::Operation *>& opData)
{
    opData.clear();
    Json::Value root;
    Json::Reader reader;
    Json::Value node;
    reader.parse(data, root);

    Json::Value::Members operation = root.getMemberNames();
    for (size_t iter = 0; iter < operation.size(); iter++) {
        string operationKey = "operation" + std::to_string(iter);
        node = root[operationKey];
        opData.push_back(PhiOpJsonDeSerialize(node.toStyledString()));
    }
}

void PluginJson::OpJsonDeSerialize(
    const string& data, vector<mlir::Operation *>& opData)
{
    Json::Value opJson;
    Json::Reader reader;
    Json::Value node;
    reader.parse(data, opJson);
    string opCode = opJson["OperationName"].asString();
    if (opCode == PhiOp::getOperationName().str()) {
        opData.push_back(PhiOpJsonDeSerialize(opJson.toStyledString()));
    } else if (opCode == CallOp::getOperationName().str()) {
        opData.push_back(CallOpJsonDeSerialize(opJson.toStyledString()));
    } else if (opCode == AssignOp::getOperationName().str()) {
        opData.push_back(AssignOpJsonDeSerialize(opJson.toStyledString()));
    } else if (opCode == CondOp::getOperationName().str()) {
        opData.push_back(CondOpJsonDeSerialize(opJson.toStyledString()));
    } else if (opCode == RetOp::getOperationName().str()) {
        opData.push_back(RetOpJsonDeSerialize(opJson.toStyledString()));
    } else if (opCode == FallThroughOp::getOperationName().str()) {
        opData.push_back(FallThroughOpJsonDeSerialize(opJson.toStyledString()));
    } else if (opCode == BaseOp::getOperationName().str()) {
        uint64_t opID = GetID(opJson["id"]);
        mlir::OpBuilder *opBuilder = PluginServer::GetInstance()->GetOpBuilder();
        opBuilder->create<BaseOp>(opBuilder->getUnknownLoc(), opID, opCode);
    }
}
} // namespace PinJson
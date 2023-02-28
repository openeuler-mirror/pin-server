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
using namespace mlir;
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

Json::Value PluginJson::TypeJsonSerialize (PluginIR::PluginTypeBase type)
{
    Json::Value root;
    Json::Value operationObj;
    Json::Value item;

    uint64_t ReTypeId;
    uint64_t ReTypeWidth;

    ReTypeId = static_cast<uint64_t>(type.getPluginTypeID());
    item["id"] = std::to_string(ReTypeId);

    if (auto Ty = type.dyn_cast<PluginIR::PluginStructType>()) {
        std::string tyName = Ty.getName();
        item["structtype"] = tyName;
        size_t paramIndex = 0;
        ArrayRef<Type> paramsType = Ty.getBody();
        for (auto ty :paramsType) {
            std::string paramStr = "elemType" + std::to_string(paramIndex++);
            item["structelemType"][paramStr] = TypeJsonSerialize(ty.dyn_cast<PluginIR::PluginTypeBase>());
        }
        paramIndex = 0;
        ArrayRef<std::string> paramsNames = Ty.getElementNames();
        for (auto name :paramsNames) {
            std::string paramStr = "elemName" + std::to_string(paramIndex++);
            item["structelemName"][paramStr] = name;
        }
    }

    if (auto Ty = type.dyn_cast<PluginIR::PluginFunctionType>()) {
        auto fnrestype = Ty.getReturnType().dyn_cast<PluginIR::PluginTypeBase>();
        item["fnreturntype"] = TypeJsonSerialize(fnrestype);
        size_t paramIndex = 0;
        ArrayRef<Type> paramsType = Ty.getParams();
        for (auto ty : Ty.getParams()) {
            string paramStr = "argType" + std::to_string(paramIndex++);
            item["fnargsType"][paramStr] = TypeJsonSerialize(ty.dyn_cast<PluginIR::PluginTypeBase>());
        }
    }

    if (auto Ty = type.dyn_cast<PluginIR::PluginArrayType>()) {
        auto elemTy = Ty.getElementType().dyn_cast<PluginIR::PluginTypeBase>();
        item["elementType"] = TypeJsonSerialize(elemTy);
        uint64_t elemNum = Ty.getNumElements();
        item["arraysize"] = std::to_string(elemNum);
    }

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
        case IDefineCode::LIST : {
            opValue = ListOpDeSerialize(valueJson.toStyledString());
            break;
        }
        case IDefineCode::StrCST : {
            opValue = StrOpJsonDeSerialize(valueJson.toStyledString());
            break;
        }
        case IDefineCode::ArrayRef : {
            opValue = ArrayOpJsonDeSerialize(valueJson.toStyledString());
            break;
        }
        case IDefineCode::Decl : {
            opValue = DeclBaseOpJsonDeSerialize(valueJson.toStyledString());
            break;
        }
        case IDefineCode::FieldDecl : {
            opValue = FieldDeclOpJsonDeSerialize(valueJson.toStyledString());
            break;
        }
        case IDefineCode::AddrExp : {
            opValue = AddressOpJsonDeSerialize(valueJson.toStyledString());
            break;
        }
        case IDefineCode::Constructor : {
            opValue = ConstructorOpJsonDeSerialize(valueJson.toStyledString());
            break;
        }
        case IDefineCode::Vec : {
            opValue = VecOpJsonDeSerialize(valueJson.toStyledString());
            break;
        }
        case IDefineCode::BLOCK : {
            opValue = BlockOpJsonDeSerialize(valueJson.toStyledString());
            break;
        }
        case IDefineCode::COMPONENT : {
            opValue = ComponentOpJsonDeSerialize(valueJson.toStyledString());
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
        } else if (opCode == DebugOp::getOperationName().str()) {
            uint64_t opID = GetID(opJson["id"]);
            opBuilder->create<DebugOp>(opBuilder->getUnknownLoc(), opID);
        } else if (opCode == BaseOp::getOperationName().str()) {
            uint64_t opID = GetID(opJson["id"]);
            opBuilder->create<BaseOp>(opBuilder->getUnknownLoc(), opID, opCode);
        } else if (opCode == AsmOp::getOperationName().str()) {
            AsmOpJsonDeserialize(opJson.toStyledString());
        } else if (opCode == SwitchOp::getOperationName().str()) {
            printf("switch op deserialize\n");
            SwitchOpJsonDeserialize(opJson.toStyledString());
        } else if (opCode == GotoOp::getOperationName().str()) {
            GotoOpJsonDeSerialize(opJson.toStyledString());
        } else if (opCode == LabelOp::getOperationName().str()) {
            LabelOpJsonDeSerialize(opJson.toStyledString());
        } else if (opCode == TransactionOp::getOperationName().str()) {
            TransactionOpJsonDeSerialize(opJson.toStyledString());
        } else if (opCode == ResxOp::getOperationName().str()) {
            ResxOpJsonDeSerialize(opJson.toStyledString());
        } else if (opCode == EHDispatchOp::getOperationName().str()) {
            EHDispatchOpJsonDeSerialize(opJson.toStyledString());
        } else if (opCode == EHMntOp::getOperationName().str()) {
            EHMntOpJsonDeSerialize(opJson.toStyledString());
        } else if (opCode == BindOp::getOperationName().str()) {
            BindOpJsonDeSerialize(opJson.toStyledString());
        } else if (opCode == TryOp::getOperationName().str()) {
            TryOpJsonDeSerialize(opJson.toStyledString());
        } else if (opCode == CatchOp::getOperationName().str()) {
            CatchOpJsonDeSerialize(opJson.toStyledString());
        } else if (opCode == NopOp::getOperationName().str()) {
            NopOpJsonDeSerialize(opJson.toStyledString());
        } else if (opCode == EHElseOp::getOperationName().str()) {
            EHElseOpJsonDeSerialize(opJson.toStyledString());
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
        PluginIR::PluginTypeBase retType = TypeJsonDeSerialize(node["retType"].toStyledString());          
        FunctionOp fOp = opBuilder.create<FunctionOp>(
                location, id, funcAttributes["funcName"], declaredInline, retType);
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
    } else if (id == static_cast<uint64_t>(PluginIR::ArrayTyID)) {
        mlir::Type elemTy = TypeJsonDeSerialize(type["elementType"].toStyledString());
        uint64_t elemNum = GetID(type["arraysize"]);
        baseType = PluginIR::PluginArrayType::get(PluginServer::GetInstance()->GetContext(), elemTy, elemNum);
    } else if (id == static_cast<uint64_t>(PluginIR::VectorTyID)) {
        mlir::Type elemTy = TypeJsonDeSerialize(type["elementType"].toStyledString());
        uint64_t elemNum = GetID(type["vectorelemnum"]);
        baseType = PluginIR::PluginVectorType::get(PluginServer::GetInstance()->GetContext(), elemTy, elemNum);
    } else if (id == static_cast<uint64_t>(PluginIR::FunctionTyID)) {
        mlir::Type returnTy = TypeJsonDeSerialize(type["fnreturntype"].toStyledString());
        llvm::SmallVector<Type> typelist;
        Json::Value::Members fnTypeNum = type["fnargsType"].getMemberNames();
        uint64_t argsNum = fnTypeNum.size();
        for (size_t paramIndex = 0; paramIndex < argsNum; paramIndex++) {
            string Key = "argType" + std::to_string(paramIndex);
            mlir::Type paramTy = TypeJsonDeSerialize(type["fnargsType"][Key].toStyledString());
            typelist.push_back(paramTy);
        }
        baseType = PluginIR::PluginFunctionType::get(PluginServer::GetInstance()->GetContext(), returnTy, typelist);
    } else if (id == static_cast<uint64_t>(PluginIR::StructTyID)) {
        std::string tyName = type["structtype"].asString();
        llvm::SmallVector<Type> typelist;
        Json::Value::Members elemTypeNum = type["structelemType"].getMemberNames();
        for (size_t paramIndex = 0; paramIndex < elemTypeNum.size(); paramIndex++) {
            string Key = "elemType" + std::to_string(paramIndex);
            mlir::Type paramTy = TypeJsonDeSerialize(type["structelemType"][Key].toStyledString());
            typelist.push_back(paramTy);
        }
        llvm::SmallVector<std::string> names;
        Json::Value::Members elemNameNum = type["structelemName"].getMemberNames();
        for (size_t paramIndex = 0; paramIndex < elemTypeNum.size(); paramIndex++) {
            std::string Key = "elemName" + std::to_string(paramIndex);
            std::string elemName = type["structelemName"][Key].asString();
            names.push_back(elemName);
        }
        baseType = PluginIR::PluginStructType::get(PluginServer::GetInstance()->GetContext(), tyName, typelist, names);
    }
    else {
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
        string operationKey = "ID" + std::to_string(iter);
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
    mlir::OpBuilder *opBuilder = PluginServer::GetInstance()->GetOpBuilder();
    Json::Value calleeJson = node["callee"];
    CallOp op;
    if (calleeJson.isNull()) {
        op = opBuilder->create<CallOp>(opBuilder->getUnknownLoc(), id, ops);
    } else {
        mlir::StringRef callName(calleeJson.asString());
        op = opBuilder->create<CallOp>(opBuilder->getUnknownLoc(),
                                       id, callName, ops);
    }
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

mlir::Value PluginJson::ListOpDeSerialize(const string& data)
{
    Json::Value root;
    Json::Reader reader;
    reader.parse(data, root);
    uint64_t id = GetID(root["id"]);
    bool readOnly = (bool)atoi(root["readOnly"].asString().c_str());
    bool hasPurpose = (bool)atoi(root["hasPurpose"].asString().c_str());
    Json::Value operandJson = root["operands"];
    Json::Value::Members operandMember = operandJson.getMemberNames();
    llvm::SmallVector<mlir::Value, 4> ops;
    for (size_t opIter = 0; opIter < operandMember.size(); opIter++) {
        mlir::Value opValue = ValueJsonDeSerialize(operandJson[std::to_string(opIter).c_str()]);
        ops.push_back(opValue);
    }
    mlir::Type retType = TypeJsonDeSerialize(root["retType"].toStyledString().c_str());
    mlir::OpBuilder *opBuilder = PluginServer::GetInstance()->GetOpBuilder();
    mlir::Value trrelist = opBuilder->create<ListOp>(opBuilder->getUnknownLoc(), id,
                            IDefineCode::LIST, readOnly, hasPurpose, ops, retType);
    return trrelist;
}

mlir::Value PluginJson::StrOpJsonDeSerialize(const string& data)
{
    Json::Value root;
    Json::Reader reader;
    reader.parse(data, root);
    uint64_t id = GetID(root["id"]);
    mlir::StringRef str(root["str"].asString());
    bool readOnly = (bool)atoi(root["readOnly"].asString().c_str());
    mlir::Type retType = TypeJsonDeSerialize(root["retType"].toStyledString().c_str());
    mlir::OpBuilder *opBuilder = PluginServer::GetInstance()->GetOpBuilder();
    mlir::Value strop = opBuilder->create<StrOp>(opBuilder->getUnknownLoc(), id,
                            IDefineCode::StrCST, readOnly, str, retType);
    return strop;
}
mlir::Value PluginJson::ArrayOpJsonDeSerialize(const string& data)
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
    mlir::Value arrayref = opBuilder->create<ArrayOp>(opBuilder->getUnknownLoc(), id,
                            IDefineCode::ArrayRef, readOnly, base, offset, retType);
    return arrayref;
}

mlir::Value PluginJson::DeclBaseOpJsonDeSerialize(const string& data)
{
    Json::Value root;
    Json::Reader reader;
    reader.parse(data, root);
    uint64_t id = GetID(root["id"]);
    bool readOnly = (bool)atoi(root["readOnly"].asString().c_str());
    bool addressable = (bool)atoi(root["addressable"].asString().c_str());
    bool used = (bool)atoi(root["used"].asString().c_str());
    int32_t uid = GetID(root["uid"]);
    mlir::Value initial = ValueJsonDeSerialize(root["initial"]);
    mlir::Value name = ValueJsonDeSerialize(root["name"]);
    llvm::Optional<uint64_t> chain;
    if (root["chain"]) {
        chain = GetID(root["chain"]);
    }
    mlir::Type retType = TypeJsonDeSerialize(root["retType"].toStyledString().c_str());
    mlir::OpBuilder *opBuilder = PluginServer::GetInstance()->GetOpBuilder();
    mlir::Value declOp = opBuilder->create<DeclBaseOp>(opBuilder->getUnknownLoc(), id,
                            IDefineCode::Decl, readOnly, addressable, used, uid, initial, name, chain, retType);
    return declOp;
}

mlir::Value PluginJson::FieldDeclOpJsonDeSerialize(const string& data)
{
    Json::Value root;
    Json::Reader reader;
    reader.parse(data, root);
    uint64_t id = GetID(root["id"]);
    bool readOnly = (bool)atoi(root["readOnly"].asString().c_str());
    bool addressable = (bool)atoi(root["addressable"].asString().c_str());
    bool used = (bool)atoi(root["used"].asString().c_str());
    int32_t uid = GetID(root["uid"]);
    mlir::Value initial = ValueJsonDeSerialize(root["initial"]);
    mlir::Value name = ValueJsonDeSerialize(root["name"]);
    uint64_t chain = GetID(root["chain"]);
    mlir::Value fieldOffset = ValueJsonDeSerialize(root["fieldOffset"]);
    mlir::Value fieldBitOffset = ValueJsonDeSerialize(root["fieldBitOffset"]);
    mlir::Type retType = TypeJsonDeSerialize(root["retType"].toStyledString().c_str());
    mlir::OpBuilder *opBuilder = PluginServer::GetInstance()->GetOpBuilder();
    mlir::Value fieldOp = opBuilder->create<FieldDeclOp>(opBuilder->getUnknownLoc(), id,
                            IDefineCode::FieldDecl, readOnly, addressable, used, uid, initial, name, chain,
                            fieldOffset, fieldBitOffset, retType);
    return fieldOp;
}

mlir::Value PluginJson::AddressOpJsonDeSerialize(const string& data)
{
    Json::Value root;
    Json::Reader reader;
    reader.parse(data, root);
    uint64_t id = GetID(root["id"]);
    bool readOnly = (bool)atoi(root["readOnly"].asString().c_str());
    mlir::Value operand = ValueJsonDeSerialize(root["operand"]);
    mlir::Type retType = TypeJsonDeSerialize(root["retType"].toStyledString().c_str());
    mlir::OpBuilder *opBuilder = PluginServer::GetInstance()->GetOpBuilder();
    mlir::Value addrOp = opBuilder->create<AddressOp>(opBuilder->getUnknownLoc(), id,
                            IDefineCode::AddrExp, readOnly, operand, retType);
    return addrOp;
}

mlir::Value PluginJson::ConstructorOpJsonDeSerialize(const string& data)
{
    Json::Value root;
    Json::Reader reader;
    reader.parse(data, root);
    uint64_t id = GetID(root["id"]);
    bool readOnly = (bool)atoi(root["readOnly"].asString().c_str());
    uint64_t len = GetID(root["len"]);
    Json::Value idxJson = root["idx"];
    Json::Value::Members idxMember = idxJson.getMemberNames();
    llvm::SmallVector<mlir::Value, 4> idx, val;
    for (size_t iter = 0; iter < idxMember.size(); iter++) {
        mlir::Value opValue = ValueJsonDeSerialize(idxJson[std::to_string(iter).c_str()]);
        idx.push_back(opValue);
    }
    Json::Value valJson = root["val"];
    Json::Value::Members valMember = valJson.getMemberNames();
    for (size_t iter = 0; iter < valMember.size(); iter++) {
        mlir::Value opValue = ValueJsonDeSerialize(valJson[std::to_string(iter).c_str()]);
        val.push_back(opValue);
    }
    mlir::Type retType = TypeJsonDeSerialize(root["retType"].toStyledString().c_str());
    mlir::OpBuilder *opBuilder = PluginServer::GetInstance()->GetOpBuilder();
    mlir::Value constructorOp = opBuilder->create<ConstructorOp>(opBuilder->getUnknownLoc(), id,
                            IDefineCode::Constructor, readOnly, len, idx, val, retType);
    return constructorOp;
}

mlir::Value PluginJson::VecOpJsonDeSerialize(const string& data)
{
    Json::Value root;
    Json::Reader reader;
    reader.parse(data, root);
    uint64_t id = GetID(root["id"]);
    bool readOnly = (bool)atoi(root["readOnly"].asString().c_str());
    uint64_t len = GetID(root["len"]);
    Json::Value elementsJson = root["elements"];
    Json::Value::Members elementsMember = elementsJson.getMemberNames();
    llvm::SmallVector<mlir::Value, 4> elements;
    for (size_t iter = 0; iter < elementsMember.size(); iter++) {
        mlir::Value opValue = ValueJsonDeSerialize(elementsJson[std::to_string(iter).c_str()]);
        elements.push_back(opValue);
    }
    mlir::Type retType = TypeJsonDeSerialize(root["retType"].toStyledString().c_str());
    mlir::OpBuilder *opBuilder = PluginServer::GetInstance()->GetOpBuilder();
    mlir::Value vecOp = opBuilder->create<VecOp>(opBuilder->getUnknownLoc(), id,
                            IDefineCode::Vec, readOnly, len, elements, retType);
    return vecOp;
}

mlir::Value PluginJson::BlockOpJsonDeSerialize(const string& data)
{
    Json::Value root;
    Json::Reader reader;
    reader.parse(data, root);
    uint64_t id = GetID(root["id"]);
    bool readOnly = (bool)atoi(root["readOnly"].asString().c_str());
    uint64_t supercontext = GetID(root["supercontext"]);
    llvm::Optional<mlir::Value> vars, subblocks, chain, abstract_origin;
    if (root["vars"]) {
        vars = ValueJsonDeSerialize(root["vars"]);
    }

    if (root["subblocks"]) {
        subblocks = ValueJsonDeSerialize(root["subblocks"]);
    }
    if (root["chain"]) {
        chain = ValueJsonDeSerialize(root["chain"]);
    }
    if (root["abstract_origin"]) {
        abstract_origin = ValueJsonDeSerialize(root["abstract_origin"]);
    }
    mlir::Type retType = TypeJsonDeSerialize(root["retType"].toStyledString().c_str());
    mlir::OpBuilder *opBuilder = PluginServer::GetInstance()->GetOpBuilder();
    mlir::Value blockOp = opBuilder->create<BlockOp>(
                opBuilder->getUnknownLoc(), id, IDefineCode::BLOCK, readOnly, vars, supercontext, subblocks,
                chain, abstract_origin, retType);
    return blockOp;
}

mlir::Value PluginJson::ComponentOpJsonDeSerialize(const string& data)
{
    Json::Value root;
    Json::Reader reader;
    reader.parse(data, root);
    uint64_t id = GetID(root["id"]);
    bool readOnly = (bool)atoi(root["readOnly"].asString().c_str());
    mlir::Value component = ValueJsonDeSerialize(root["component"]);
    mlir::Value field = ValueJsonDeSerialize(root["field"]);
    mlir::Type retType = TypeJsonDeSerialize(root["retType"].toStyledString().c_str());
    mlir::OpBuilder *opBuilder = PluginServer::GetInstance()->GetOpBuilder();
    mlir::Value componentOp = opBuilder->create<ComponentOp>(
                opBuilder->getUnknownLoc(), id, IDefineCode::COMPONENT, readOnly, component, field, retType);
    return componentOp;
}

mlir::Operation *PluginJson::GotoOpJsonDeSerialize(const string& data)
{
    Json::Value node;
    Json::Reader reader;
    reader.parse(data, node);
    int64_t id = GetID(node["id"]);
    int64_t address = GetID(node["address"]);
    mlir::Value dest = ValueJsonDeSerialize(node["dest"]);
    int64_t successaddr = GetID(node["successaddr"]);
    mlir::Block* success = PluginServer::GetInstance()->FindBlock(successaddr);
    mlir::OpBuilder *opBuilder = PluginServer::GetInstance()->GetOpBuilder();
    GotoOp op = opBuilder->create<GotoOp>(opBuilder->getUnknownLoc(), id, address, dest, success, successaddr);
    return op.getOperation();
}

mlir::Operation *PluginJson::TransactionOpJsonDeSerialize(const string& data)
{
    Json::Value node;
    Json::Reader reader;
    reader.parse(data, node);
    int64_t id = GetID(node["id"]);
    int64_t address = GetID(node["address"]);
    mlir::Value labelNorm = ValueJsonDeSerialize(node["labelNorm"]);
    mlir::Value labelUninst = ValueJsonDeSerialize(node["labelUninst"]);
    mlir::Value labelOver = ValueJsonDeSerialize(node["labelOver"]);
    int64_t fallthroughaddr = GetID(node["fallthroughaddr"]);
    int64_t abortaddr = GetID(node["abortaddr"]);
    mlir::Block* fallthrough = PluginServer::GetInstance()->FindBlock(fallthroughaddr);
    mlir::Block* abort = PluginServer::GetInstance()->FindBlock(abortaddr);
    llvm::SmallVector<uint64_t, 4> stmtaddr;
    Json::Value stmtaddrJson = node["stmtaddr"];
    for (size_t index = 0; index < stmtaddrJson.getMemberNames().size(); index++) {
        string key = std::to_string(index);
        uint64_t addr = GetID(stmtaddrJson[key.c_str()]);
        stmtaddr.push_back(addr);
    }
    mlir::OpBuilder *opBuilder = PluginServer::GetInstance()->GetOpBuilder();
    TransactionOp op = opBuilder->create<TransactionOp>(opBuilder->getUnknownLoc(), id, address, stmtaddr, labelNorm,
                                        labelUninst, labelOver, fallthrough, fallthroughaddr, abort, abortaddr);
    return op.getOperation();
}

mlir::Operation *PluginJson::ResxOpJsonDeSerialize(const string& data)
{
    Json::Value node;
    Json::Reader reader;
    reader.parse(data, node);
    int64_t id = GetID(node["id"]);
    int64_t address = GetID(node["address"]);
    int64_t region = GetID(node["region"]);
    mlir::OpBuilder *opBuilder = PluginServer::GetInstance()->GetOpBuilder();
    ResxOp op = opBuilder->create<ResxOp>(opBuilder->getUnknownLoc(),
                                         id, address, region);
    return op.getOperation();
}

mlir::Operation *PluginJson::EHMntOpJsonDeSerialize(const string& data)
{
    Json::Value node;
    Json::Reader reader;
    reader.parse(data, node);
    int64_t id = GetID(node["id"]);
    mlir::Value decl = ValueJsonDeSerialize(node["decl"]);
    mlir::OpBuilder *opBuilder = PluginServer::GetInstance()->GetOpBuilder();
    EHMntOp op = opBuilder->create<EHMntOp>(opBuilder->getUnknownLoc(), id, decl);
    return op.getOperation();
}


mlir::Operation *PluginJson::EHDispatchOpJsonDeSerialize(const string& data)
{
    Json::Value node;
    Json::Reader reader;
    reader.parse(data, node);
    int64_t id = GetID(node["id"]);
    int64_t address = GetID(node["address"]);
    int64_t region = GetID(node["region"]);
    llvm::SmallVector<mlir::Block*, 4> ehHandlers;
    llvm::SmallVector<uint64_t, 4> ehHandlersaddrs;
    Json::Value ehHandlersJson = node["ehHandlersaddrs"];
    Json::Value::Members ehHandlersMember = ehHandlersJson.getMemberNames();
    for (size_t iter = 0; iter < ehHandlersMember.size(); iter++) {
        string key = std::to_string(iter);
        uint64_t ehaddr = GetID(ehHandlersJson[key.c_str()]);
        mlir::Block* succ = PluginServer::GetInstance()->FindBlock(ehaddr);
        ehHandlers.push_back(succ);
        ehHandlersaddrs.push_back(ehaddr);
    }
    mlir::OpBuilder *opBuilder = PluginServer::GetInstance()->GetOpBuilder();
    EHDispatchOp op = opBuilder->create<EHDispatchOp>(opBuilder->getUnknownLoc(),
                                         id, address, region, ehHandlers, ehHandlersaddrs);
    return op.getOperation();
}

mlir::Operation *PluginJson::LabelOpJsonDeSerialize(const string& data)
{
    Json::Value node;
    Json::Reader reader;
    reader.parse(data, node);
    int64_t id = GetID(node["id"]);
    mlir::Value label = ValueJsonDeSerialize(node["label"]);
    mlir::OpBuilder *opBuilder = PluginServer::GetInstance()->GetOpBuilder();
    LabelOp op = opBuilder->create<LabelOp>(opBuilder->getUnknownLoc(), id, label);
    return op.getOperation();
}

mlir::Operation *PluginJson::BindOpJsonDeSerialize(const string& data)
{
    Json::Value node;
    Json::Reader reader;
    reader.parse(data, node);
    int64_t id = GetID(node["id"]);
    mlir::Value vars = ValueJsonDeSerialize(node["vars"]);
    mlir::Value block = ValueJsonDeSerialize(node["block"]);

    Json::Value bodyJson = node["body"];
    Json::Value::Members bodyMember = bodyJson.getMemberNames();
    llvm::SmallVector<uint64_t, 4> bodyaddrs;
    for (size_t iter = 0; iter < bodyMember.size(); iter++) {
        string key = std::to_string(iter);
        uint64_t addr = GetID(bodyJson[key.c_str()]);
        bodyaddrs.push_back(addr);
    }
    mlir::OpBuilder *opBuilder = PluginServer::GetInstance()->GetOpBuilder();
    BindOp op = opBuilder->create<BindOp>(opBuilder->getUnknownLoc(), id, vars, bodyaddrs, block);
    return op.getOperation();
}

mlir::Operation *PluginJson::TryOpJsonDeSerialize(const string& data)
{
    Json::Value node;
    Json::Reader reader;
    reader.parse(data, node);
    int64_t id = GetID(node["id"]);
    Json::Value evalJson = node["eval"];
    Json::Value::Members evalMember = evalJson.getMemberNames();
    llvm::SmallVector<uint64_t, 4> evaladdrs, cleanupaddrs;
    for (size_t iter = 0; iter < evalMember.size(); iter++) {
        string key = std::to_string(iter);
        uint64_t addr = GetID(evalJson[key.c_str()]);
        evaladdrs.push_back(addr);
    }
    Json::Value cleanupJson = node["cleanup"];
    Json::Value::Members cleanupMember = cleanupJson.getMemberNames();
    for (size_t iter = 0; iter < cleanupMember.size(); iter++) {
        string key = std::to_string(iter);
        uint64_t addr = GetID(cleanupJson[key.c_str()]);
        cleanupaddrs.push_back(addr);
    }

    int64_t kind = GetID(node["kind"]);
    mlir::OpBuilder *opBuilder = PluginServer::GetInstance()->GetOpBuilder();
    TryOp op = opBuilder->create<TryOp>(opBuilder->getUnknownLoc(), id, evaladdrs, cleanupaddrs, kind);
    return op.getOperation();
}

mlir::Operation *PluginJson::CatchOpJsonDeSerialize(const string& data)
{
    Json::Value node;
    Json::Reader reader;
    reader.parse(data, node);
    int64_t id = GetID(node["id"]);
    mlir::Value types = ValueJsonDeSerialize(node["types"]);

    Json::Value handlerJson = node["handler"];
    Json::Value::Members handlerMember = handlerJson.getMemberNames();
    llvm::SmallVector<uint64_t, 4> handleraddrs;
    for (size_t iter = 0; iter < handlerMember.size(); iter++) {
        string key = std::to_string(iter);
        uint64_t addr = GetID(handlerJson[key.c_str()]);
        handleraddrs.push_back(addr);
    }
    mlir::OpBuilder *opBuilder = PluginServer::GetInstance()->GetOpBuilder();
    CatchOp op = opBuilder->create<CatchOp>(opBuilder->getUnknownLoc(), id, types, handleraddrs);
    return op.getOperation();
}

mlir::Operation *PluginJson::NopOpJsonDeSerialize(const string& data)
{
    Json::Value node;
    Json::Reader reader;
    reader.parse(data, node);
    uint64_t id = GetID(node["id"]);
    mlir::OpBuilder *opBuilder = PluginServer::GetInstance()->GetOpBuilder();
    NopOp op = opBuilder->create<NopOp>(opBuilder->getUnknownLoc(), id);
    PluginServer::GetInstance()->InsertDefOperation(id, op.getOperation());
    return op.getOperation();
}

mlir::Operation *PluginJson::EHElseOpJsonDeSerialize(const string& data)
{
    Json::Value node;
    Json::Reader reader;
    reader.parse(data, node);
    uint64_t id = GetID(node["id"]);

    llvm::SmallVector<uint64_t, 4> nbody, ebody;

    Json::Value nbodyJson = node["nbody"];
    Json::Value::Members nbodyMember = nbodyJson.getMemberNames();

    for (size_t iter = 0; iter < nbodyMember.size(); iter++) {
        string key = std::to_string(iter);
        uint64_t addr = GetID(nbodyJson[key.c_str()]);
        nbody.push_back(addr);
    }

    Json::Value ebodyJson = node["ebody"];
    Json::Value::Members ebodyMember = ebodyJson.getMemberNames();

    for (size_t iter = 0; iter < ebodyMember.size(); iter++) {
        string key = std::to_string(iter);
        uint64_t addr = GetID(ebodyJson[key.c_str()]);
        ebody.push_back(addr);
    }
    mlir::OpBuilder *opBuilder = PluginServer::GetInstance()->GetOpBuilder();
    EHElseOp op = opBuilder->create<EHElseOp>(opBuilder->getUnknownLoc(), id, nbody, ebody);
    PluginServer::GetInstance()->InsertDefOperation(id, op.getOperation());
    return op.getOperation();
}

mlir::Operation *PluginJson::AsmOpJsonDeserialize(const string& data)
{
    Json::Value node;
    Json::Reader reader;
    reader.parse(data, node);
    Json::Value operandJson = node["operands"];
    Json::Value::Members operandMember = operandJson.getMemberNames();
    llvm::SmallVector<mlir::Value, 4> ops;
    for (size_t opIter = 0; opIter < operandMember.size(); opIter++) {
        string key = std::to_string(opIter);
        mlir::Value opValue = ValueJsonDeSerialize(operandJson[key.c_str()]);
        ops.push_back(opValue);
    }
    uint64_t id = GetID(node["id"]);
    mlir::StringRef statement(node["statement"].asString());
    uint32_t nInputs = GetID(node["nInputs"]);
    uint32_t nOutputs = GetID(node["nOutputs"]);
    uint32_t nClobbers = GetID(node["nClobbers"]);
    mlir::OpBuilder *opBuilder = PluginServer::GetInstance()->GetOpBuilder();
    AsmOp op = opBuilder->create<AsmOp>(opBuilder->getUnknownLoc(), id, statement, nInputs, nOutputs,
                                             nClobbers, ops);
    PluginServer::GetInstance()->InsertDefOperation(id, op.getOperation());
    return op.getOperation();
}

mlir::Operation *PluginJson::SwitchOpJsonDeserialize(const string& data)
{
    Json::Value node;
    Json::Reader reader;
    reader.parse(data, node);
    uint64_t id = GetID(node["id"]);
    uint64_t address = GetID(node["address"]);
    uint64_t defaultDestAddr = GetID(node["defaultaddr"]);
    mlir::Block* defaultDest = PluginServer::GetInstance()->FindBlock(defaultDestAddr);


    Json::Value operandJson = node["operands"];
    Json::Value::Members operandMember = operandJson.getMemberNames();
    llvm::SmallVector<mlir::Value, 4> ops;
    mlir::Value index, defaultLabel;
    for (size_t opIter = 0; opIter < operandMember.size(); opIter++) {
        string key = std::to_string(opIter);
        mlir::Value opValue = ValueJsonDeSerialize(operandJson[key.c_str()]);
        if (opIter == 0) {
            index = opValue;
            continue;
        } else if (opIter == 1) {
            defaultLabel = opValue;
            continue;
        }
        ops.push_back(opValue);
    }

    Json::Value caseaddrJson = node["case"];
    llvm::SmallVector<uint64_t, 4> caseaddr;
    llvm::SmallVector<mlir::Block*, 4> caseDest;
    for (size_t index = 0; index < caseaddrJson.getMemberNames().size(); index++) {
        string key = std::to_string(index);
        uint64_t addr = GetID(caseaddrJson[key.c_str()]);
        mlir::Block* casebb = PluginServer::GetInstance()->FindBlock(addr);
        caseaddr.push_back(addr);
        caseDest.push_back(casebb);
    }
    mlir::OpBuilder *opBuilder = PluginServer::GetInstance()->GetOpBuilder();
    SwitchOp op = opBuilder->create<SwitchOp>(opBuilder->getUnknownLoc(), id, index, address, defaultLabel, ops, defaultDest,
                                            defaultDestAddr, caseDest, caseaddr);
    PluginServer::GetInstance()->InsertDefOperation(id, op.getOperation());
    return op.getOperation();
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
    } else if (opCode == DebugOp::getOperationName().str()) {
        uint64_t opID = GetID(opJson["id"]);
        mlir::OpBuilder *opBuilder = PluginServer::GetInstance()->GetOpBuilder();
        opBuilder->create<DebugOp>(opBuilder->getUnknownLoc(), opID);
    } else if (opCode == BaseOp::getOperationName().str()) {
        uint64_t opID = GetID(opJson["id"]);
        mlir::OpBuilder *opBuilder = PluginServer::GetInstance()->GetOpBuilder();
        opBuilder->create<BaseOp>(opBuilder->getUnknownLoc(), opID, opCode);
    } else if (opCode == AsmOp::getOperationName().str()) {
        opData.push_back(AsmOpJsonDeserialize(opJson.toStyledString()));
    } else if (opCode == SwitchOp::getOperationName().str()) {
        opData.push_back(SwitchOpJsonDeserialize(opJson.toStyledString()));
    } else if (opCode == GotoOp::getOperationName().str()) {
        opData.push_back(GotoOpJsonDeSerialize(opJson.toStyledString()));
    } else if (opCode == LabelOp::getOperationName().str()) {
        opData.push_back(LabelOpJsonDeSerialize(opJson.toStyledString()));
    } else if (opCode == TransactionOp::getOperationName().str()) {
        opData.push_back(TransactionOpJsonDeSerialize(opJson.toStyledString()));
    } else if (opCode == ResxOp::getOperationName().str()) {
        opData.push_back(ResxOpJsonDeSerialize(opJson.toStyledString()));
    } else if (opCode == EHDispatchOp::getOperationName().str()) {
        opData.push_back(EHDispatchOpJsonDeSerialize(opJson.toStyledString()));
    } else if (opCode == EHMntOp::getOperationName().str()) {
        opData.push_back(EHMntOpJsonDeSerialize(opJson.toStyledString()));
    } else if (opCode == BindOp::getOperationName().str()) {
        opData.push_back(BindOpJsonDeSerialize(opJson.toStyledString()));
    } else if (opCode == TryOp::getOperationName().str()) {
        opData.push_back(TryOpJsonDeSerialize(opJson.toStyledString()));
    } else if (opCode == CatchOp::getOperationName().str()) {
        opData.push_back(CatchOpJsonDeSerialize(opJson.toStyledString()));
    } else if (opCode == NopOp::getOperationName().str()) {
        opData.push_back(NopOpJsonDeSerialize(opJson.toStyledString()));
    } else if (opCode == EHElseOp::getOperationName().str()) {
        opData.push_back(EHElseOpJsonDeSerialize(opJson.toStyledString()));
    }
}
} // namespace PinJson
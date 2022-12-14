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
    This file contains the implementation of the client PluginServer class.
*/

#include <iostream>
#include <memory>
#include <string>
#include <vector>
#include <thread>

#include "Dialect/PluginDialect.h"
#include "PluginAPI/PluginServerAPI.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "user.h"
#include "PluginServer/PluginLog.h"
#include "PluginServer/PluginServer.h"
#include "Dialect/PluginTypes.h"

namespace PinServer {
using namespace mlir::Plugin;
using std::cout;
using std::endl;
using std::pair;
static std::unique_ptr<Server> g_server; // grpc对象指针
static PluginServer g_service; // 插件server对象

PluginServer *PluginServer::GetInstance(void)
{
    return &g_service;
}

int PluginServer::RegisterUserFunc(InjectPoint inject, UserFunc func)
{
    if ((inject >= HANDLE_MAX) || (func == nullptr)) {
        return -1;
    }
    string name = "funcname" + std::to_string((uint64_t)&func);
    userFunc[inject].push_back(RecordedUserFunc(name, func));
    return 0;
}

int PluginServer::RegisterPassManagerSetup(InjectPoint inject, const ManagerSetupData& setupData, UserFunc func)
{
    if (inject != HANDLE_MANAGER_SETUP) {
        return -1;
    }

    Json::Value root;
    root["refPassName"] = setupData.refPassName;
    root["passNum"] = setupData.passNum;
    root["passPosition"] = setupData.passPosition;
    string params = root.toStyledString();

    userFunc[inject].push_back(RecordedUserFunc(params, func));
    return 0;
}

vector<mlir::Operation *> PluginServer::GetOpResult(void)
{
    vector<mlir::Operation *> retOps = opData;
    opData.clear();
    return retOps;
}

vector<mlir::Plugin::FunctionOp> PluginServer::GetFunctionOpResult(void)
{
    vector<mlir::Plugin::FunctionOp> retOps = funcOpData;
    funcOpData.clear();
    return retOps;
}

vector<mlir::Plugin::LocalDeclOp> PluginServer::GetLocalDeclResult()
{
    vector<mlir::Plugin::LocalDeclOp> retOps = decls;
    decls.clear();
    return retOps;
}

vector<mlir::Plugin::LoopOp> PluginServer::LoopOpsResult()
{
    vector<mlir::Plugin::LoopOp> retLoops = loops;
    loops.clear();
    return retLoops;
}

LoopOp PluginServer::LoopOpResult()
{
    mlir::Plugin::LoopOp retLoop = loop;
    return retLoop;
}

void PluginServer::EraseBlock(mlir::Block* b)
{
    if (auto bbit = basicblockMaps.find(b); bbit != basicblockMaps.end()) {
        uint64_t addr = bbit->second;
        basicblockMaps.erase(bbit);
        if (auto bit = blockMaps.find(addr); bit != blockMaps.end()) {
            blockMaps.erase(bit);
        }
    }
}

mlir::Block* PluginServer::FindBlock(uint64_t id)
{
    auto iter = this->blockMaps.find(id);
    assert(iter != this->blockMaps.end());
    return iter->second;
}

uint64_t PluginServer::GetBlockResult(mlir::Block* b)
{
    uint64_t newAddr = GetIdResult();
    mlir::Block* block = opBuilder.createBlock(b);
    this->blockMaps.insert({newAddr, block});
    this->basicblockMaps.insert({block, newAddr});
    return newAddr;
}

uint64_t PluginServer::FindBasicBlock(mlir::Block* b)
{
    auto bbIter = basicblockMaps.find(b);
    assert(bbIter != basicblockMaps.end());
    return bbIter->second;
}

pair<mlir::Block*, mlir::Block*> PluginServer::EdgeResult()
{
    pair<mlir::Block*, mlir::Block*> e;
    e.first = edge.first;
    e.second = edge.second;
    return e;
}

vector<pair<mlir::Block*, mlir::Block*> > PluginServer::EdgesResult()
{
    vector<pair<mlir::Block*, mlir::Block*> > retEdges;
    retEdges = edges;
    edges.clear();
    return retEdges;
}

bool PluginServer::GetBoolResult()
{
    return this->boolResult;
}

uint64_t PluginServer::GetIdResult()
{
    return this->idResult;
}

vector<uint64_t> PluginServer::GetIdsResult()
{
    vector<uint64_t> retIds = idsResult;
    idsResult.clear();
    return retIds;
}

mlir::Value PluginServer::GetValueResult()
{
    return this->valueResult;
}

vector<mlir::Plugin::PhiOp> PluginServer::GetPhiOpsResult()
{
    vector<mlir::Plugin::PhiOp> retOps;
    for (auto item : opData) {
        PhiOp p = llvm::dyn_cast<mlir::Plugin::PhiOp>(item);
        retOps.push_back(p);
    }
    opData.clear();
    return retOps;
}

void PluginServer::JsonGetAttributes(Json::Value node, map<string, string>& attributes)
{
    Json::Value::Members attMember = node.getMemberNames();
    for (unsigned int i = 0; i < attMember.size(); i++) {
        string key = attMember[i];
        string value = node[key.c_str()].asString();
        attributes[key] = value;
    }
}

static uintptr_t GetID(Json::Value node)
{
    string id = node.asString();
    return atol(id.c_str());
}

mlir::Value PluginServer::ValueJsonDeSerialize(Json::Value valueJson)
{
    uint64_t opId = GetID(valueJson["id"]);
    IDefineCode defCode = IDefineCode(
            atoi(valueJson["defCode"].asString().c_str()));
    switch (defCode) {
        case IDefineCode::MemRef : {
            break;
        }
        case IDefineCode::IntCST : {
            break;
        }
        default:
            break;
    }
    mlir::Type retType = TypeJsonDeSerialize(
            valueJson["retType"].toStyledString());
    mlir::Value opValue = opBuilder.create<PlaceholderOp>(
            opBuilder.getUnknownLoc(), opId, defCode, retType);
    return opValue;
}

void PluginServer::JsonDeSerialize(const string& key, const string& data)
{
    if (key == "FuncOpResult") {
        FuncOpJsonDeSerialize(data);
    } else if (key == "LocalDeclOpResult") {
        LocalDeclOpJsonDeSerialize(data);
    } else if (key == "LoopOpResult") {
        LoopOpJsonDeSerialize (data);
    } else if (key == "LoopOpsResult") {
        LoopOpsJsonDeSerialize (data);
    } else if (key == "BoolResult") {
        this->boolResult = (bool)atol(data.c_str());
    } else if (key == "VoidResult") {
        ;
    } else if (key == "EdgeResult") {
        EdgeJsonDeSerialize(data);
    } else if (key == "EdgesResult") {
        EdgesJsonDeSerialize(data);
    } else if (key == "IdsResult") {
        IdsJsonDeSerialize(data);
    } else if (key == "IdResult") {
        this->idResult = atol(data.c_str());
    } else if (key == "ValueResult") {
        context.getOrLoadDialect<PluginDialect>();
        opBuilder = mlir::OpBuilder(&context);
        this->valueResult = ValueJsonDeSerialize(data.c_str());
    } else if (key == "GetPhiOps") {
        GetPhiOpsJsonDeSerialize(data);
    } else {
        cout << "not Json,key:" << key << ",value:" << data << endl;
    }
}

Json::Value PluginServer::TypeJsonSerialize (PluginIR::PluginTypeBase& type)
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

PluginIR::PluginTypeBase PluginServer::TypeJsonDeSerialize(const string& data)
{
    Json::Value root;
    Json::Reader reader;
    Json::Value node;
    reader.parse(data, root);

    PluginIR::PluginTypeBase baseType;

    Json::Value type = root["type"];
    uint64_t id = GetID(type["id"]);
    PluginIR::PluginTypeID PluginTypeId = static_cast<PluginIR::PluginTypeID>(id);

    if (type["signed"] && (id >= static_cast<uint64_t>(PluginIR::UIntegerTy1ID) && id <= static_cast<uint64_t>(PluginIR::IntegerTy64ID))) {
        string s = type["signed"].asString();
        uint64_t width = GetID(type["width"]);
        if (s == "1") {
            baseType = PluginIR::PluginIntegerType::get(&context, width, PluginIR::PluginIntegerType::Signed);
        }
        else {
            baseType = PluginIR::PluginIntegerType::get(&context, width, PluginIR::PluginIntegerType::Unsigned);
        }
    }
    else if (type["width"] && (id == static_cast<uint64_t>(PluginIR::FloatTyID) || id == static_cast<uint64_t>(PluginIR::DoubleTyID)) ) {
        uint64_t width = GetID(type["width"]);
        baseType = PluginIR::PluginFloatType::get(&context, width);
    }else if (id == static_cast<uint64_t>(PluginIR::PointerTyID)) {
        mlir::Type elemTy = TypeJsonDeSerialize(type["elementType"].toStyledString());
        auto ty = elemTy.dyn_cast<PluginIR::PluginTypeBase>();
        baseType = PluginIR::PluginPointerType::get(&context, elemTy);
    }else {
        if (PluginTypeId == PluginIR::VoidTyID)
            baseType = PluginIR::PluginVoidType::get(&context);
        if (PluginTypeId == PluginIR::BooleanTyID)
            baseType = PluginIR::PluginBooleanType::get(&context);
        if (PluginTypeId == PluginIR::UndefTyID)
            baseType = PluginIR::PluginUndefType::get(&context);
    }

    pluginType = baseType;
    return baseType;
}

bool PluginServer::ProcessBlock(mlir::Block* block, mlir::Region& rg,
                                const Json::Value& blockJson)
{
    if (blockJson.isNull()) {
        return false;
    }
    // todo process func return type
    // todo isDeclaration

    // process each stmt
    opBuilder.setInsertionPointToStart(block);
    Json::Value::Members opMember = blockJson.getMemberNames();
    for (Json::Value::Members::iterator opIdx = opMember.begin();
            opIdx != opMember.end(); opIdx++) {
        string baseOpKey = *opIdx;
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
            opBuilder.create<BaseOp>(opBuilder.getUnknownLoc(), opID, opCode);
        }
    }
    // fprintf(stderr, "[bb] op:%ld, succ: %d\n", block->getOperations().size(), block->getNumSuccessors());
    return true;
}

void PluginServer::FuncOpJsonDeSerialize(const string& data)
{
    Json::Value root;
    Json::Reader reader;
    Json::Value node;
    reader.parse(data, root);

    Json::Value::Members operation = root.getMemberNames();

    context.getOrLoadDialect<PluginDialect>();
    opBuilder = mlir::OpBuilder(&context);
    for (Json::Value::Members::iterator iter = operation.begin();
         iter != operation.end(); iter++) {
        string operationKey = *iter;
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
        for (Json::Value::Members::iterator bbIdx = bbMember.begin();
             bbIdx != bbMember.end(); bbIdx++) {
            string blockKey = *bbIdx;
            Json::Value blockJson = regionJson[blockKey];
            mlir::Block* block = opBuilder.createBlock(&bodyRegion);
            this->blockMaps.insert({GetID(blockJson["address"]), block});
            this->basicblockMaps.insert({block, GetID(blockJson["address"])});
        }
        
        for (Json::Value::Members::iterator bbIdx = bbMember.begin();
             bbIdx != bbMember.end(); bbIdx++) {
            string blockKey = *bbIdx;
            Json::Value blockJson = regionJson[blockKey];
            uint64_t bbAddress = GetID(blockJson["address"]);
            ProcessBlock(this->blockMaps[bbAddress], bodyRegion, blockJson["ops"]);
        }
        funcOpData.push_back(fOp);
        opBuilder.setInsertionPointAfter(fOp.getOperation());
    }
}

void PluginServer::LocalDeclOpJsonDeSerialize(const string& data)
{
    Json::Value root;
    Json::Reader reader;
    Json::Value node;
    reader.parse(data, root);

    Json::Value::Members operation = root.getMemberNames();

    context.getOrLoadDialect<PluginDialect>();
    opBuilder = mlir::OpBuilder(&context);
    for (Json::Value::Members::iterator iter = operation.begin(); iter != operation.end(); iter++) {
        string operationKey = *iter;
        node = root[operationKey];
        int64_t id = GetID(node["id"]);
        Json::Value attributes = node["attributes"];
        map<string, string> declAttributes;
        JsonGetAttributes(attributes, declAttributes);
	    string symName = declAttributes["symName"];
	    uint64_t typeID = atol(declAttributes["typeID"].c_str());
	    uint64_t typeWidth = atol(declAttributes["typeWidth"].c_str());
        auto location = opBuilder.getUnknownLoc();
        LocalDeclOp op = opBuilder.create<LocalDeclOp>(location, id, symName, typeID, typeWidth);
        decls.push_back(op);
    }
}
void PluginServer::LoopOpsJsonDeSerialize(const string& data)
{
    Json::Value root;
    Json::Reader reader;
    Json::Value node;
    reader.parse(data, root);

    Json::Value::Members operation = root.getMemberNames();
    context.getOrLoadDialect<PluginDialect>();
    mlir::OpBuilder builder(&context);
    for (Json::Value::Members::iterator iter = operation.begin(); iter != operation.end(); iter++) {
        string operationKey = *iter;
        node = root[operationKey];
        int64_t id = GetID(node["id"]);
        Json::Value attributes = node["attributes"];
        map<string, string> loopAttributes;
        JsonGetAttributes(attributes, loopAttributes);
        uint32_t index = atoi(attributes["index"].asString().c_str());
        uint64_t innerId = atol(loopAttributes["innerLoopId"].c_str());
        uint64_t outerId = atol(loopAttributes["outerLoopId"].c_str());
        uint32_t numBlock = atoi(loopAttributes["numBlock"].c_str());
        auto location = builder.getUnknownLoc();
        LoopOp op = builder.create<LoopOp>(location, id, index, innerId, outerId, numBlock);
        loops.push_back(op);
    }
}

void PluginServer::LoopOpJsonDeSerialize(const string& data)
{
    Json::Value root;
    Json::Reader reader;
    reader.parse(data, root);

    context.getOrLoadDialect<PluginDialect>();
    mlir::OpBuilder builder(&context);

    uint64_t id = GetID(root["id"]);
    Json::Value attributes = root["attributes"];
    uint32_t index = atoi(attributes["index"].asString().c_str());
    uint64_t innerLoopId = atol(attributes["innerLoopId"].asString().c_str());
    uint64_t outerLoopId = atol(attributes["outerLoopId"].asString().c_str());
    uint32_t numBlock = atoi(attributes["numBlock"].asString().c_str());
    auto location = builder.getUnknownLoc();
    loop = builder.create<LoopOp>(location, id, index, innerLoopId, outerLoopId, numBlock);
}

void PluginServer::EdgesJsonDeSerialize(const string& data)
{
    Json::Value root;
    Json::Reader reader;
    Json::Value node;
    reader.parse(data, root);

    Json::Value::Members operation = root.getMemberNames();
    context.getOrLoadDialect<PluginDialect>();
    mlir::OpBuilder builder(&context);
    for (Json::Value::Members::iterator iter = operation.begin(); iter != operation.end(); iter++) {
        string operationKey = *iter;
        node = root[operationKey];
        uint64_t src = atol(node["src"].asString().c_str());
        uint64_t dest = atol(node["dest"].asString().c_str());
        pair<mlir::Block*, mlir::Block*> e;
        auto iterSrc = this->blockMaps.find(src);
        if(iterSrc != blockMaps.end())
            edge.first = iterSrc->second;
        else
            edge.first = nullptr;

        auto iterDest = this->blockMaps.find(dest);
        if(iterDest != blockMaps.end())
            edge.second = iterDest->second;
        else
            edge.second = nullptr;

        edges.push_back(e);
    }
}

void PluginServer::EdgeJsonDeSerialize(const string& data)
{
    Json::Value root;
    Json::Reader reader;
    reader.parse(data, root);
    uint64_t src = atol(root["src"].asString().c_str());
    uint64_t dest = atol(root["dest"].asString().c_str());
    auto iterSrc = this->blockMaps.find(src);
    if(iterSrc != blockMaps.end())
        edge.first = iterSrc->second;
    else
        edge.first = nullptr;

    auto iterDest = this->blockMaps.find(dest);
    if(iterDest != blockMaps.end())
        edge.second = iterDest->second;
    else
        edge.second = nullptr;
}

void PluginServer::IdsJsonDeSerialize(const string& data)
{
    Json::Value root;
    Json::Reader reader;
    Json::Value node;
    reader.parse(data, root);

    Json::Value::Members operation = root.getMemberNames();
    context.getOrLoadDialect<PluginDialect>();
    mlir::OpBuilder builder(&context);
    for (Json::Value::Members::iterator iter = operation.begin(); iter != operation.end(); iter++) {
        string operationKey = *iter;
        node = root[operationKey];
        uint64_t id = atol(node["id"].asString().c_str());
        idsResult.push_back(id);
    }
}

void PluginServer::CallOpJsonDeSerialize(const string& data)
{
    Json::Value node;
    Json::Reader reader;
    reader.parse(data, node);
    Json::Value operandJson = node["operands"];
    Json::Value::Members operandMember = operandJson.getMemberNames();
    llvm::SmallVector<mlir::Value, 4> ops;
    for (Json::Value::Members::iterator opIter = operandMember.begin();
            opIter != operandMember.end(); opIter++) {
        string key = *opIter;
        mlir::Value opValue = ValueJsonDeSerialize(operandJson[key.c_str()]);
        ops.push_back(opValue);
    }
    int64_t id = GetID(node["id"]);
    mlir::StringRef callName(node["callee"].asString());
    mlir::Type retType = TypeJsonDeSerialize(node["retType"].toStyledString());
    CallOp op = opBuilder.create<CallOp>(opBuilder.getUnknownLoc(),
                                            id, callName, ops, retType);
    opData.push_back(op.getOperation());
}

void PluginServer::CondOpJsonDeSerialize(const string& data)
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
    assert (this->blockMaps.find(tbaddr) != this->blockMaps.end());
    assert (this->blockMaps.find(fbaddr) != this->blockMaps.end());
    mlir::Block* tb = this->blockMaps[tbaddr];
    mlir::Block* fb = this->blockMaps[fbaddr];
    IComparisonCode iCode = IComparisonCode(
            atoi(node["condCode"].asString().c_str()));
    CondOp op = opBuilder.create<CondOp>(opBuilder.getUnknownLoc(), id,
                address, iCode, LHS, RHS, tb, fb, tbaddr, fbaddr,
                trueLabel, falseLabel);
    opData.push_back(op.getOperation());
}

void PluginServer::RetOpJsonDeSerialize(const string& data)
{
    Json::Value node;
    Json::Reader reader;
    reader.parse(data, node);
    int64_t address = GetID(node["address"]);
    RetOp op = opBuilder.create<RetOp>(opBuilder.getUnknownLoc(), address);
    opData.push_back(op.getOperation());
}

void PluginServer::FallThroughOpJsonDeSerialize(const string& data)
{
    Json::Value node;
    Json::Reader reader;
    reader.parse(data, node);
    int64_t address = GetID(node["address"]);
    int64_t destaddr = GetID(node["destaddr"]);
    assert (this->blockMaps.find(destaddr) != this->blockMaps.end());
    mlir::Block* succ = this->blockMaps[destaddr];
    FallThroughOp op = opBuilder.create<FallThroughOp>(opBuilder.getUnknownLoc(),
                                                        address, succ, destaddr);
    opData.push_back(op.getOperation());
}

void PluginServer::PhiOpJsonDeSerialize(const string& data)
{
    Json::Value node;
    Json::Reader reader;
    reader.parse(data, node);
    Json::Value operandJson = node["operands"];
    Json::Value::Members operandMember = operandJson.getMemberNames();
    llvm::SmallVector<mlir::Value, 4> ops;
    for (Json::Value::Members::iterator opIter = operandMember.begin();
            opIter != operandMember.end(); opIter++) {
        string key = *opIter;
        mlir::Value opValue = ValueJsonDeSerialize(operandJson[key.c_str()]);
        ops.push_back(opValue);
    }
    int64_t id = GetID(node["id"]);
    uint32_t capacity = atoi(node["capacity"].asString().c_str());
    uint32_t nArgs = atoi(node["nArgs"].asString().c_str());
    mlir::Type retType = TypeJsonDeSerialize(node["retType"].toStyledString());
    PhiOp op = opBuilder.create<PhiOp>(opBuilder.getUnknownLoc(),
                                        id, capacity, nArgs, ops, retType);
    opData.push_back(op.getOperation());
}

void PluginServer::AssignOpJsonDeSerialize(const string& data)
{
    Json::Value node;
    Json::Reader reader;
    reader.parse(data, node);
    Json::Value operandJson = node["operands"];
    Json::Value::Members operandMember = operandJson.getMemberNames();
    llvm::SmallVector<mlir::Value, 4> ops;
    for (Json::Value::Members::iterator opIter = operandMember.begin();
            opIter != operandMember.end(); opIter++) {
        string key = *opIter;
        mlir::Value opValue = ValueJsonDeSerialize(operandJson[key.c_str()]);
        ops.push_back(opValue);
    }
    int64_t id = GetID(node["id"]);
    IExprCode iCode = IExprCode(atoi(node["exprCode"].asString().c_str()));
    mlir::Type retType = TypeJsonDeSerialize(node["retType"].toStyledString());
    AssignOp op = opBuilder.create<AssignOp>(opBuilder.getUnknownLoc(),
                                                id, iCode, ops, retType);
    opData.push_back(op.getOperation());
}

void PluginServer::GetPhiOpsJsonDeSerialize(const string& data)
{
    opData.clear();
    Json::Value root;
    Json::Reader reader;
    Json::Value node;
    reader.parse(data, root);

    Json::Value::Members operation = root.getMemberNames();
    context.getOrLoadDialect<PluginDialect>();
    mlir::OpBuilder builder(&context);
    for (Json::Value::Members::iterator iter = operation.begin(); iter != operation.end(); iter++) {
        string operationKey = *iter;
        node = root[operationKey];
        PhiOpJsonDeSerialize(node.toStyledString());
    }
}

/* 线程函数，执行用户注册函数，客户端返回数据后退出 */
static void ExecCallbacks(const string& name)
{
    PluginServer::GetInstance()->ExecFunc(name);
}

void PluginServer::ServerSend(ServerReaderWriter<ServerMsg, ClientMsg>* stream, const string& key,
    const string& value)
{
    ServerMsg serverMsg;
    serverMsg.set_attribute(key);
    serverMsg.set_value(value);
    stream->Write(serverMsg);
}

/* 处理从client接收到的消息 */
int PluginServer::ClientMsgProc(ServerReaderWriter<ServerMsg, ClientMsg>* stream, const string& attribute,
    const string& value)
{
    if ((attribute != "injectPoint") && (attribute != apiFuncName)) {
        JsonDeSerialize(attribute, value);
        return 0;
    }
    if (attribute == "injectPoint") {
        std::thread userfunc(ExecCallbacks, value);
        userfunc.detach();
    }
    if ((attribute == apiFuncName) && (value == "done")) {
        SemPost();
    }

    while (1) {
        SemWait();
        UserFunStateEnum state = GetUserFunState();
        if (state == STATE_END) {
            ServerSend(stream, "userFunc", "execution completed");
            SetUserFunState(STATE_WAIT_BEGIN);
            break;
        } else if (state == STATE_BEGIN) {
            ServerSend(stream, apiFuncName, apiFuncParams);
            SetUserFunState(STATE_WAIT_RETURN);
            break;
        } else if (state == STATE_WAIT_RETURN) {
            if ((attribute == apiFuncName) && (value == "done")) {
                SetUserFunState(STATE_RETURN);
                SetApiFuncName(""); // 已通知，清空
                ClientReturnSemPost();
            }
        }
    }
    return 0;
}

void PluginServer::ExecFunc(const string& value)
{
    int index = value.find_first_of(":");
    string point = value.substr(0, index);
    string name = value.substr(index + 1, -1);
    InjectPoint inject = (InjectPoint)atoi(point.c_str());
    if (inject >= HANDLE_MAX) {
        return;
    }

    auto it = userFunc.find(inject);
    if (it != userFunc.end()) {
        for (auto& funcSet : it->second) {
            if (funcSet.GetName() == name) {
                UserFunc func = funcSet.GetFunc();
                func(); // 执行用户注册函数
                SetUserFunState(STATE_END);
                SemPost();
            }
        }
    }
}

void PluginServer::ParseArgv(const string& data)
{
    Json::Value root;
    Json::Reader reader;
    reader.parse(data, root);

    Json::Value::Members member = root.getMemberNames();
    for (Json::Value::Members::iterator iter = member.begin(); iter != member.end(); iter++) {
        string key = *iter;
        string value = root[key].asString();
        args[key] = value;
    }
}

void PluginServer::SendRegisteredUserFunc(ServerReaderWriter<ServerMsg, ClientMsg>* stream)
{
    for (auto it = userFunc.begin(); it != userFunc.end(); it++) {
        string key = "injectPoint";
        for (auto& funcSet : it->second) {
            string value = std::to_string(it->first) + ":";
            value += funcSet.GetName();
            ServerSend(stream, key, value);
        }
    }
    ServerSend(stream, "injectPoint", "finished");
}

Status PluginServer::ReceiveSendMsg(ServerContext* context, ServerReaderWriter<ServerMsg, ClientMsg>* stream)
{
    ClientMsg clientMsg;
    ServerMsg serverMsg;
    
    while (stream->Read(&clientMsg)) {
        LOGD("rec from client:%s,%s\n", clientMsg.attribute().c_str(), clientMsg.value().c_str());
        string attribute = clientMsg.attribute();
        if (attribute == "start") {
            string arg = clientMsg.value();
            ParseArgv(arg);
            
            ServerSend(stream, "start", "ok");
            SendRegisteredUserFunc(stream);
        } else if (attribute == "stop") {
            ServerSend(stream, "stop", "ok");
            SetShutdownFlag(true);    // 关闭标志
        } else {
            ClientMsgProc(stream, attribute, clientMsg.value());
        }
    }
    return Status::OK;
}

static void ServerExitThread(void)
{
    int delay = 100000;
    pid_t initPid = 1;
    while (1) {
        if (g_service.GetShutdownFlag() || (getppid() == initPid)) {
            g_server->Shutdown();
            break;
        }

        usleep(delay);
    }
}

static void TimeoutFunc(union sigval sig)
{
    int delay = 1; // server延时1秒等待client发指令关闭，若client异常,没收到关闭指令，延时1秒自动关闭
    LOGW("server ppid:%d timeout!\n", getppid());
    PluginServer::GetInstance()->SetUserFunState(STATE_TIMEOUT);
    sleep(delay);
    PluginServer::GetInstance()->SetShutdownFlag(true);
}

void PluginServer::TimerStart(int interval)    // interval:单位ms
{
    int msTons = 1000000; // ms转ns倍数
    int msTos = 1000; // s转ms倍数
    struct itimerspec time_value;
    time_value.it_value.tv_sec = (interval / msTos);
    time_value.it_value.tv_nsec = (interval % msTos) * msTons;
    time_value.it_interval.tv_sec = 0;
    time_value.it_interval.tv_nsec = 0;
    
    timer_settime(timerId, 0, &time_value, NULL);
}

void PluginServer::TimerInit(void)
{
    struct sigevent evp;
    int sival = 123; // 传递整型参数，可以自定义
    memset(&evp, 0, sizeof(struct sigevent));
    evp.sigev_value.sival_ptr = timerId;
    evp.sigev_value.sival_int = sival;
    evp.sigev_notify = SIGEV_THREAD;
    evp.sigev_notify_function = TimeoutFunc;

    if (timer_create(CLOCK_REALTIME, &evp, &timerId) == -1) {
        LOGE("timer create fail\n");
    }
}

void RunServer(int timeout, string& port) // port由client启动server时传入
{
    string serverAddress = "0.0.0.0:" + port;
    
    ServerBuilder builder;
    int serverPort = 0;
    // Listen on the given address without any authentication mechanism.
    builder.AddListeningPort(serverAddress, grpc::InsecureServerCredentials(), &serverPort);
    
    // Register "service" as the instance through which we'll communicate with
    // clients. In this case, it corresponds to an *synchronous* service.
    builder.RegisterService(&g_service);
    // Finally assemble the server.
    g_server = std::unique_ptr<Server>(builder.BuildAndStart());
    LOGI("Server ppid:%d listening on %s\n", getppid(), serverAddress.c_str());
    if (serverPort != atoi(port.c_str())) {
        LOGW("server start fail\n");
        return;
    }

    g_service.SetShutdownFlag(false);
    g_service.SetTimeout(timeout);
    g_service.TimerInit();
    g_service.SetUserFunState(STATE_WAIT_BEGIN);
    g_service.SemInit();

    RegisterCallbacks();
    
    std::thread serverExtiThread(ServerExitThread);
    serverExtiThread.join();

    // Wait for the server to shutdown. Note that some other thread must be
    // responsible for shutting down the server for this call to ever return.
    g_server->Wait();
    g_service.SemDestroy();
    LOGI("server ppid:%d quit!\n", getppid());
}
} // namespace PinServer

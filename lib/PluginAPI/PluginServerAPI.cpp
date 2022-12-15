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
   Description: This file contains the implementation of the PluginAPI_Server class
*/

#include "PluginAPI/PluginServerAPI.h"
#include "PluginServer/PluginLog.h"

namespace PluginAPI {
using namespace PinServer;
using namespace mlir::Plugin;

int CheckAttribute(string &attribute)
{
    /* if (attribute == "") {
        printf("param attribute is NULL,check fail!\n");
        return -1;
    } */
    return 0;
}

int CheckID(uintptr_t id)
{
    return 0;
}

static uint64_t GetValueId(mlir::Value v)
{
    mlir::Operation *op = v.getDefiningOp();
    if (auto mOp = llvm::dyn_cast<MemOp>(op)) {
        return mOp.id();
    } else if (auto ssaOp = llvm::dyn_cast<SSAOp>(op)) {
        return ssaOp.id();
    } else if (auto cstOp = llvm::dyn_cast<ConstOp>(op)) {
        return cstOp.id();
    } else if (auto phOp = llvm::dyn_cast<PlaceholderOp>(op)) {
        return phOp.id();
    }
    return 0;
}

void PluginServerAPI::WaitClientResult(const string& funName, const string& params)
{
    PluginServer *server = PluginServer::GetInstance();
    server->SetApiFuncName(funName);
    server->SetApiFuncParams(params);
    server->SetUserFunState(STATE_BEGIN);
    server->SemPost();
    while (1) {
        server->ClientReturnSemWait();
        if (server->GetUserFunState() == STATE_RETURN) { // wait client result
            server->SetUserFunState(STATE_WAIT_BEGIN);
            break;
        }
    }
}

bool PluginServerAPI::SetCurrentDefInSSA(uint64_t varId, uint64_t defId)
{
    Json::Value root;
    string funName = __func__;
    root["varId"] = std::to_string(varId);
    root["defId"] = std::to_string(defId);
    string params = root.toStyledString();
    WaitClientResult(funName, params);
    return PluginServer::GetInstance()->GetBoolResult();
}

mlir::Value PluginServerAPI::GetCurrentDefFromSSA(uint64_t varId)
{
    Json::Value root;
    string funName = __func__;
    root["varId"] = std::to_string(varId);
    string params = root.toStyledString();
    WaitClientResult(funName, params);
    return PluginServer::GetInstance()->GetValueResult();
}

mlir::Value PluginServerAPI::CopySSAOp(uint64_t id)
{
    Json::Value root;
    string funName = __func__;
    root["id"] = std::to_string(id);
    string params = root.toStyledString();
    WaitClientResult(funName, params);
    return PluginServer::GetInstance()->GetValueResult();
}

mlir::Value PluginServerAPI::CreateSSAOp(mlir::Type t)
{
    Json::Value root;
    string funName = __func__;
    auto baseTy = t.dyn_cast<PluginIR::PluginTypeBase>();
    root = PluginServer::GetInstance()->TypeJsonSerialize(baseTy);
    string params = root.toStyledString();
    WaitClientResult(funName, params);
    return PluginServer::GetInstance()->GetValueResult();
}

vector<FunctionOp> PluginServerAPI::GetFunctionOpResult(const string& funName, const string& params)
{
    WaitClientResult(funName, params);
    vector<FunctionOp> retOps = PluginServer::GetInstance()->GetFunctionOpResult();
    return retOps;
}

vector<FunctionOp> PluginServerAPI::GetAllFunc()
{
    Json::Value root;
    string funName = __func__;
    string params = root.toStyledString();

    return GetFunctionOpResult(funName, params);
}

PhiOp PluginServerAPI::GetPhiOp(uint64_t id)
{
    Json::Value root;
    string funName = __func__;
    root["id"] = std::to_string(id);
    string params = root.toStyledString();
    WaitClientResult(funName, params);
    vector<mlir::Operation*> opRet = PluginServer::GetInstance()->GetOpResult();
    return llvm::dyn_cast<PhiOp>(opRet[0]);
}

CallOp PluginServerAPI::GetCallOp(uint64_t id)
{
    Json::Value root;
    string funName = __func__;
    root["id"] = std::to_string(id);
    string params = root.toStyledString();
    WaitClientResult(funName, params);
    vector<mlir::Operation*> opRet = PluginServer::GetInstance()->GetOpResult();
    return llvm::dyn_cast<CallOp>(opRet[0]);
}

bool PluginServerAPI::SetLhsInCallOp(uint64_t callId, uint64_t lhsId)
{
    Json::Value root;
    string funName = __func__;
    root["callId"] = std::to_string(callId);
    root["lhsId"] = std::to_string(lhsId);
    string params = root.toStyledString();
    WaitClientResult(funName, params);
    return PluginServer::GetInstance()->GetBoolResult();
}

bool PluginServerAPI::AddArgInPhiOp(uint64_t phiId,
                                    uint64_t argId,
                                    uint64_t predId,
                                    uint64_t succId)
{
    Json::Value root;
    string funName = __func__;
    root["phiId"] = std::to_string(phiId);
    root["argId"] = std::to_string(argId);
    root["predId"] = std::to_string(predId);
    root["succId"] = std::to_string(succId);
    string params = root.toStyledString();
    WaitClientResult(funName, params);
    return PluginServer::GetInstance()->GetBoolResult();
}

uint64_t PluginServerAPI::CreateCondOp(uint64_t blockId, IComparisonCode iCode,
                                       uint64_t lhs, uint64_t rhs,
                                       uint64_t tbaddr, uint64_t fbaddr)
{
    Json::Value root;
    string funName = __func__;
    root["blockId"] = std::to_string(blockId);
    root["condCode"] = std::to_string(static_cast<int32_t>(iCode));
    root["lhsId"] = std::to_string(lhs);
    root["rhsId"] = std::to_string(rhs);
    root["tbaddr"] = std::to_string(tbaddr);
    root["fbaddr"] = std::to_string(fbaddr);
    string params = root.toStyledString();
    WaitClientResult(funName, params);
    return PluginServer::GetInstance()->GetIdResult();
}

uint64_t PluginServerAPI::CreateAssignOp(uint64_t blockId, IExprCode iCode, vector<uint64_t> &argIds)
{
    Json::Value root;
    string funName = __func__;
    root["blockId"] = std::to_string(blockId);
    root["exprCode"] = std::to_string(static_cast<int32_t>(iCode));
    Json::Value item;
    size_t idx = 0;
    for (auto v : argIds) {
        string idStr = "id" + std::to_string(idx++);
        item[idStr] = std::to_string(v);
    }
    root["argIds"] = item;
    string params = root.toStyledString();
    WaitClientResult(funName, params);
    return PluginServer::GetInstance()->GetIdResult();
}

uint64_t PluginServerAPI::CreateCallOp(uint64_t blockId, uint64_t funcId,
                                       vector<uint64_t> &argIds)
{
    Json::Value root;
    string funName = __func__;
    root["blockId"] = std::to_string(blockId);
    root["funcId"] = std::to_string(funcId);
    Json::Value item;
    size_t idx = 0;
    for (auto v : argIds) {
        string idStr = "id" + std::to_string(idx++);
        item[idStr] = std::to_string(v);
    }
    root["argIds"] = item;
    string params = root.toStyledString();
    WaitClientResult(funName, params);
    return PluginServer::GetInstance()->GetIdResult();
}

mlir::Value PluginServerAPI::CreateConstOp(mlir::Attribute attr, mlir::Type type)
{
    Json::Value root;
    string funName = __func__;
    auto baseTy = type.dyn_cast<PluginIR::PluginTypeBase>();
    root = PluginServer::GetInstance()->TypeJsonSerialize(baseTy);
    string valueStr;
    if (type.isa<PluginIR::PluginIntegerType>()) {
        valueStr = std::to_string(attr.cast<mlir::IntegerAttr>().getInt());
    }
    root["value"] = valueStr;
    string params = root.toStyledString();
    WaitClientResult(funName, params);
    return PluginServer::GetInstance()->GetValueResult();
}

mlir::Value PluginServerAPI::GetResultFromPhi(uint64_t phiId)
{
    Json::Value root;
    string funName = __func__;
    root["id"] = std::to_string(phiId);
    string params = root.toStyledString();
    WaitClientResult(funName, params);
    return PluginServer::GetInstance()->GetValueResult();
}

PhiOp PluginServerAPI::CreatePhiOp(uint64_t argId, uint64_t blockId)
{
    Json::Value root;
    string funName = __func__;
    root["blockId"] = std::to_string(blockId);
    root["argId"] = std::to_string(argId);
    string params = root.toStyledString();
    WaitClientResult(funName, params);
    vector<mlir::Operation*> opRet = PluginServer::GetInstance()->GetOpResult();
    return llvm::dyn_cast<PhiOp>(opRet[0]);
}

mlir::Value PluginServerAPI::ConfirmValue(mlir::Value v)
{
    Json::Value root;
    string funName = "ConfirmValue";
    uint64_t valId = GetValueId(v);
    root["valId"] = std::to_string(valId);
    string params = root.toStyledString();
    WaitClientResult(funName, params);
    return PluginServer::GetInstance()->GetValueResult();
}

mlir::Value PluginServerAPI::BuildMemRef(PluginIR::PluginTypeBase type,
                                         mlir::Value base, mlir::Value offset)
{
    Json::Value root;
    string funName = "BuildMemRef";
    uint64_t baseId = GetValueId(base);
    uint64_t offsetId = GetValueId(offset);
    root["baseId"] = baseId;
    root["offsetId"] = offsetId;
    root["type"] = (PluginServer::GetInstance()->TypeJsonSerialize(type).toStyledString());
    string params = root.toStyledString();
    WaitClientResult(funName, params);
    return PluginServer::GetInstance()->GetValueResult();
}

PluginIR::PluginTypeID PluginServerAPI::GetTypeCodeFromString(string type)
{
    if (type == "VoidTy") {
        return PluginIR::PluginTypeID::VoidTyID;
    }else if (type == "UIntegerTy1") {
        return PluginIR::PluginTypeID::UIntegerTy1ID;
    }else if (type == "UIntegerTy8") {
        return PluginIR::PluginTypeID::UIntegerTy8ID;
    }else if (type == "UIntegerTy16") {
        return PluginIR::PluginTypeID::UIntegerTy16ID;
    }else if (type == "UIntegerTy32") {
        return PluginIR::PluginTypeID::UIntegerTy32ID;
    }else if (type == "UIntegerTy64") {
        return PluginIR::PluginTypeID::UIntegerTy64ID;
    }else if (type == "IntegerTy1") {
        return PluginIR::PluginTypeID::IntegerTy1ID;
    }else if (type == "IntegerTy8") {
        return PluginIR::PluginTypeID::IntegerTy8ID;
    }else if (type == "IntegerTy16") {
        return PluginIR::PluginTypeID::IntegerTy16ID;
    }else if (type == "IntegerTy32") {
        return PluginIR::PluginTypeID::IntegerTy32ID;
    }else if (type == "IntegerTy64") {
        return PluginIR::PluginTypeID::IntegerTy64ID;
    }else if (type == "BooleanTy") {
        return PluginIR::PluginTypeID::BooleanTyID;
    }else if (type == "FloatTy") {
        return PluginIR::PluginTypeID::FloatTyID;
    }else if (type == "DoubleTy") {
        return PluginIR::PluginTypeID::DoubleTyID;
    }
    
    return PluginIR::PluginTypeID::UndefTyID;
}

vector<LocalDeclOp> PluginServerAPI::GetDeclOperationResult(const string&funName,
                                                            const string& params)
{
    WaitClientResult(funName, params);
    vector<LocalDeclOp> retDecls = PluginServer::GetInstance()->GetLocalDeclResult();
    return retDecls;
}

vector<LocalDeclOp> PluginServerAPI::GetDecls(uint64_t funcID)
{
    Json::Value root;
    string funName("GetLocalDecls");
    root["funcId"] = std::to_string(funcID);
    string params = root.toStyledString();

    return GetDeclOperationResult(funName, params);
}

vector<LoopOp> PluginServerAPI::GetLoopsResult(const string& funName,
                                               const string& params)
{
    WaitClientResult(funName, params);
    vector<LoopOp> loops = PluginServer::GetInstance()->LoopOpsResult();
    return loops;
}

LoopOp PluginServerAPI::GetLoopResult(const string& funName, const string& params)
{
    WaitClientResult(funName, params);
    LoopOp loop = PluginServer::GetInstance()->LoopOpResult();
    return loop;
}

bool PluginServerAPI::GetBoolResult(const string& funName, const string& params)
{
    WaitClientResult(funName, params);
    return PluginServer::GetInstance()->GetBoolResult();
}

pair<mlir::Block*, mlir::Block*> PluginServerAPI::EdgeResult(const string& funName, const string& params)
{
    WaitClientResult(funName, params);
    pair<mlir::Block*, mlir::Block*> e = PluginServer::GetInstance()->EdgeResult();
    return e;
}

vector<pair<mlir::Block*, mlir::Block*> > PluginServerAPI::EdgesResult(const string& funName, const string& params)
{
    WaitClientResult(funName, params);
    vector<pair<mlir::Block*, mlir::Block*> > retEdges = PluginServer::GetInstance()->EdgesResult();
    return retEdges;
}

mlir::Block* PluginServerAPI::BlockResult(const string& funName, const string& params)
{
    WaitClientResult(funName, params);
    uint64_t blockId =  PluginServer::GetInstance()->GetIdResult();
    return PluginServer::GetInstance()->FindBlock(blockId);
}

vector<mlir::Block*> PluginServerAPI::BlocksResult(const string& funName, const string& params)
{
    vector<mlir::Block*> res;
    PluginServer *server = PluginServer::GetInstance();
    WaitClientResult(funName, params);
    vector<uint64_t> blockIds = server->GetIdsResult();
    for(auto b : blockIds) {
        res.push_back(server->FindBlock(b));
    }
    return res;
}

vector<LoopOp> PluginServerAPI::GetLoopsFromFunc(uint64_t funcID)
{
    Json::Value root;
    string funName("GetLoopsFromFunc");
    root["funcId"] = std::to_string(funcID);
    string params = root.toStyledString();

    return GetLoopsResult(funName, params);
}

bool PluginServerAPI::IsDomInfoAvailable()
{
    Json::Value root;
    string funName("IsDomInfoAvailable");
    return GetDomInfoAvaiResult(funName);
}

bool PluginServerAPI::GetDomInfoAvaiResult(const string& funName)
{
    Json::Value root;
    WaitClientResult(funName, root.toStyledString());
    return PluginServer::GetInstance()->GetBoolResult();
}

LoopOp PluginServerAPI::AllocateNewLoop(uint64_t funcID)
{
    Json::Value root;
    string funName("AllocateNewLoop");
    root["funcId"] = std::to_string(funcID);
    string params = root.toStyledString();

    return GetLoopResult(funName, params);
}

LoopOp PluginServerAPI::GetLoopById(uint64_t loopID)
{
    Json::Value root;
    string funName("GetLoopById");
    root["loopId"] = std::to_string(loopID);
    string params = root.toStyledString();

    return GetLoopResult(funName, params);
}

void PluginServerAPI::DeleteLoop(uint64_t loopID)
{
    Json::Value root;
    string funName("DeleteLoop");
    root["loopId"] = std::to_string(loopID);
    string params = root.toStyledString();
    WaitClientResult(funName, params);
}

void PluginServerAPI::AddLoop(uint64_t loopID, uint64_t outerID, uint64_t funcID)
{
    Json::Value root;
    string funName("AddLoop");
    root["loopId"] = loopID;
    root["outerId"] = outerID;
    root["funcId"] = funcID;
    string params = root.toStyledString();
    WaitClientResult(funName, params);
}

void PluginServerAPI::AddBlockToLoop(uint64_t blockID, uint64_t loopID)
{
    Json::Value root;
    string funName("AddBlockToLoop");
    root["blockId"] = blockID;
    root["loopId"] = loopID;
    string params = root.toStyledString();
    WaitClientResult(funName, params);
}

bool PluginServerAPI::IsBlockInLoop(uint64_t loopID, uint64_t blockID)
{
    Json::Value root;
    string funName("IsBlockInside");
    root["loopId"] = std::to_string(loopID);
    root["blockId"] = std::to_string(blockID);
    string params = root.toStyledString();

    return GetBoolResult(funName, params);
}

mlir::Block* PluginServerAPI::GetHeader(uint64_t loopID)
{
    Json::Value root;
    string funName("GetHeader");
    root["loopId"] = std::to_string(loopID);
    string params = root.toStyledString();

    return BlockResult(funName, params);
}

mlir::Block* PluginServerAPI::GetLatch(uint64_t loopID)
{
    Json::Value root;
    string funName("GetLatch");
    root["loopId"] = std::to_string(loopID);
    string params = root.toStyledString();

    return BlockResult(funName, params);
}

void PluginServerAPI::SetHeader(uint64_t loopID, uint64_t blockID)
{
    Json::Value root;
    string funName("SetHeader");
    root["loopId"] = std::to_string(loopID);
    root["blockId"] = std::to_string(blockID);
    string params = root.toStyledString();

    WaitClientResult(funName, params);
}

void PluginServerAPI::SetLatch(uint64_t loopID, uint64_t blockID)
{
    Json::Value root;
    string funName("SetLatch");
    root["loopId"] = std::to_string(loopID);
    root["blockId"] = std::to_string(blockID);
    string params = root.toStyledString();

    WaitClientResult(funName, params);
}

pair<mlir::Block*, mlir::Block*> PluginServerAPI::LoopSingleExit(uint64_t loopID)
{
    Json::Value root;
    string funName("GetLoopSingleExit");
    root["loopId"] = std::to_string(loopID);
    string params = root.toStyledString();

    return EdgeResult(funName, params);
}

vector<pair<mlir::Block*, mlir::Block*> > PluginServerAPI::GetLoopExitEdges(uint64_t loopID)
{
    Json::Value root;
    string funName("GetLoopExits");
    root["loopId"] = std::to_string(loopID);
    string params = root.toStyledString();

    return EdgesResult(funName, params);
}

vector<mlir::Block*> PluginServerAPI::GetLoopBody(uint64_t loopID)
{
    Json::Value root;
    string funName("GetBlocksInLoop");
    root["loopId"] = std::to_string(loopID);
    string params = root.toStyledString();

    return BlocksResult(funName, params);
}

LoopOp PluginServerAPI::GetBlockLoopFather(uint64_t blockID)
{
    Json::Value root;
    string funName("GetBlockLoopFather");
    root["blockId"] = std::to_string(blockID);
    string params = root.toStyledString();

    return GetLoopResult(funName, params);
}

mlir::Block* PluginServerAPI::FindBlock(uint64_t b)
{
    PluginServer *server = PluginServer::GetInstance();
    return server->FindBlock(b);
}

uint64_t PluginServerAPI::FindBasicBlock(mlir::Block* b)
{
    PluginServer *server = PluginServer::GetInstance();
    return server->FindBasicBlock(b);
}

bool PluginServerAPI::InsertValue(uint64_t id, mlir::Value v)
{
    PluginServer *server = PluginServer::GetInstance();
    return server->InsertValue(id, v);
}

bool PluginServerAPI::RedirectFallthroughTarget(FallThroughOp& fop,
                                                uint64_t src, uint64_t dest)
{
    Json::Value root;
    string funName = __func__;
    root["src"] = src;
    root["dest"] = dest;
    string params = root.toStyledString();
    WaitClientResult(funName, params);
    //update server
    PluginServer *server = PluginServer::GetInstance();
    fop->setSuccessor(server->FindBlock(dest), 0);
    return true;
}

mlir::Operation* PluginServerAPI::GetSSADefOperation(uint64_t addr)
{
    return PluginServer::GetInstance()->FindDefOperation(addr);
}

} // namespace Plugin_IR

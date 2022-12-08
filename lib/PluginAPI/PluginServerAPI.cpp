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

uint64_t PluginServerAPI::CreateCondOp(IComparisonCode iCode,
                                       uint64_t lhs, uint64_t rhs)
{
    Json::Value root;
    string funName = __func__;
    root["condCode"] = std::to_string(static_cast<int32_t>(iCode));
    root["lhsId"] = std::to_string(lhs);
    root["rhsId"] = std::to_string(rhs);
    string params = root.toStyledString();
    WaitClientResult(funName, params);
    return PluginServer::GetInstance()->GetIdResult();
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

vector<LocalDeclOp> PluginServerAPI::GetDeclOperationResult(const string&funName, const string& params)
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

vector<LoopOp> PluginServerAPI::GetLoopsResult(const string& funName, const string& params)
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
    return PluginServer::GetInstance()->BoolResult();
}

pair<uint64_t, uint64_t> PluginServerAPI::EdgeResult(const string& funName, const string& params)
{
    WaitClientResult(funName, params);
    pair<uint64_t, uint64_t> e = PluginServer::GetInstance()->EdgeResult();
    return e;
}

vector<pair<uint64_t, uint64_t> > PluginServerAPI::EdgesResult(const string& funName, const string& params)
{
    WaitClientResult(funName, params);
    vector<pair<uint64_t, uint64_t> > retEdges = PluginServer::GetInstance()->EdgesResult();
    return retEdges;
}

uint64_t PluginServerAPI::BlockResult(const string& funName, const string& params)
{
    WaitClientResult(funName, params);
    return PluginServer::GetInstance()->BlockIdResult();
}

vector<uint64_t> PluginServerAPI::BlocksResult(const string& funName, const string& params)
{
    WaitClientResult(funName, params);
    vector<uint64_t> retBlocks = PluginServer::GetInstance()->BlockIdsResult();
    return retBlocks;
}

vector<LoopOp> PluginServerAPI::GetLoopsFromFunc(uint64_t funcID)
{
    Json::Value root;
    string funName("GetLoopsFromFunc");
    root["funcId"] = std::to_string(funcID);
    string params = root.toStyledString();

    return GetLoopsResult(funName, params);
}

// FIXME: 入参void
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

bool PluginServerAPI::IsBlockInLoop(uint64_t loopID, uint64_t blockID)
{
    Json::Value root;
    string funName("IsBlockInside");
    root["loopId"] = std::to_string(loopID);
    root["blockId"] = std::to_string(blockID);
    string params = root.toStyledString();

    return GetBoolResult(funName, params);
}

uint64_t PluginServerAPI::GetHeader(uint64_t loopID)
{
    Json::Value root;
    string funName("GetHeader");
    root["loopId"] = std::to_string(loopID);
    string params = root.toStyledString();

    return BlockResult(funName, params);
}

uint64_t PluginServerAPI::GetLatch(uint64_t loopID)
{
    Json::Value root;
    string funName("GetLatch");
    root["loopId"] = std::to_string(loopID);
    string params = root.toStyledString();

    return BlockResult(funName, params);
}

pair<uint64_t, uint64_t> PluginServerAPI::LoopSingleExit(uint64_t loopID)
{
    Json::Value root;
    string funName("GetLoopSingleExit");
    root["loopId"] = std::to_string(loopID);
    string params = root.toStyledString();

    return EdgeResult(funName, params);
}

vector<pair<uint64_t, uint64_t> > PluginServerAPI::GetLoopExitEdges(uint64_t loopID)
{
    Json::Value root;
    string funName("GetExitEdges");
    root["loopId"] = std::to_string(loopID);
    string params = root.toStyledString();

    return EdgesResult(funName, params);
}

vector<uint64_t> PluginServerAPI::GetLoopBody(uint64_t loopID)
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

} // namespace Plugin_IR

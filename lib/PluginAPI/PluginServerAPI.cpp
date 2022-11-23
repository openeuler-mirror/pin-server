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

vector<FunctionOp> PluginServerAPI::GetOperationResult(const string& funName, const string& params)
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

    return GetOperationResult(funName, params);
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

} // namespace Plugin_IR

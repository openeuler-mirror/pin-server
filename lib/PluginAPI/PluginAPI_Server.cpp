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

#include "PluginAPI/PluginAPI_Server.h"
#include "PluginServer.h"

namespace Plugin_API {
int CheckOpcode(Opcode op)
{
    if ((op == OP_UNDEF) || (op >= OP_END)) {
        printf("op:%d not defined! check Opcode fail\n", op);
        return -1;
    }
    return 0;
}

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

void PluginAPI_Server::WaitClientResult(const string& funName, const string& params)
{
    PluginServer *server = PluginServer::GetInstance();
    server->SetFunName(funName);
    server->Setparams(params);
    server->SetUserFunState(STATE_BEGIN);
    while (1) {
        if (server->GetUserFunState() == STATE_RETURN) { // wait client result
            server->SetUserFunState(STATE_WAIT_BEGIN);
            break;
        }
    }
}

vector<Operation> PluginAPI_Server::GetOperationResult(const string& funName, const string& params)
{
    WaitClientResult(funName, params);
    vector<Operation> retOps = PluginServer::GetInstance()->GetOperationResult();
    return retOps;
}

Decl PluginAPI_Server::GetDeclResult(const string& funName, const string& params)
{
    WaitClientResult(funName, params);
    Decl decl = PluginServer::GetInstance()->GetDeclResult();
    return decl;
}

Type PluginAPI_Server::GetTypeResult(const string& funName, const string& params)
{
    WaitClientResult(funName, params);
    Type type = PluginServer::GetInstance()->GetTypeResult();
    return type;
}

vector<Operation> PluginAPI_Server::SelectOperation(Opcode op, string attribute)
{
    if ((CheckOpcode(op) != 0) || (CheckAttribute(attribute) != 0)) {
        return {};
    }

    Json::Value root;
    root["Opcode"] = op;
    root["string"] = attribute;
    string funName = __func__;
    string params = root.toStyledString();

    return GetOperationResult(funName, params);
}

vector<Operation> PluginAPI_Server::GetAllFunc(string attribute)
{
    if (CheckAttribute(attribute) != 0) {
        return {};
    }
    
    Json::Value root;
    root["string"] = attribute;
    string funName = __func__;
    string params = root.toStyledString();

    return GetOperationResult(funName, params);
}

Decl PluginAPI_Server::SelectDeclByID(uintptr_t id)
{
    if (CheckID(id) != 0) {
        return {};
    }

    Json::Value root;
    root["uintptr_t"] = std::to_string(id);
    string funName = __func__;
    string params = root.toStyledString();

    return GetDeclResult(funName, params);
}

TypeCode PluginAPI_Server::GetTypeCodeFromString(string type)
{
    if (type == "TC_VOID") {
        return TC_VOID;
    } else if (type == "TC_BOOL") {
        return TC_BOOL;
    } else if (type == "TC_U1") {
        return TC_U1;
    } else if (type == "TC_U8") {
        return TC_U8;
    } else if (type == "TC_U16") {
        return TC_U16;
    } else if (type == "TC_U32") {
        return TC_U32;
    } else if (type == "TC_U64") {
        return TC_U64;
    } else if (type == "TC_I1") {
        return TC_I1;
    } else if (type == "TC_I8") {
        return TC_I8;
    } else if (type == "TC_I16") {
        return TC_I16;
    } else if (type == "TC_I32") {
        return TC_I32;
    } else if (type == "TC_I64") {
        return TC_I64;
    } else if (type == "TC_FP16") {
        return TC_FP16;
    } else if (type == "TC_FP32") {
        return TC_FP32;
    } else if (type == "TC_FP64") {
        return TC_FP64;
    } else if (type == "TC_FP80") {
        return TC_FP80;
    }
    return TC_END;
}
} // namespace Plugin_IR

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
    This file contains the declaration of the PluginServer class.
*/

#ifndef GCC_PLUGIN_CLIENT_H
#define GCC_PLUGIN_CLIENT_H

#include <iostream>
#include <memory>
#include <string>
#include <vector>
#include <time.h>
#include <signal.h>

#include <json/json.h>
#include <grpcpp/grpcpp.h>
#include "plugin.grpc.pb.h"
#include "IR/Operation.h"

using Plugin_IR::Opcode;
using Plugin_IR::TypeCode;
using Plugin_IR::DeclCode;
using Plugin_IR::Operation;
using Plugin_IR::Decl;
using Plugin_IR::Type;

using grpc::Server;
using grpc::ServerBuilder;
using grpc::ServerContext;
using grpc::ServerReaderWriter;
using grpc::Status;

using plugin::PluginService;
using plugin::ClientMsg;
using plugin::ServerMsg;

using std::vector;
using std::string;
using std::map;

typedef struct {
    uintptr_t id;
    TypeCode typeCode;
    uint8_t tQual;
    map<string, string> attributes;
} TypeData;

typedef struct {
    uintptr_t id;
    DeclCode declCode;
    map<string, string> attributes;
    TypeData declType;
} DeclData;

typedef struct {
    uintptr_t id;
    Opcode opCode;
    TypeData resultType;
    map<string, DeclData> operands;
    map<string, string> attributes;
} OperationData;

typedef enum {
    STATE_WAIT_BEGIN = 0,
    STATE_BEGIN,
    STATE_WAIT_RETURN,
    STATE_RETURN,
    STATE_END,
} UserFunStateEnum;

typedef std::function<void(void)> UserOptimize;
class PluginServer final : public PluginService::Service {
public:
    Status Optimize(ServerContext* context, ServerReaderWriter<ServerMsg, ClientMsg>* stream) override;
    void ServerSend(ServerReaderWriter<ServerMsg, ClientMsg>* stream, const string& key, const string& value);
    void SendFunc(ServerReaderWriter<ServerMsg, ClientMsg>* stream, const string& attribute, const string& value);
    static PluginServer *GetInstance(void);
    vector<Operation> GetOperationResult(void);
    Decl GetDeclResult(void);
    Type GetTypeResult(void);
    void RegisterUserOptimize(UserOptimize func);
    void SetInjectPoint(const string& inject)
    {
        injectPoint_ = inject;
    }
    int GetShutdownFlag(void)
    {
        return shutdown_;
    }
    void SetShutdownFlag(int flag)
    {
        shutdown_ = flag;
    }
    void SetFunName(const string& name)
    {
        funName = name;
    }
    void Setparams(const string& params)
    {
        funParams = params;
    }
    string &GetFunName(void)
    {
        return funName;
    }
    string &Getparams(void)
    {
        return funParams;
    }
    void SetUserFunState(UserFunStateEnum state)
    {
        userFunState = state;
    }
    UserFunStateEnum GetUserFunState(void)
    {
        return userFunState;
    }
    void SetTimeout(int time)
    {
        timeout_ = time;
    }
    vector<UserOptimize>& getFunc(void)
    {
        return userFunc;
    }
    void OperationJsonDeSerialize(const string& data);
    void DeclJsonDeSerialize(const string& data);
    void TypeJsonDeSerialize(const string& data);
    void JsonDeSerialize(const string& key, const string& data);
    void ParseArgv(const string& data);
    void TimerInit(void);
    void TimerStart(int interval);
    map<string, string>& GetArgs(void)
    {
        return args;
    }
    void JsonGetAttributes(Json::Value node, map<string, string>& attributes);
    
private:
    string injectPoint_;
    int shutdown_;
    UserFunStateEnum userFunState;
    vector<OperationData> opData;
    DeclData dlData;
    TypeData tpData;
    vector<UserOptimize> userFunc;
    string funName;
    string funParams;
    int timeout_;
    timer_t timerId;
    map<string, string> args;
}; // class PluginServer

string UserInit(void);

#endif

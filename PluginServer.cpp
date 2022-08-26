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

#include "Plugin_Log.h"
#include "PluginServer.h"

using namespace Plugin_IR;
using namespace Plugin_Server_LOG;

using std::cout;
using std::endl;
static std::unique_ptr<Server> g_server;
static PluginServer g_service;

PluginServer *PluginServer::GetInstance(void)
{
    return &g_service;
}

void PluginServer::RegisterUserOptimize(UserOptimize func)
{
    if (func != nullptr) {
        userFunc.push_back(func);
    }
}

vector<Operation> PluginServer::GetOperationResult(void)
{
    vector<Operation> retOps;
    for (auto& v : opData) {
        Operation irFunc(OP_FUNCTION);
        irFunc.SetID(v.id);
        irFunc.SetOpcode(v.opCode);
        irFunc.GetResultTypes().SetID(v.resultType.id);
        irFunc.GetResultTypes().SetTypeCode(v.resultType.typeCode);
        irFunc.GetResultTypes().SetTQual(v.resultType.tQual);
        for (auto m = v.resultType.attributes.rbegin(); m != v.resultType.attributes.rend(); m++) {
            irFunc.GetResultTypes().AddAttribute(m->first, m->second);
        }

        for (auto m = v.attributes.rbegin(); m != v.attributes.rend(); m++) {
            irFunc.AddAttribute(m->first, m->second);
        }

        for (auto m = v.operands.rbegin(); m != v.operands.rend(); m++) {
            DeclData declData = v.operands[m->first];
            Decl decl;
            decl.SetID(declData.id);
            decl.SetDeclCode(declData.declCode);
            for (auto iter = declData.attributes.rbegin(); iter != declData.attributes.rend(); iter++) {
                decl.AddAttribute(iter->first, iter->second);
            }

            decl.GetType().SetID(declData.declType.id);
            decl.GetType().SetTypeCode(declData.declType.typeCode);
            decl.GetType().SetTQual(declData.declType.tQual);
            for (auto iter = declData.declType.attributes.rbegin();
                iter != declData.declType.attributes.rend(); iter++) {
                decl.GetType().AddAttribute(iter->first, iter->second);
            }
            irFunc.AddOperand(m->first, decl);
        }
        retOps.push_back(irFunc);
    }
    opData.clear();
    return retOps;
}

Decl PluginServer::GetDeclResult(void)
{
    Decl decl;
    Type type;

    decl.SetID(dlData.id);
    decl.SetDeclCode(dlData.declCode);
    for (auto iter = dlData.attributes.rbegin(); iter != dlData.attributes.rend(); iter++) {
        decl.AddAttribute(iter->first, iter->second);
    }

    type.SetID(dlData.declType.id);
    type.SetTypeCode(dlData.declType.typeCode);
    type.SetTQual(dlData.declType.tQual);
    for (auto iter = dlData.declType.attributes.rbegin();
        iter != dlData.declType.attributes.rend(); iter++) {
        type.AddAttribute(iter->first, iter->second);
    }
    decl.SetType(type);
	
    return decl;
}

Type PluginServer::GetTypeResult(void)
{
    Type type;

    type.SetID(tpData.id);
    type.SetTypeCode(tpData.typeCode);
    type.SetTQual(tpData.tQual);
    for (auto iter = tpData.attributes.rbegin(); iter != tpData.attributes.rend(); iter++) {
        type.AddAttribute(iter->first, iter->second);
    }

    return type;
}

void PluginServer::JsonGetAttributes(Json::Value node, map<string, string>& attributes)
{
    Json::Value::Members attMember = node.getMemberNames();
    for (int i = 0; i < attMember.size(); i++) {
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

void PluginServer::JsonDeSerialize(const string& key, const string& data)
{
    if (key == "OperationResult") {
        OperationJsonDeSerialize(data);
    } else if (key == "DeclResult") {
        DeclJsonDeSerialize(data);
    } else if (key == "TypeResult") {
        TypeJsonDeSerialize(data);
    } else {
        cout << "not Json,key:" << key << ",value:" << data << endl;
    }
}

void PluginServer::OperationJsonDeSerialize(const string& data)
{
    Json::Value root;
    Json::Reader reader;
    Json::Value node;
    reader.parse(data, root);

    OperationData op;
    Json::Value::Members operation = root.getMemberNames();

    for (Json::Value::Members::iterator iter = operation.begin(); iter != operation.end(); iter++) {
        string operationKey = *iter;
        node = root[operationKey];
        op.id = GetID(node["id"]);
        op.opCode = (Opcode)node["opCode"].asInt();
        Json::Value attributes = node["attributes"];
        JsonGetAttributes(attributes, op.attributes);

        Json::Value resultType = node["resultType"];
        op.resultType.id = GetID(resultType["id"]);
        op.resultType.typeCode = (TypeCode)resultType["typeCode"].asInt();
        op.resultType.tQual = resultType["tQual"].asInt();
        JsonGetAttributes(resultType["attributes"], op.resultType.attributes);

        Json::Value operands = node["operands"];
        Json::Value::Members opKey = operands.getMemberNames();
        for (int i = 0; i < opKey.size(); i++) {
            string key = opKey[i];
            Json::Value decl = operands[key.c_str()];
            DeclData operand;
            operand.id = GetID(decl["id"]);
            operand.declCode = (DeclCode)decl["declCode"].asInt();
            JsonGetAttributes(decl["attributes"], operand.attributes);

            Json::Value declType = decl["declType"];
            operand.declType.id = GetID(declType["id"]);
            operand.declType.typeCode = (TypeCode)declType["typeCode"].asInt();
            operand.declType.tQual = declType["tQual"].asInt();
            JsonGetAttributes(declType["attributes"], operand.declType.attributes);
            op.operands[key] = operand;
        }
        opData.push_back(op);
    }
}

void PluginServer::DeclJsonDeSerialize(const string& data)
{
    Json::Value root;
    Json::Reader reader;
    Json::Value node;
    reader.parse(data, root);

    Json::Value decl = root["decl"];
    dlData.id = GetID(decl["id"]);
    dlData.declCode = (DeclCode)decl["declCode"].asInt();
    JsonGetAttributes(decl["attributes"], dlData.attributes);

    Json::Value declType = decl["declType"];
    dlData.declType.id = GetID(declType["id"]);
    dlData.declType.typeCode = (TypeCode)declType["typeCode"].asInt();
    dlData.declType.tQual = declType["tQual"].asInt();

    JsonGetAttributes(declType["attributes"], dlData.declType.attributes);
}

void PluginServer::TypeJsonDeSerialize(const string& data)
{
    Json::Value root;
    Json::Reader reader;
    Json::Value node;
    reader.parse(data, root);

    Json::Value type = root["type"];
    tpData.id = GetID(type["id"]);
    tpData.typeCode = (TypeCode)type["typeCode"].asInt();
    tpData.tQual = type["tQual"].asInt();
    JsonGetAttributes(type["attributes"], tpData.attributes);
}

static void StartUserFunc(void)
{
    while (1) {
        vector<UserOptimize> userFunc = g_service.getFunc();
        for (auto& v : userFunc) {
            if (v != nullptr) {
                v();
            }
        }
        break;
    }
    g_service.SetUserFunState(STATE_END);    // 用户函数已执行完毕
}

void PluginServer::ServerSend(ServerReaderWriter<ServerMsg, ClientMsg>* stream, const string& key,
    const string& value)
{
    ServerMsg serverMsg;
    serverMsg.set_attribute(key);
    serverMsg.set_value(value);
    stream->Write(serverMsg);
}

void PluginServer::SendFunc(ServerReaderWriter<ServerMsg, ClientMsg>* stream, const string& attribute,
    const string& value)
{
    while (1) {
        if (GetUserFunState() == STATE_END) {
            ServerSend(stream, "userFuncEnd", "end");
            TimerStart(timeout_);
            break;
        }

        if (GetUserFunState() == STATE_BEGIN) {
            ServerSend(stream, funName, funParams);
            SetUserFunState(STATE_WAIT_RETURN);
            TimerStart(timeout_);
            break;
        } else if (GetUserFunState() == STATE_WAIT_RETURN) {
            if ((attribute == funName) && (value == "done")) {
                SetUserFunState(STATE_RETURN);
                funName = ""; // 已通知，清空
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

Status PluginServer::Optimize(ServerContext* context, ServerReaderWriter<ServerMsg, ClientMsg>* stream)
{
    ClientMsg clientMsg;
    ServerMsg serverMsg;
    
    while (stream->Read(&clientMsg)) {
        TimerStart(0);    // 关闭定时器
        LOG("rec from client:%s,%s\n", clientMsg.attribute().c_str(), clientMsg.value().c_str());
        string attribute = clientMsg.attribute();
        if (attribute == "start") {
            string arg = clientMsg.value();
            ParseArgv(arg);
            ServerSend(stream, "start", "ok");

            std::thread userfunc(StartUserFunc);
            userfunc.detach();
            ServerSend(stream, "injectPoint", injectPoint_);
            TimerStart(timeout_);
        } else if ((attribute == "injectPoint") || (attribute == funName)) {
            if (clientMsg.value() == "done") {
                SendFunc(stream, attribute, "done");
            } else if (clientMsg.value() == "error") {
                cout << "gcc inject fail,point:" << injectPoint_ << endl;
            }
        } else if (attribute == "stop") {
            ServerSend(stream, "stop", "ok");
            SetShutdownFlag(1);    // 关闭标志
        } else {
            string value = clientMsg.value();
            JsonDeSerialize(attribute, value);
        }
    }
    return Status::OK;
}

static void ServerExitThread(void)
{
    int delay = 100000;
    while (1) {
        if (g_service.GetShutdownFlag()) {
            g_server->Shutdown();
            break;
        }
        usleep(delay);
    }
}

static void TimeoutFunc(union sigval sig)
{
    printf("server timeout!\n");
    PluginServer::GetInstance()->SetShutdownFlag(1);
}

void PluginServer::TimerStart(int interval)    // interval:单位ms
{
    int msTons = 1000000;
    struct itimerspec time_value;
    time_value.it_value.tv_sec = 0;
    time_value.it_value.tv_nsec = interval * msTons;
    time_value.it_interval.tv_sec = 0;
    time_value.it_interval.tv_nsec = 0;
    
    timer_settime(&timerId, 0, &time_value, NULL);
}

void PluginServer::TimerInit(void)
{
    struct sigevent evp;
    int sival = 123;
    memset(&evp, 0, sizeof(struct sigevent));
    evp.sigev_value.sival_ptr = timerId;
    evp.sigev_value.sival_int = sival;
    evp.sigev_notify = SIGEV_THREAD;
    evp.sigev_notify_function = TimeoutFunc;

    if (timer_create(CLOCK_REALTIME, &evp, &timerId) == -1) {
        printf("timer create fail\n");
    }
}

static void RunServer(int timeout)
{
    string server_address("0.0.0.0:50051");
    
    ServerBuilder builder;
    // Listen on the given address without any authentication mechanism.
    builder.AddListeningPort(server_address, grpc::InsecureServerCredentials());
    // Register "service" as the instance through which we'll communicate with
    // clients. In this case, it corresponds to an *synchronous* service.
    builder.RegisterService(&g_service);
    // Finally assemble the server.
    g_server = std::unique_ptr<Server>(builder.BuildAndStart());
    g_service.SetShutdownFlag(0);
    g_service.SetTimeout(timeout);
    g_service.TimerInit();
    g_service.SetUserFunState(STATE_WAIT_BEGIN);

    string inject = UserInit();
    g_service.SetInjectPoint(inject);
    cout << "Server listening on " << server_address << endl;
    
    std::thread serverExtiThread(ServerExitThread);
    serverExtiThread.join();

    // Wait for the server to shutdown. Note that some other thread must be
    // responsible for shutting down the server for this call to ever return.
    g_server->Wait();
}

int main(int argc, char** argv)
{
    int timeout = atoi(argv[0]);
    RunServer(timeout);
    CloseLog();
    return 0;
}

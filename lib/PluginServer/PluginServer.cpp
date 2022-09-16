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

#include "user.h"
#include "PluginServer/PluginLog.h"
#include "PluginServer/PluginServer.h"

namespace PinServer {
using namespace Plugin_IR;

using std::cout;
using std::endl;
static std::unique_ptr<Server> g_server; // grpc对象指针
static PluginServer g_service; // 插件server对象

PluginServer *PluginServer::GetInstance(void)
{
    return &g_service;
}

int PluginServer::RegisterUserFunc(InjectPoint inject, const string& name, UserFunc func)
{
    if ((inject >= HANDLE_MAX) || (func == nullptr)) {
        return -1;
    }

    userFunc[inject].push_back(RecordedUserFunc(name, func));
    return 0;
}

vector<Operation> PluginServer::GetOperationResult(void)
{
    vector<Operation> retOps = opData;
    opData.clear();
    return retOps;
}

Decl PluginServer::GetDeclResult(void)
{
    Decl decl = dlData;
    dlData.GetAttributes().clear();
    return decl;
}

Type PluginServer::GetTypeResult(void)
{
    Type type = tpData;
    tpData.GetAttributes().clear();
    return type;
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

    Operation op;
    Json::Value::Members operation = root.getMemberNames();

    for (Json::Value::Members::iterator iter = operation.begin(); iter != operation.end(); iter++) {
        string operationKey = *iter;
        node = root[operationKey];
        op.SetID(GetID(node["id"]));
        op.SetOpcode((Opcode)node["opCode"].asInt());
        Json::Value attributes = node["attributes"];
        JsonGetAttributes(attributes, op.GetAttributes());

        Json::Value resultType = node["resultType"];
        op.GetResultTypes().SetID(GetID(resultType["id"]));
        op.GetResultTypes().SetTypeCode((TypeCode)resultType["typeCode"].asInt());
        op.GetResultTypes().SetTQual(resultType["tQual"].asInt());
        JsonGetAttributes(resultType["attributes"], op.GetResultTypes().GetAttributes());

        Json::Value operands = node["operands"];
        Json::Value::Members opKey = operands.getMemberNames();
        for (unsigned int i = 0; i < opKey.size(); i++) {
            string key = opKey[i];
            Json::Value decl = operands[key.c_str()];
            Decl operand;
            operand.SetID(GetID(decl["id"]));
            operand.SetDeclCode((DeclCode)decl["declCode"].asInt());
            JsonGetAttributes(decl["attributes"], operand.GetAttributes());

            Json::Value declType = decl["declType"];
            operand.GetType().SetID(GetID(declType["id"]));
            operand.GetType().SetTypeCode((TypeCode)declType["typeCode"].asInt());
            operand.GetType().SetTQual(declType["tQual"].asInt());
            JsonGetAttributes(declType["attributes"], operand.GetType().GetAttributes());
            op.AddOperand(key, operand);
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
    dlData.SetID(GetID(decl["id"]));
    dlData.SetDeclCode((DeclCode)decl["declCode"].asInt());
    JsonGetAttributes(decl["attributes"], dlData.GetAttributes());
    Json::Value declType = decl["declType"];
    Type type;
    type.SetID(GetID(declType["id"]));
    type.SetTypeCode((TypeCode)declType["typeCode"].asInt());
    type.SetTQual(declType["tQual"].asInt());
    JsonGetAttributes(declType["attributes"], type.GetAttributes());
    dlData.SetType(type);
}

void PluginServer::TypeJsonDeSerialize(const string& data)
{
    Json::Value root;
    Json::Reader reader;
    Json::Value node;
    reader.parse(data, root);

    Json::Value type = root["type"];
    tpData.SetID(GetID(type["id"]));
    tpData.SetTypeCode((TypeCode)type["typeCode"].asInt());
    tpData.SetTQual(type["tQual"].asInt());
    JsonGetAttributes(type["attributes"], tpData.GetAttributes());
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

    while (1) {
        UserFunStateEnum state = GetUserFunState();
        if (state == STATE_END) {
            ServerSend(stream, "userFunc", "execution completed");
            SetUserFunState(STATE_WAIT_BEGIN);
            TimerStart(timeout);
            break;
        }

        if (state == STATE_BEGIN) {
            ServerSend(stream, apiFuncName, apiFuncParams);
            SetUserFunState(STATE_WAIT_RETURN);
            TimerStart(timeout);
            break;
        } else if (state == STATE_WAIT_RETURN) {
            if ((attribute == apiFuncName) && (value == "done")) {
                SetUserFunState(STATE_RETURN);
                apiFuncName = ""; // 已通知，清空
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
        TimerStart(0);    // 关闭定时器
        LOGD("rec from client:%s,%s\n", clientMsg.attribute().c_str(), clientMsg.value().c_str());
        string attribute = clientMsg.attribute();
        if (attribute == "start") {
            string arg = clientMsg.value();
            ParseArgv(arg);
            
            ServerSend(stream, "start", "ok");
            SendRegisteredUserFunc(stream);
            TimerStart(timeout);
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
    LOGW("server timeout!\n");
    PluginServer::GetInstance()->SetShutdownFlag(true);
}

void PluginServer::TimerStart(int interval)    // interval:单位ms
{
    int msTons = 1000000; // ms转ns倍数
    struct itimerspec time_value;
    time_value.it_value.tv_sec = 0;
    time_value.it_value.tv_nsec = interval * msTons;
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

static void RunServer(int timeout, string& port) // port由client启动server时传入
{
    string serverAddress = "0.0.0.0:" + port;
    
    ServerBuilder builder;
    // Listen on the given address without any authentication mechanism.
    builder.AddListeningPort(serverAddress, grpc::InsecureServerCredentials());
    // Register "service" as the instance through which we'll communicate with
    // clients. In this case, it corresponds to an *synchronous* service.
    builder.RegisterService(&g_service);
    // Finally assemble the server.
    g_server = std::unique_ptr<Server>(builder.BuildAndStart());
    g_service.SetShutdownFlag(false);
    g_service.SetTimeout(timeout);
    g_service.TimerInit();
    g_service.SetUserFunState(STATE_WAIT_BEGIN);

    RegisterCallbacks();
    LOGI("Server listening on %s\n", serverAddress.c_str());
    
    std::thread serverExtiThread(ServerExitThread);
    serverExtiThread.join();

    // Wait for the server to shutdown. Note that some other thread must be
    // responsible for shutting down the server for this call to ever return.
    g_server->Wait();
}
} // namespace PinServer

int main(int argc, char** argv)
{
    int timeout = atoi(argv[0]);
    std::string port = argv[1];
    PinServer::LogPriority priority = (PinServer::LogPriority)atoi(argv[2]);
    PinServer::SetLogPriority(priority);
    PinServer::RunServer(timeout, port);
    PinServer::CloseLog();
    return 0;
}

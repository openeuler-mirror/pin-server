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

#ifndef PLUGIN_SERVER_H
#define PLUGIN_SERVER_H

#include <memory>
#include <string>
#include <vector>
#include <time.h>
#include <signal.h>
#include <semaphore.h>

#include <json/json.h>
#include <grpcpp/grpcpp.h>
#include "plugin.grpc.pb.h"
#include "IR/Operation.h"

namespace PinServer {
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

typedef enum {
    STATE_WAIT_BEGIN = 0,
    STATE_BEGIN,
    STATE_WAIT_RETURN,
    STATE_RETURN,
    STATE_END,
    STATE_TIMEOUT,
} UserFunStateEnum;

typedef std::function<void(void)> UserFunc;
enum InjectPoint : uint8_t {
    HANDLE_PARSE_TYPE = 0,
    HANDLE_PARSE_DECL,
    HANDLE_PRAGMAS,
    HANDLE_PARSE_FUNCTION,
    HANDLE_BEFORE_IPA,
    HANDLE_AFTER_IPA,
    HANDLE_BEFORE_EVERY_PASS,
    HANDLE_AFTER_EVERY_PASS,
    HANDLE_BEFORE_ALL_PASS,
    HANDLE_AFTER_ALL_PASS,
    HANDLE_COMPILE_END,
    HANDLE_MANAGER_SETUP,
    HANDLE_MAX,
};

// 参考点名称
enum RefPassName {
    PASS_CFG,
    PASS_SSA,
    PASS_LOOP,
};

enum PassPosition {
    PASS_INSERT_AFTER,
    PASS_INSERT_BEFORE,
    PASS_REPLACE,
};

struct ManagerSetupData {
    RefPassName refPassName;
    int passNum; // 指定passName的第几次执行作为参考点
    PassPosition passPosition; // 指定pass是添加在参考点之前还是之后
};

class RecordedUserFunc {
public:
    RecordedUserFunc () = default;
    ~RecordedUserFunc () = default;
    RecordedUserFunc (const string& name, UserFunc func)
    {
        this->name = name;
        this->func = func;
    }
    string GetName(void)
    {
        return name;
    }
    UserFunc GetFunc(void)
    {
        return func;
    }
private:
    string name;
    UserFunc func;
};

class PluginServer final : public PluginService::Service {
public:
    /* 定义的grpc服务端和客户端通信的接口函数 */
    Status ReceiveSendMsg(ServerContext* context, ServerReaderWriter<ServerMsg, ClientMsg>* stream) override;
    /* 服务端发送数据给client接口 */
    void ServerSend(ServerReaderWriter<ServerMsg, ClientMsg>* stream, const string& key, const string& value);
    /* 处理从client接收到的消息 */
    int ClientMsgProc(ServerReaderWriter<ServerMsg, ClientMsg>* stream, const string& attribute, const string& value);
    /* 获取server对象实例,有且只有一个实例对象 */
    static PluginServer *GetInstance(void);
    vector<Operation> GetOperationResult(void);
    Decl GetDeclResult(void);
    Type GetTypeResult(void);
    /* 回调函数接口，用于向server注册用户需要执行的函数 */
    int RegisterUserFunc(InjectPoint inject, UserFunc func);
    int RegisterPassManagerSetup(InjectPoint inject, const ManagerSetupData& passData, UserFunc func);
    /* 执行用户注册的回调函数,根据value查找对应的函数,value格式 InjectPoint:funName */
    void ExecFunc(const string& value);
    /* 将注册点和函数名发到客户端, stream为grpc当前数据流指针 */
    void SendRegisteredUserFunc(ServerReaderWriter<ServerMsg, ClientMsg>* stream);
    bool GetShutdownFlag(void)
    {
        return shutdown;
    }
    void SetShutdownFlag(bool flag)
    {
        shutdown = flag;
    }
    void SetApiFuncName(const string& name)
    {
        apiFuncName = name;
    }
    void SetApiFuncParams(const string& params)
    {
        apiFuncParams = params;
    }
    string &GetApiFuncName(void)
    {
        return apiFuncName;
    }
    string &GetApiFuncParams(void)
    {
        return apiFuncParams;
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
        timeout = time;
    }
    /* 将从client接收到的Operation类型数据反序列化 */
    void OperationJsonDeSerialize(const string& data);
    /* 将从client接收到的Decl类型数据反序列化 */
    void DeclJsonDeSerialize(const string& data);
    /* 将从client接收到的Type类型数据反序列化 */
    void TypeJsonDeSerialize(const string& data);
    /* json反序列化，根据key值分别调用Operation/Decl/Type反序列化接口函数 */
    void JsonDeSerialize(const string& key, const string& data);
    /* 解析客户端发送过来的-fplugin-arg参数，并保存在私有变量args中 */
    void ParseArgv(const string& data);
    void TimerInit(void); // 超时定时器初始化
    void TimerStart(int interval); // 启动定时器，interval为0表示关闭定时器
    map<string, string>& GetArgs(void)
    {
        return args;
    }
    /* 将json格式数据解析成map<string, string>格式 */
    void JsonGetAttributes(Json::Value node, map<string, string>& attributes);
    void SemInit(void)
    {
        sem_init(&sem[0], 0, 0);
        sem_init(&sem[1], 0, 0);
    }
    void SemPost(void) // 开始执行用户函数或者用户函数结束触发该信号量
    {
        sem_post(&sem[0]);
    }
    void SemWait(void)
    {
        sem_wait(&sem[0]);
    }
    void ClientReturnSemPost(void) // client返回数据后触发该信号量
    {
        sem_post(&sem[1]);
    }
    void ClientReturnSemWait(void)
    {
        sem_wait(&sem[1]);
    }
    void SemDestroy(void)
    {
        sem_destroy(&sem[0]);
        sem_destroy(&sem[1]);
    }

private:
    bool shutdown; // 是否关闭server
    /* 用户函数执行状态，client返回结果后为STATE_RETURN,开始执行下一个函数 */
    volatile UserFunStateEnum userFunState;
    vector<Operation> opData;
    Decl dlData;
    Type tpData;
    /* 保存用户注册的回调函数，它们将在注入点事件触发后调用 */
    map<InjectPoint, vector<RecordedUserFunc>> userFunc;
    string apiFuncName; // 保存用户调用PluginAPI的函数名
    string apiFuncParams; // 保存用户调用PluginAPI函数的参数
    int timeout;
    timer_t timerId;
    map<string, string> args; // 保存gcc编译时用户传入参数
    sem_t sem[2];
}; // class PluginServer

void RunServer(int timeout, string& port);
} // namespace PinServer

#endif

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
    主要完成功能：完成和server之间grpc数据解析，提供接口函数获取client数据，提供注册
    函数完成用户事件注册，并在对应事件触发时回调用户函数
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
#include "Dialect/PluginOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Builders.h"
#include "Dialect/PluginTypes.h"

#include "PluginServer/PluginLog.h"
#include "PluginServer/PluginCom.h"
#include "PluginServer/PluginOptBase.h"

namespace PinServer {
using PinLog::PluginLog;
using PinLog::LogPriority;
using PinCom::PluginCom;
using plugin::ClientMsg;
using plugin::ServerMsg;
using PluginOpt::InjectPoint;
using PluginOpt::ManagerSetup;
using PluginOpt::PluginOptBase;

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

class RecordedOpt {
public:
    RecordedOpt() = default;
    ~RecordedOpt() = default;
    RecordedOpt(const string& name, std::shared_ptr<PluginOptBase> opt)
    {
        this->name = name;
        this->opt = opt;
    }
    RecordedOpt(const string& name, const string& param, std::shared_ptr<PluginOptBase> opt)
    {
        this->name = name;
        this->param = param;
        this->opt = opt;
    }
    string GetName()
    {
        return name;
    }
    string GetParam()
    {
        return param;
    }
    std::shared_ptr<PluginOptBase> GetOpt()
    {
        return opt;
    }
private:
    string name;
    string param;
    std::shared_ptr<PluginOptBase> opt;
};

class PluginServer {
public:
    PluginServer(LogPriority priority, const string& port)
    {
        userFunState = STATE_WAIT_BEGIN;
        sem_init(&clientWaitSem, 0, 0);
        sem_init(&clientReturnSem, 0, 0);
        log = PluginLog::GetInstance();
        log->SetPriority(priority);
        this->port = port;
        pluginServerPtr = this;
    }
    ~PluginServer()
    {
        sem_destroy(&clientWaitSem);
        sem_destroy(&clientReturnSem);
        log->LOGI("server ppid:%d quit!\n", getppid());
    }
    /* 处理从client接收到的消息 */
    int ClientMsgProc(const string& attribute, const string& value);
    mlir::MLIRContext *GetContext()
    {
        return this->context;
    }
    void SetOpBuilder(mlir::OpBuilder *builder)
    {
        this->opBuilder = builder;
    }
    mlir::OpBuilder *GetOpBuilder()
    {
        return this->opBuilder;
    }
    /* 回调函数接口，用于向server注册用户需要执行的函数 */
    bool RegisterOpt(std::shared_ptr<PluginOptBase> optBase);
    bool RegisterPassManagerOpt(ManagerSetup& passData, std::shared_ptr<PluginOptBase> optBase);
    map<string, string>& GetArgs()
    {
        return args;
    }
    /* 执行用户注册的回调函数,根据value查找对应的函数,value格式 InjectPoint:funName */
    void ExecFunc(const string& value);
    void RunServer();
    /* 获取server对象实例,有且只有一个实例对象 */
    static PluginServer *GetInstance(void);
    
    int64_t GetIntegerDataResult(const string& funName, const string& params)
    {
        RemoteCallClientWithAPI(funName, params);
        return pluginCom.GetIntegerDataResult();
    }
    string GetStringDataResult(const string& funName, const string& params)
    {
        RemoteCallClientWithAPI(funName, params);
        return pluginCom.GetStringDataResult();
    }
    vector<mlir::Plugin::FunctionOp> GetFunctionOpResult(const string& funName, const string& params)
    {
        RemoteCallClientWithAPI(funName, params);
        return pluginCom.GetFunctionOpResult();
    }

    vector<mlir::Plugin::DeclBaseOp> GetFuncDeclsResult(const string& funName, const string& params)
    {
        RemoteCallClientWithAPI(funName, params);
        return pluginCom.GetFuncDeclsResult();
    }

    mlir::Plugin::FieldDeclOp GetMakeNodeResult(const string& funName, const string& params)
    {
        RemoteCallClientWithAPI(funName, params);
        return pluginCom.GetMakeNodeResult();
    }

    llvm::SmallVector<mlir::Plugin::FieldDeclOp> GetFieldsResult(const string& funName, const string& params)
    {
        RemoteCallClientWithAPI(funName, params);
        return pluginCom.GetFieldsResult();
    }

    mlir::Plugin::DeclBaseOp GetBuildDeclResult(const string& funName, const string& params)
    {
        RemoteCallClientWithAPI(funName, params);
        return pluginCom.GetBuildDeclResult();
    }

    PluginIR::PluginTypeBase GetDeclTypeResult(const string& funName, const string& params)
    {
        RemoteCallClientWithAPI(funName, params);
        return pluginCom.GetDeclTypeResult();
    }

    vector<mlir::Plugin::LocalDeclOp> GetLocalDeclResult(const string& funName, const string& params)
    {
        RemoteCallClientWithAPI(funName, params);
        return pluginCom.GetLocalDeclResult();
    }
    mlir::Plugin::CGnodeOp GetCGnodeOpResult(const string& funName, const string& params)
    {
        RemoteCallClientWithAPI(funName, params);
        return pluginCom.GetCGnodeOpResult();
    }
    mlir::Plugin::LoopOp LoopOpResult(const string& funName, const string& params)
    {
        RemoteCallClientWithAPI(funName, params);
        return pluginCom.LoopOpResult();
    }
    vector<mlir::Plugin::LoopOp> LoopOpsResult(const string& funName, const string& params)
    {
        RemoteCallClientWithAPI(funName, params);
        return pluginCom.LoopOpsResult();
    }
    vector<std::pair<mlir::Block*, mlir::Block*> > EdgesResult(const string& funName, const string& params)
    {
        RemoteCallClientWithAPI(funName, params);
        return pluginCom.EdgesResult();
    }
    std::pair<mlir::Block*, mlir::Block*> EdgeResult(const string& funName, const string& params)
    {
        RemoteCallClientWithAPI(funName, params);
        return pluginCom.EdgeResult();
    }
    vector<mlir::Operation *> GetOpResult(const string& funName, const string& params)
    {
        RemoteCallClientWithAPI(funName, params);
        return pluginCom.GetOpResult();
    }
    bool GetBoolResult(const string& funName, const string& params)
    {
        RemoteCallClientWithAPI(funName, params);
        return pluginCom.GetBoolResult();
    }
    uint64_t GetIdResult(const string& funName, const string& params)
    {
        RemoteCallClientWithAPI(funName, params);
        return pluginCom.GetIdResult();
    }
    vector<uint64_t> GetIdsResult(const string& funName, const string& params)
    {
        RemoteCallClientWithAPI(funName, params);
        return pluginCom.GetIdsResult();
    }
    mlir::Value GetValueResult(const string& funName, const string& params)
    {
        RemoteCallClientWithAPI(funName, params);
        return pluginCom.GetValueResult();
    }
    vector<mlir::Value> GetValuesResult(const string& funName, const string& params)
    {
        RemoteCallClientWithAPI(funName, params);
        return pluginCom.GetValuesResult();
    }
    vector<mlir::Plugin::PhiOp> GetPhiOpsResult(const string& funName, const string& params)
    {
        RemoteCallClientWithAPI(funName, params);
        return pluginCom.GetPhiOpsResult();
    }

    uint64_t GetBlockResult(mlir::Block*);
    void EraseBlock(mlir::Block*);
    mlir::Block* FindBlock(uint64_t);
    void InsertCreatedBlock(uint64_t, mlir::Block*);
    bool HaveBlock(uint64_t id)
    {
        return this->blockMaps.find(id) != this->blockMaps.end();
    }

    void ClearMaps()
    {
        valueMaps.clear();
        blockMaps.clear();
        basicblockMaps.clear();
        opMaps.clear();
    }

    uint64_t FindBasicBlock(mlir::Block*);
    bool InsertValue(uint64_t, mlir::Value);
    bool HaveValue(uint64_t);
    mlir::Value GetValue(uint64_t);
    mlir::Operation* FindOperation(uint64_t);
    bool InsertOperation(uint64_t, mlir::Operation*);
    void RemoteCallClientWithAPI(const string& api, const string& params);

private:
    /* 用户函数执行状态，client返回结果后为STATE_RETURN,开始执行下一个函数 */
    volatile UserFunStateEnum userFunState;
    mlir::MLIRContext *context;
    mlir::OpBuilder* opBuilder = nullptr;

    /* 保存用户注册的回调函数，它们将在注入点事件触发后调用 */
    map<InjectPoint, vector<RecordedOpt>> userOpts;
    string apiFuncName; // 保存用户调用PluginAPI的函数名
    string apiFuncParams; // 保存用户调用PluginAPI函数的参数
    string port; // server使用的端口号
    map<string, string> args; // 保存client编译时用户传入参数
    sem_t clientWaitSem; // 等待client结果信号量
    sem_t clientReturnSem; // client返回结果信号量
    PluginCom pluginCom;
    PluginLog *log;

    std::map<uint64_t, mlir::Value> valueMaps;
    // process Block.
    std::map<uint64_t, mlir::Block*> blockMaps;
    std::map<mlir::Block*, uint64_t> basicblockMaps;
    // std::map<uint64_t, mlir::Operation*> defOpMaps;
    std::map<uint64_t, mlir::Operation*> opMaps; // 保存所有的<gimpleid, op>键值对

    /* 解析客户端发送过来的-fplugin-arg参数，并保存在私有变量args中 */
    void ParseArgv(const string& data);
    void ServerSemPost(const string& port); // server服务起来后通知client
    /* 将注册点和函数名发到客户端, stream为grpc当前数据流指针 */
    void SendRegisteredUserOpts();
    void ExecCallbacks(const string& name);
    static PluginServer *pluginServerPtr;
}; // class PluginServer
} // namespace PinServer

#endif

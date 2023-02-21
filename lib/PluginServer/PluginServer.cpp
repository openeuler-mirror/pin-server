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

#include <string>
#include <vector>
#include <thread>
#include <fcntl.h>
#include <sys/stat.h>

#include "PluginAPI/PluginServerAPI.h"
#include "user/user.h"
#include "PluginServer/PluginServer.h"

namespace PinServer {
using namespace mlir::Plugin;
using namespace PluginOpt;
using std::cout;
using std::endl;
using std::pair;
PluginServer *PluginServer::pluginServerPtr = nullptr;
PluginServer *PluginServer::GetInstance()
{
    return pluginServerPtr;
}

bool PluginServer::RegisterOpt(std::shared_ptr<PluginOptBase> optBase)
{
    InjectPoint inject = optBase->GetInject();
    if ((inject >= HANDLE_MAX) || (optBase == nullptr)) {
        return false;
    }

    string name = "funcname" + std::to_string((uintptr_t)optBase.get());
    userOpts[inject].push_back(RecordedOpt(name, optBase));
    this->context = optBase->GetContext();
    static mlir::OpBuilder opBuilder_temp = mlir::OpBuilder(context);
    opBuilder = &opBuilder_temp;
    return true;
}

bool PluginServer::RegisterPassManagerOpt(ManagerSetup& setupData, std::shared_ptr<PluginOptBase> optBase)
{
    Json::Value root;
    root["refPassName"] = setupData.GetPassName();
    root["passNum"] = setupData.GetPassNum();
    root["passPosition"] = setupData.GetPassPosition();
    string params = root.toStyledString();
    string name = "funcname" + std::to_string((uintptr_t)optBase.get());
    userOpts[HANDLE_MANAGER_SETUP].push_back(RecordedOpt(name, params, optBase));
    this->context = optBase->GetContext();
    static mlir::OpBuilder opBuilder_temp = mlir::OpBuilder(context);
    opBuilder = &opBuilder_temp;
    return true;
}

void PluginServer::EraseBlock(mlir::Block* b)
{
    if (auto bbit = basicblockMaps.find(b); bbit != basicblockMaps.end()) {
        uint64_t addr = bbit->second;
        basicblockMaps[b] = 0;
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

mlir::Operation* PluginServer::FindDefOperation(uint64_t id)
{
    auto iter = this->defOpMaps.find(id);
    assert(iter != this->defOpMaps.end());
    return iter->second;
}

bool PluginServer::InsertDefOperation(uint64_t id, mlir::Operation* op)
{
    auto iter = this->defOpMaps.find(id);
    this->defOpMaps.insert({id, op});
    return true;
}

void PluginServer::InsertCreatedBlock(uint64_t id, mlir::Block* block)
{
    this->blockMaps.insert({id, block});
    this->basicblockMaps.insert({block, id});
}

uint64_t PluginServer::GetBlockResult(mlir::Block* b)
{
    uint64_t newAddr = pluginCom.GetIdResult();
    mlir::OpBuilder opBuilder = mlir::OpBuilder(this->context);
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

bool PluginServer::InsertValue(uint64_t id, mlir::Value v)
{
    auto iter = this->valueMaps.find(id);
    assert(iter == this->valueMaps.end());
    this->valueMaps.insert({id, v});
    return true;
}

bool PluginServer::HaveValue(uint64_t id)
{
    return this->valueMaps.find(id) != this->valueMaps.end();
}

mlir::Value PluginServer::GetValue(uint64_t id)
{
    auto iter = this->valueMaps.find(id);
    assert(iter != this->valueMaps.end());
    return iter->second;
}

void PluginServer::RemoteCallClientWithAPI(const string& api, const string& params)
{
    if (api == "") {
        return;
    }

    apiFuncName = api;
    apiFuncParams = params;
    userFunState = STATE_BEGIN;
    sem_post(&clientWaitSem);
    while (1) {
        sem_wait(&clientReturnSem);
        if (userFunState == STATE_RETURN) { // wait client result
            userFunState = STATE_WAIT_BEGIN;
            break;
        }
    }
}

/* 线程函数，执行用户注册函数，客户端返回数据后退出 */
void PluginServer::ExecCallbacks(const string& name)
{
    ExecFunc(name);
}

/* 处理从client接收到的消息 */
int PluginServer::ClientMsgProc(const string& key, const string& value)
{
    log->LOGD("rec from client:%s,%s\n", key.c_str(), value.c_str());
    if (key == "start") {
        ParseArgv(value);
        pluginCom.ServerSend("start", "ok");
        SendRegisteredUserOpts();
        return 0;
    } else if (key == "stop") {
        pluginCom.ServerSend("stop", "ok");
        pluginCom.ShutDown();
        return 0;
    }
    if ((key != "injectPoint") && (key != apiFuncName)) {
        pluginCom.JsonDeSerialize(key, value);
        return 0;
    }
    if (key == "injectPoint") {
        std::thread userfunc(&PluginServer::ExecCallbacks, this, value);
        userfunc.detach();
    }
    if ((key == apiFuncName) && (value == "done")) {
        sem_post(&clientWaitSem);
    }

    while (1) {
        sem_wait(&clientWaitSem);
        if (userFunState == STATE_END) {
            pluginCom.ServerSend("userFunc", "execution completed");
            userFunState = STATE_WAIT_BEGIN;
            break;
        } else if (userFunState == STATE_BEGIN) {
            pluginCom.ServerSend(apiFuncName, apiFuncParams);
            userFunState = STATE_WAIT_RETURN;
            break;
        } else if (userFunState == STATE_WAIT_RETURN) {
            if ((key == apiFuncName) && (value == "done")) {
                userFunState = STATE_RETURN;
                apiFuncName = ""; // 已通知，清空
                sem_post(&clientReturnSem);
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

    uint64_t param = 0;
    if (inject == HANDLE_MANAGER_SETUP) {
        param = atol(name.substr(name.find_first_of(":") + 1, -1).c_str());
        name = name.substr(0, name.find_first_of(","));
    }

    auto it = userOpts.find(inject);
    if (it != userOpts.end()) {
        for (auto& userOpt : it->second) {
            if (userOpt.GetName() != name) {
                continue;
            }

            if (userOpt.GetOpt()->Gate()) {
                if (it->first == HANDLE_MANAGER_SETUP) {
                    userOpt.GetOpt()->SetFuncAddr(param);
                }
                userOpt.GetOpt()->DoOptimize();
            }
            ClearMaps();
            userFunState = STATE_END;
            sem_post(&clientWaitSem);
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

void PluginServer::SendRegisteredUserOpts()
{
    for (auto it = userOpts.begin(); it != userOpts.end(); it++) {
        string key = "injectPoint";
        for (auto& userOpt : it->second) {
            string value = std::to_string(it->first) + ":";
            if (it->first == HANDLE_MANAGER_SETUP) {
                value += userOpt.GetName() + ",params:" + userOpt.GetParam();
            } else {
                value += userOpt.GetName();
            }
            pluginCom.ServerSend(key, value);
        }
    }
    pluginCom.ServerSend("injectPoint", "finished");
}

void PluginServer::ServerSemPost(const string& port)
{
    mode_t mask = umask(0);
    mode_t mode = 0666; // 权限是rwrwrw，跨进程时，其他用户也要可以访问
    string semFile = "wait_server_startup" + port;
    sem_t *sem = sem_open(semFile.c_str(), O_CREAT, mode, 0);
    umask(mask);
    sem_post(sem);
    sem_close(sem);
}

void PluginServer::RunServer()
{
    if (!pluginCom.RegisterServer(port)) {
        log->LOGE("server start fail,port:%s\n", port.c_str());
        return;
    }
    log->LOGI("Server ppid:%d listening on port:%s\n", getppid(), port.c_str());
    ServerSemPost(port);

    RegisterCallbacks();
    log->LOGI("RunServer: RegisterCallbacks Done.\n");
    pluginCom.Run();
}
} // namespace PinServer

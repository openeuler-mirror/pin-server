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
    This file contains the declaration of the PluginCom class.
    主要完成功能：和client之间通信、数据解析、数据反序列化
*/

#ifndef PLUGIN_COM_H
#define PLUGIN_COM_H

#include "Dialect/PluginOps.h"
#include "Dialect/PluginTypes.h"
#include "PluginServer/PluginJson.h"
#include "PluginServer/PluginGrpc.h"

namespace PinCom {
using PinGrpc::PluginGrpc;
using PinJson::PluginJson;
using std::vector;
using std::string;

class PluginCom {
public:
    bool RegisterServer(const string& port)
    {
        return pluginGrpc.RegisterServer(port);
    }
    void Run()
    {
        pluginGrpc.Run();
    }
    void ShutDown()
    {
        pluginGrpc.ShutDown();
    }
    void ServerSend(const string& key, const string& value)
    {
        pluginGrpc.ServerSend(key, value);
    }
    /* json反序列化，根据key值分别调用Operation/Decl/Type反序列化接口函数 */
    void JsonDeSerialize(const string& key, const string& data);
    int64_t GetIntegerDataResult(void);
    string GetStringDataResult(void);
    vector<mlir::Plugin::FunctionOp> GetFunctionOpResult(void);
    vector<mlir::Plugin::LocalDeclOp> GetLocalDeclResult(void);
    mlir::Plugin::LoopOp LoopOpResult(void);
    vector<mlir::Plugin::LoopOp> LoopOpsResult(void);
    vector<std::pair<mlir::Block*, mlir::Block*> > EdgesResult(void);
    std::pair<mlir::Block*, mlir::Block*> EdgeResult(void);
    vector<mlir::Operation *> GetOpResult(void);
    bool GetBoolResult(void);
    uint64_t GetIdResult(void);
    vector<uint64_t> GetIdsResult(void);
    mlir::Value GetValueResult(void);
    vector<mlir::Plugin::PhiOp> GetPhiOpsResult(void);

private:
    PluginGrpc pluginGrpc;
    PluginJson json;
    int64_t integerResult;
    string stringResult;
    vector<mlir::Plugin::FunctionOp> funcOpData;
    vector<mlir::Plugin::LocalDeclOp> decls;
    vector<mlir::Plugin::LoopOp> loops;
    mlir::Plugin::LoopOp loop;
    vector<std::pair<mlir::Block*, mlir::Block*> > edges;
    std::pair<mlir::Block*, mlir::Block*> edge;
    vector<mlir::Operation *> opData;
    bool boolResult;
    uint64_t idResult;
    vector<uint64_t> idsResult;
    mlir::Value valueResult;
};
} // namespace PinCom

#endif
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
    This file contains the declaration of the PluginAPI_Server class.
*/

#ifndef PLUGIN_FRAMEWORK_SERVER_API_H
#define PLUGIN_FRAMEWORK_SERVER_API_H

#include "BasicPluginOpsAPI.h"
#include "PluginServer/PluginServer.h"
#include "Dialect/PluginTypes.h"

namespace PluginAPI {

using std::vector;
using std::string;
using std::pair;
using namespace mlir::Plugin;
class PluginServerAPI : public BasicPluginOpsAPI {
public:
    PluginServerAPI () = default;
    ~PluginServerAPI () = default;

    uint64_t CreateBlock(mlir::Block*, uint64_t, uint64_t) override;

    vector<FunctionOp> GetAllFunc() override;
    vector<LocalDeclOp> GetDecls(uint64_t) override;
    PhiOp GetPhiOp(uint64_t) override;
    CallOp GetCallOp(uint64_t) override;
    PluginIR::PluginTypeID GetTypeCodeFromString(string type);
    LoopOp AllocateNewLoop(uint64_t funcID);
    LoopOp GetLoopById(uint64_t loopID);
    vector<LoopOp> GetLoopsFromFunc(uint64_t funcID);
    bool IsBlockInLoop(uint64_t loopID, uint64_t blockID);
    void DeleteLoop(uint64_t loopID);
    void AddLoop(uint64_t loopID, uint64_t outerID, uint64_t funcID);
    pair<uint64_t, uint64_t> LoopSingleExit(uint64_t loopID);
    vector<pair<uint64_t, uint64_t> > GetLoopExitEdges(uint64_t loopID);
    uint64_t GetHeader(uint64_t loopID);
    uint64_t GetLatch(uint64_t loopID);
    vector<uint64_t> GetLoopBody(uint64_t loopID);
    LoopOp GetBlockLoopFather(uint64_t blockID);
    /* Plugin API for CallOp. */
    bool SetLhsInCallOp(uint64_t, uint64_t);
    /* Plugin API for CondOp. */
    uint64_t CreateCondOp(IComparisonCode, uint64_t, uint64_t) override;
    mlir::Value GetResultFromPhi(uint64_t) override;

private:
    vector<FunctionOp> GetFunctionOpResult(const string& funName, const string& params);
    vector<LocalDeclOp> GetDeclOperationResult(const string& funName, const string& params);
    LoopOp GetLoopResult(const string&funName, const string& params);
    vector<LoopOp> GetLoopsResult(const string& funName, const string& params);
    bool GetBoolResult(const string& funName, const string& params);
    pair<uint64_t, uint64_t> EdgeResult(const string& funName, const string& params);
    vector<pair<uint64_t, uint64_t> > EdgesResult(const string& funName, const string& params);
    uint64_t BlockResult(const string& funName, const string& params);
    vector<uint64_t> BlocksResult(const string& funName, const string& params);
    void WaitClientResult(const string& funName, const string& params);
}; // class PluginServerAPI
} // namespace PluginAPI

#endif // PLUGIN_FRAMEWORK_SERVER_API_H

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

    vector<FunctionOp> GetAllFunc() override;
    FunctionOp GetFunctionOpById(uint64_t);
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
    void AddBlockToLoop(mlir::Block*, LoopOp *);
    pair<mlir::Block*, mlir::Block*> LoopSingleExit(uint64_t loopID);
    vector<pair<mlir::Block*, mlir::Block*> > GetLoopExitEdges(uint64_t loopID);
    mlir::Block* GetHeader(uint64_t loopID);
    mlir::Block* GetLatch(uint64_t loopID);
    void SetHeader(LoopOp*, mlir::Block*);
    void SetLatch(LoopOp*, mlir::Block*);
    vector<mlir::Block*> GetLoopBody(uint64_t loopID);
    LoopOp GetBlockLoopFather(mlir::Block*);
    mlir::Block* FindBlock(uint64_t);
    uint64_t FindBasicBlock(mlir::Block*);
    bool InsertValue(uint64_t, mlir::Value);
    /* Plugin API for CallOp. */
    bool SetLhsInCallOp(uint64_t, uint64_t) override;
    uint64_t CreateCallOp(uint64_t, uint64_t, vector<uint64_t> &) override;
    /* Plugin API for CondOp. */
    uint64_t CreateCondOp(uint64_t, IComparisonCode, uint64_t, uint64_t, uint64_t, uint64_t) override;
    mlir::Value GetResultFromPhi(uint64_t) override;
    bool IsDomInfoAvailable() override;
    /* Plugin API for AssignOp. */
    uint64_t CreateAssignOp(uint64_t, IExprCode, vector<uint64_t> &) override;
    /* Plugin API for PhiOp. */
    uint32_t AddArgInPhiOp(uint64_t, uint64_t, uint64_t, uint64_t) override;
    PhiOp CreatePhiOp(uint64_t, uint64_t) override;
    /* Plugin API for ConstOp. */
    mlir::Value CreateConstOp(mlir::Attribute, mlir::Type) override;
    void DebugValue(uint64_t) override;
    bool IsLtoOptimize() override;
    bool IsWholeProgram() override;
    
    mlir::Value GetCurrentDefFromSSA(uint64_t) override;
    bool SetCurrentDefInSSA(uint64_t, uint64_t) override;
    mlir::Value CopySSAOp(uint64_t) override;
    mlir::Value CreateSSAOp(mlir::Type) override;
    mlir::Value ConfirmValue(mlir::Value);
    mlir::Value BuildMemRef(PluginIR::PluginTypeBase, mlir::Value, mlir::Value);
    bool RedirectFallthroughTarget(FallThroughOp&, mlir::Block*, mlir::Block*) override;
    mlir::Operation* GetSSADefOperation(uint64_t) override;
    void InsertCreatedBlock(uint64_t id, mlir::Block* block);

    int64_t GetInjectDataAddress() override; // 获取注入点返回数据的地址
    string GetDeclSourceFile(int64_t) override;
    string VariableName(int64_t) override;
    string FuncName(int64_t) override;
    string GetIncludeFile() override;
    int GetDeclSourceLine(int64_t) override;
    int GetDeclSourceColumn(int64_t) override;
private:
    mlir::Block* BlockResult(const string& funName, const string& params);
    vector<mlir::Block*> BlocksResult(const string& funName, const string& params);
    bool GetDomInfoAvaiResult(const string& funName);
}; // class PluginServerAPI
} // namespace PluginAPI

#endif // PLUGIN_FRAMEWORK_SERVER_API_H

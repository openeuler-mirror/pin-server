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

*/

#ifndef BASIC_PLUGIN_OPS_FRAMEWORK_API_H
#define BASIC_PLUGIN_OPS_FRAMEWORK_API_H

#include "Dialect/PluginOps.h"
#include "Dialect/PluginTypes.h"

#include <vector>
#include <string>

namespace PluginAPI {
using std::vector;
using std::string;
using std::pair;
using namespace mlir::Plugin;

/* The BasicPluginOpsAPI class defines the basic plugin API, both the plugin
   client and the server should inherit this class and implement there own
   defined API. */
class BasicPluginOpsAPI {
public:
    BasicPluginOpsAPI() = default;
    virtual ~BasicPluginOpsAPI() = default;

    virtual int64_t GetInjectDataAddress() = 0;
    virtual string GetDeclSourceFile(int64_t) = 0;
    virtual string VariableName(int64_t) = 0;
    virtual string FuncName(int64_t) = 0;
    virtual string GetIncludeFile() = 0;
    virtual int GetDeclSourceLine(int64_t) = 0;
    virtual int GetDeclSourceColumn(int64_t) = 0;

    // CGnodeOp
    virtual vector<CGnodeOp> GetAllCGnode() = 0;
    virtual CGnodeOp GetCGnodeOpById(uint64_t) = 0;

    virtual vector<FunctionOp> GetAllFunc() = 0;
    virtual FunctionOp GetFunctionOpById(uint64_t) = 0;
    virtual vector<mlir::Plugin::DeclBaseOp> GetFuncDecls(uint64_t) = 0;
    virtual llvm::SmallVector<mlir::Plugin::FieldDeclOp> GetFields(uint64_t) = 0;
    virtual mlir::Plugin::DeclBaseOp BuildDecl(IDefineCode, llvm::StringRef, PluginIR::PluginTypeBase) = 0;

    virtual mlir::Plugin::FieldDeclOp MakeNode(IDefineCode) = 0;
    virtual void SetDeclName(uint64_t newfieldId, uint64_t fieldId) = 0;
    virtual void SetDeclType(uint64_t newfieldId, uint64_t fieldId) = 0;
    virtual void SetDeclAlign(uint64_t newfieldId, uint64_t fieldId) = 0;
    virtual void SetUserAlign(uint64_t newfieldId, uint64_t fieldId) = 0;
    virtual void SetSourceLocation(uint64_t newfieldId, uint64_t fieldId) = 0;
    virtual void SetAddressable(uint64_t newfieldId, uint64_t fieldId) = 0;
    virtual void SetNonAddressablep(uint64_t newfieldId, uint64_t fieldId) = 0;
    virtual void SetVolatile(uint64_t newfieldId, uint64_t fieldId) = 0;
    virtual void SetDeclContext(uint64_t newfieldId, uint64_t declId) = 0;
    virtual void SetDeclChain(uint64_t newfieldId, uint64_t fieldId) = 0;

    virtual unsigned GetDeclTypeSize(uint64_t declId) = 0;

    virtual void SetTypeFields(uint64_t declId, uint64_t fieldId) = 0;
    virtual void LayoutType(uint64_t declId) = 0;
    virtual void LayoutDecl(uint64_t declId) = 0;
    virtual PluginIR::PluginTypeBase GetDeclType(uint64_t declId) = 0;

    virtual vector<LocalDeclOp> GetDecls(uint64_t) = 0;
    virtual LoopOp AllocateNewLoop(uint64_t) = 0;
    virtual vector<LoopOp> GetLoopsFromFunc(uint64_t) = 0;
    virtual LoopOp GetLoopById(uint64_t) = 0;
    virtual void AddLoop(uint64_t, uint64_t, uint64_t) = 0;
    virtual void AddBlockToLoop(mlir::Block*, LoopOp *) = 0;
    virtual void DeleteLoop(uint64_t) = 0;
    virtual vector<mlir::Block*> GetLoopBody(uint64_t) = 0;
    virtual bool IsBlockInLoop(uint64_t, uint64_t) = 0;
    virtual pair<mlir::Block*, mlir::Block*> LoopSingleExit(uint64_t) = 0;
    virtual vector<pair<mlir::Block*, mlir::Block*> > GetLoopExitEdges(uint64_t) = 0;
    virtual LoopOp GetBlockLoopFather(mlir::Block*) = 0;
    virtual PhiOp GetPhiOp(uint64_t) = 0;
    virtual CallOp GetCallOp(uint64_t) = 0;
    virtual bool SetLhsInCallOp(uint64_t, uint64_t) = 0;
    virtual uint64_t CreateCallOp(uint64_t, uint64_t, vector<uint64_t> &) = 0;
    virtual uint64_t CreateCondOp(uint64_t, IComparisonCode, uint64_t, uint64_t, uint64_t, uint64_t) = 0;
    virtual mlir::Value GetResultFromPhi(uint64_t) = 0;
    virtual bool IsDomInfoAvailable() = 0;
    virtual uint64_t CreateAssignOp(uint64_t, IExprCode, vector<uint64_t> &) = 0;
    virtual mlir::Value CreateConstOp(mlir::Attribute, mlir::Type) = 0;
    virtual uint32_t AddArgInPhiOp(uint64_t, uint64_t, uint64_t, uint64_t) = 0;
    virtual PhiOp CreatePhiOp(uint64_t, uint64_t) = 0;
    virtual void DebugValue(uint64_t) = 0;
    virtual bool IsLtoOptimize() = 0;
    virtual bool IsWholeProgram() = 0;

    virtual mlir::Value GetCurrentDefFromSSA(uint64_t) = 0;
    virtual bool SetCurrentDefInSSA(uint64_t, uint64_t) = 0;
    virtual mlir::Value CopySSAOp(uint64_t) = 0;
    virtual mlir::Value CreateSSAOp(mlir::Type) = 0;
    virtual mlir::Value ConfirmValue(mlir::Value) = 0;
    virtual mlir::Value BuildMemRef(PluginIR::PluginTypeBase, mlir::Value, mlir::Value) = 0;
    virtual bool RedirectFallthroughTarget(FallThroughOp&, mlir::Block*, mlir::Block*) = 0;
    virtual mlir::Operation* GetSSADefOperation(uint64_t) = 0;
}; // class BasicPluginOpsAPI
} // namespace PluginAPI

#endif // BASIC_PLUGIN_OPS_FRAMEWORK_API_H

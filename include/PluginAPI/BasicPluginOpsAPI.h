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

    virtual uint64_t CreateBlock(mlir::Block*, uint64_t, uint64_t) = 0;

    virtual vector<FunctionOp> GetAllFunc() = 0;
    virtual vector<LocalDeclOp> GetDecls(uint64_t) = 0;
    virtual LoopOp AllocateNewLoop(uint64_t) = 0;
    virtual vector<LoopOp> GetLoopsFromFunc(uint64_t) = 0;
    virtual LoopOp GetLoopById(uint64_t) = 0;
    virtual void AddLoop(uint64_t, uint64_t, uint64_t) = 0;
    virtual void DeleteLoop(uint64_t) = 0;
    virtual vector<uint64_t> GetLoopBody(uint64_t) = 0;
    virtual bool IsBlockInLoop(uint64_t, uint64_t) = 0;
    virtual pair<uint64_t, uint64_t> LoopSingleExit(uint64_t) = 0;
    virtual vector<pair<uint64_t, uint64_t> > GetLoopExitEdges(uint64_t) = 0;
    virtual LoopOp GetBlockLoopFather(uint64_t) = 0;
    virtual PhiOp GetPhiOp(uint64_t) = 0;
    virtual CallOp GetCallOp(uint64_t) = 0;
    virtual bool SetLhsInCallOp(uint64_t, uint64_t) = 0;
    virtual uint64_t CreateCondOp(IComparisonCode, uint64_t, uint64_t) = 0;
    virtual mlir::Value GetResultFromPhi(uint64_t) = 0;
}; // class BasicPluginOpsAPI
} // namespace PluginAPI

#endif // BASIC_PLUGIN_OPS_FRAMEWORK_API_H
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

   Author: Chenhui Zheng
   Create: 2023-11-01
   Description:
    This file contains the declaration of the PluginAPI_Server class.
*/
#ifndef PLUGIN_FRAMEWORK_DATA_FLOW_API_H
#define PLUGIN_FRAMEWORK_DATA_FLOW_API_H

#include "BasicPluginOpsAPI.h"
#include "PluginServer/PluginServer.h"
#include "Dialect/PluginTypes.h"
#include "PluginServerAPI.h"

namespace PluginAPI {

using std::string;
using std::vector;

using namespace PinServer;
using namespace mlir::Plugin;
class DataFlowAPI {
public:
    DataFlowAPI() = default;
    ~DataFlowAPI() = default;

    // 计算支配信息
    void CalDominanceInfo(uint64_t, uint64_t);
    
    // USE-DEF
    vector<mlir::Operation*> GetImmUseStmts(mlir::Value);
    mlir::Value GetGimpleVuse(uint64_t);
    mlir::Value GetGimpleVdef(uint64_t);
    vector<mlir::Value> GetSsaUseOperand(uint64_t);
    vector<mlir::Value> GetSsaDefOperand(uint64_t);
    vector<mlir::Value> GetPhiOrStmtUse(uint64_t);
    vector<mlir::Value> GetPhiOrStmtDef(uint64_t);

    //别名分析
    bool RefsMayAlias(mlir::Value, mlir::Value, uint64_t);

    // 指针分析
    bool PTIncludesDecl(mlir::Value, uint64_t);
    bool PTsIntersect(mlir::Value, mlir::Value);

private:
    PluginServerAPI pluginAPI;
};

}  // namespace PluginAPI

#endif  // PLUGIN_FRAMEWORK_CONTROL_FLOW_API_H

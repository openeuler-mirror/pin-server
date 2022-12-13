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

#ifndef PLUGIN_FRAMEWORK_CONTROL_FLOW_API_H
#define PLUGIN_FRAMEWORK_CONTROL_FLOW_API_H

#include "BasicPluginOpsAPI.h"
#include "PluginServer/PluginServer.h"
#include "Dialect/PluginTypes.h"

namespace PluginAPI {

using std::string;
using std::vector;

using namespace PinServer;
using namespace mlir::Plugin;
class ControlFlowAPI {
public:
    ControlFlowAPI() = default;
    ~ControlFlowAPI() = default;

    bool UpdateSSA(void);
    vector<PhiOp> GetAllPhiOpInsideBlock(mlir::Block *b);
    void SetImmediateDominatorInBlock(mlir::Block *b, mlir::Block *dominated);

private:
    bool GetUpdateOperationResult(const string &funName);
    vector<PhiOp> GetPhiOperationResult(const string &funName, const string& params);
    void GetDominatorSetOperationResult(const string &funName, const string& params);
    void WaitClientResult(const string &funName, const string &params = {});
};

}  // namespace PluginAPI

#endif  // PLUGIN_FRAMEWORK_CONTROL_FLOW_API_H

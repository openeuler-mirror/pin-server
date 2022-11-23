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
using namespace mlir::Plugin;
class PluginServerAPI : public BasicPluginOpsAPI {
public:
    PluginServerAPI () = default;
    ~PluginServerAPI () = default;

    vector<FunctionOp> GetAllFunc() override;
    PluginIR::PluginTypeID GetTypeCodeFromString(string type);
private:
    vector<FunctionOp> GetOperationResult(const string& funName, const string& params);
    void WaitClientResult(const string& funName, const string& params);
}; // class PluginServerAPI
} // namespace PluginAPI

#endif // PLUGIN_FRAMEWORK_SERVER_API_H

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
using namespace mlir::Plugin;

/* The BasicPluginOpsAPI class defines the basic plugin API, both the plugin
   client and the server should inherit this class and implement there own
   defined API. */
class BasicPluginOpsAPI {
public:
    BasicPluginOpsAPI() = default;
    virtual ~BasicPluginOpsAPI() = default;

    virtual vector<FunctionOp> GetAllFunc() = 0;
}; // class BasicPluginOpsAPI
} // namespace PluginAPI

#endif // BASIC_PLUGIN_OPS_FRAMEWORK_API_H
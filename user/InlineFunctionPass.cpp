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
    This file contains the implementation of the inlineFunctionPass class.
*/

#include "PluginAPI/PluginServerAPI.h"
#include "user/InlineFunctionPass.h"

namespace PluginOpt {
using namespace PluginAPI;

static void UserOptimizeFunc(void)
{
    PluginServerAPI pluginAPI;
    vector<FunctionOp> allFunction = pluginAPI.GetAllFunc();
    int count = 0;
    for (size_t i = 0; i < allFunction.size(); i++) {
        if (allFunction[i] && allFunction[i].declaredInlineAttr().getValue())
            count++;
    }
    fprintf(stderr, "declaredInline have %d functions were declared.\n", count);
}

int InlineFunctionPass::DoOptimize()
{
    UserOptimizeFunc();
    return 0;
}
}

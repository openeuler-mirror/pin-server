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
    This file contains the declaration of the ArrayWidenPass class.
*/

#ifndef SIMPLE_LICM_PASS_H
#define SIMPLE_LICM_PASS_H

#include "PluginServer/PluginOptBase.h"

namespace PluginOpt {
class SimpleLICMPass : public PluginOptBase {
public:
    SimpleLICMPass() : PluginOptBase(HANDLE_MANAGER_SETUP)
    {
    }
    bool Gate()
    {
        return true;
    }
    int DoOptimize()
    {
        uint64_t fun = (uint64_t)GetFuncAddr();
        return DoOptimize(fun);
    }
    int DoOptimize(uint64_t fun);
};
}

#endif
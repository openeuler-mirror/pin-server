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
    This file contains the implementation of the User Init.
*/

#include "PluginAPI/PluginServerAPI.h"
#include "user/ArrayWidenPass.h"
#include "user/InlineFunctionPass.h"
#include "user/LocalVarSummeryPass.h"
#include "user/StructReorder.h"
#include "user/SimpleLICMPass.h"

void RegisterCallbacks(void)
{
    PinServer::PluginServer *pluginServer = PinServer::PluginServer::GetInstance(); 
    // pluginServer->RegisterOpt(std::make_shared<PluginOpt::InlineFunctionPass>(PluginOpt::HANDLE_BEFORE_IPA));
    // pluginServer->RegisterOpt(std::make_shared<PluginOpt::LocalVarSummeryPass>(PluginOpt::HANDLE_BEFORE_IPA));
    PluginOpt::ManagerSetup setupData(PluginOpt::PASS_LAD, 1, PluginOpt::PASS_INSERT_AFTER);
    pluginServer->RegisterPassManagerOpt(setupData, std::make_shared<PluginOpt::SimpleLICMPass>());
    // PluginOpt::ManagerSetup setupData(PluginOpt::PASS_PHIOPT, 1, PluginOpt::PASS_INSERT_AFTER);
    // pluginServer->RegisterPassManagerOpt(setupData, std::make_shared<PluginOpt::ArrayWidenPass>());
    // PluginOpt::ManagerSetup setupData(PluginOpt::PASS_MAC, 1, PluginOpt::PASS_INSERT_AFTER);
    // pluginServer->RegisterPassManagerOpt(setupData, std::make_shared<PluginOpt::StructReorderPass>());
}

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
   Description: This file contains the implementation of the PluginAPI_Server class
*/

#include "PluginAPI/ControlFlowAPI.h"
#include "PluginServer/PluginLog.h"

namespace PluginAPI {
using namespace PinServer;
using namespace mlir::Plugin;


bool ControlFlowAPI::UpdateSSA(void)
{
    Json::Value root;
    string funName = __func__;
    string params = root.toStyledString();

    return GetUpdateOperationResult(funName);
}

bool ControlFlowAPI::GetUpdateOperationResult(const string &funName)
{
    WaitClientResult(funName);
    return PluginServer::GetInstance()->GetBoolResult();
}

vector<PhiOp> ControlFlowAPI::GetPhiOperationResult(const string &funName, const string& params)
{
    WaitClientResult(funName, params);
    vector<PhiOp> retOps = PluginServer::GetInstance()->GetPhiOpsResult();
    return retOps;
}

void ControlFlowAPI::GetDominatorSetOperationResult(const string &funName, const string& params)
{
    WaitClientResult(funName, params);
    return;
}

void ControlFlowAPI::WaitClientResult(const string &funName, const string &params)
{
    PluginServer *server = PluginServer::GetInstance();
    server->SetApiFuncName(funName);
    server->SetUserFunState(STATE_BEGIN);
    server->SemPost();
    while (1) {
        server->ClientReturnSemWait();
        if (server->GetUserFunState() == STATE_RETURN) {  // wait client result
            server->SetUserFunState(STATE_WAIT_BEGIN);
            break;
        }
    }
}

vector<PhiOp> ControlFlowAPI::GetAllPhiOpInsideBlock(mlir::Block *b)
{
    PluginServer *server = PluginServer::GetInstance();
    Json::Value root;
    string funName = __func__;
    root["bbAddr"] = std::to_string(server->FindBasicBlock(b));
    string params = root.toStyledString();

    return GetPhiOperationResult(funName, params);
}

uint64_t ControlFlowAPI::CreateBlock(mlir::Block* b, uint64_t funcAddr, uint64_t bbAddr)
{
    Json::Value root;
    string funName = __func__;
    assert(funcAddr);
    assert(bbAddr);
    root["funcaddr"] = std::to_string(funcAddr);
    root["bbaddr"] = std::to_string(bbAddr);
    string params = root.toStyledString();
    WaitClientResult(funName, params);
    return PluginServer::GetInstance()->GetBlockResult(b);
}

void ControlFlowAPI::DeleteBlock(mlir::Block* b, uint64_t funcAddr,
                                  uint64_t bbAddr)
{
    Json::Value root;
    string funName = __func__;
    assert(funcAddr);
    assert(bbAddr);
    root["funcaddr"] = std::to_string(funcAddr);
    root["bbaddr"] = std::to_string(bbAddr);
    string params = root.toStyledString();
    WaitClientResult(funName, params);
    PluginServer::GetInstance()->EraseBlock(b);
    b->erase();
}

/* dir: 1 or 2 */
void ControlFlowAPI::SetImmediateDominator(uint64_t dir, uint64_t bbAddr,
                                            uint64_t domiAddr)
{
    Json::Value root;
    string funName = __func__;
    assert(dir && bbAddr && domiAddr);
    root["dir"] = std::to_string(dir);
    root["bbaddr"] = std::to_string(bbAddr);
    root["domiAddr"] = std::to_string(domiAddr);
    string params = root.toStyledString();
    WaitClientResult(funName, params);
}

/* dir: 1 or 2 */
uint64_t ControlFlowAPI::GetImmediateDominator(uint64_t dir, uint64_t bbAddr)
{
    Json::Value root;
    string funName = __func__;
    assert(dir && bbAddr);
    root["dir"] = std::to_string(dir);
    root["bbaddr"] = std::to_string(bbAddr);
    string params = root.toStyledString();
    WaitClientResult(funName, params);
    return PluginServer::GetInstance()->GetIdResult();
}

/* dir: 1 or 2 */
uint64_t ControlFlowAPI::RecomputeDominator(uint64_t dir, uint64_t bbAddr)
{
    Json::Value root;
    string funName = __func__;
    assert(dir && bbAddr);
    root["dir"] = std::to_string(dir);
    root["bbaddr"] = std::to_string(bbAddr);
    string params = root.toStyledString();
    WaitClientResult(funName, params);
    return PluginServer::GetInstance()->GetIdResult();
}


} // namespace Plugin_IR

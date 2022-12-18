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

#include "PluginAPI/ControlFlowAPI.h"
#include "PluginServer/PluginLog.h"

namespace PluginAPI {
using namespace PinServer;
using namespace mlir::Plugin;

static uint64_t GetValueId(mlir::Value v)
{
    mlir::Operation *op = v.getDefiningOp();
    if (auto mOp = llvm::dyn_cast<MemOp>(op)) {
        return mOp.id();
    } else if (auto ssaOp = llvm::dyn_cast<SSAOp>(op)) {
        return ssaOp.id();
    } else if (auto cstOp = llvm::dyn_cast<ConstOp>(op)) {
        return cstOp.id();
    } else if (auto phOp = llvm::dyn_cast<PlaceholderOp>(op)) {
        return phOp.id();
    }
    return 0;
}

static uint64_t getBlockAddress(mlir::Block* b)
{
    if (mlir::Plugin::CondOp oops = llvm::dyn_cast<mlir::Plugin::CondOp>(b->back())) {
        return oops.addressAttr().getInt();
    } else if (mlir::Plugin::FallThroughOp oops = llvm::dyn_cast<mlir::Plugin::FallThroughOp>(b->back())) {
        return oops.addressAttr().getInt();
    } else if (mlir::Plugin::RetOp oops = llvm::dyn_cast<mlir::Plugin::RetOp>(b->back())) {
        return oops.addressAttr().getInt();
    }
    abort();
}

bool ControlFlowAPI::UpdateSSA(void)
{
    Json::Value root;
    string funName = __func__;
    string params = root.toStyledString();

    return GetUpdateOperationResult(funName);
}

bool ControlFlowAPI::GetUpdateOperationResult(const string &funName)
{
    Json::Value root;
    pluginAPI.WaitClientResult(funName, root.toStyledString());
    return PluginServer::GetInstance()->GetBoolResult();
}

vector<PhiOp> ControlFlowAPI::GetPhiOperationResult(const string &funName, const string& params)
{
    pluginAPI.WaitClientResult(funName, params);;
    vector<PhiOp> retOps = PluginServer::GetInstance()->GetPhiOpsResult();
    return retOps;
}

void ControlFlowAPI::GetDominatorSetOperationResult(const string &funName, const string& params)
{
    pluginAPI.WaitClientResult(funName, params);;
    return;
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
    pluginAPI.WaitClientResult(funName, params);;
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
    pluginAPI.WaitClientResult(funName, params);;
    PluginServer::GetInstance()->EraseBlock(b);
    // b->erase();
}

/* dir: 1 or 2 */
void ControlFlowAPI::SetImmediateDominator(uint64_t dir, uint64_t bbAddr,
                                            uint64_t domiAddr)
{
    Json::Value root;
    string funName = __func__;
    if (!bbAddr || !domiAddr) return;
    assert(dir && bbAddr && domiAddr);
    root["dir"] = std::to_string(dir);
    root["bbaddr"] = std::to_string(bbAddr);
    root["domiaddr"] = std::to_string(domiAddr);
    string params = root.toStyledString();
    pluginAPI.WaitClientResult(funName, params);;
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
    pluginAPI.WaitClientResult(funName, params);;
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
    pluginAPI.WaitClientResult(funName, params);;
    return PluginServer::GetInstance()->GetIdResult();
}

mlir::Value ControlFlowAPI::CreateNewDef(mlir::Value oldValue,
                                         mlir::Operation *op)
{
    Json::Value root;
    string funName = __func__;
    // FIXME: use baseOp.
    uint64_t opId = llvm::dyn_cast<PhiOp>(op).idAttr().getInt();
    root["opId"] = std::to_string(opId);
    uint64_t valueId = GetValueId(oldValue);
    root["valueId"] = std::to_string(valueId);
    uint64_t defId = 0;
    root["defId"] = std::to_string(defId);
    string params = root.toStyledString();
    pluginAPI.WaitClientResult(funName, params);
    return PluginServer::GetInstance()->GetValueResult();
}

void ControlFlowAPI::CreateFallthroughOp(
    uint64_t address, uint64_t destaddr)
{
    Json::Value root;
    string funName = __func__;
    root["address"] = std::to_string(address);
    root["destaddr"] = std::to_string(destaddr);
    string params = root.toStyledString();
    pluginAPI.WaitClientResult(funName, params);
}

void ControlFlowAPI::RemoveEdge(uint64_t src, uint64_t dest)
{
    Json::Value root;
    string funName = __func__;
    root["src"] = std::to_string(src);
    root["dest"] = std::to_string(dest);
    string params = root.toStyledString();
    pluginAPI.WaitClientResult(funName, params);
}

} // namespace PluginAPI

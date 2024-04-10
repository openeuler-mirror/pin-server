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

namespace PluginAPI {
using namespace PinServer;
using namespace mlir::Plugin;

static uint64_t GetValueId(mlir::Value v)
{
    mlir::Operation *op = v.getDefiningOp();
    if (auto mOp = llvm::dyn_cast<MemOp>(op)) {
        return mOp.getId();
    } else if (auto ssaOp = llvm::dyn_cast<SSAOp>(op)) {
        return ssaOp.getId();
    } else if (auto cstOp = llvm::dyn_cast<ConstOp>(op)) {
        return cstOp.getId();
    } else if (auto treelistop = llvm::dyn_cast<ListOp>(op)) {
        return treelistop.getId();
    } else if (auto strop = llvm::dyn_cast<StrOp>(op)) {
        return strop.getId();
    } else if (auto arrayop = llvm::dyn_cast<ArrayOp>(op)) {
        return arrayop.getId();
    } else if (auto declop = llvm::dyn_cast<DeclBaseOp>(op)) {
        return declop.getId();
    } else if (auto fieldop = llvm::dyn_cast<FieldDeclOp>(op)) {
        return fieldop.getId();
    } else if (auto addressop = llvm::dyn_cast<AddressOp>(op)) {
        return addressop.getId();
    } else if (auto constructorop = llvm::dyn_cast<ConstructorOp>(op)) {
        return constructorop.getId();
    } else if (auto vecop = llvm::dyn_cast<VecOp>(op)) {
        return vecop.getId();
    } else if (auto blockop = llvm::dyn_cast<BlockOp>(op)) {
        return blockop.getId();
    } else if (auto compop = llvm::dyn_cast<ComponentOp>(op)) {
        return compop.getId();
    } else if (auto phOp = llvm::dyn_cast<PlaceholderOp>(op)) {
        return phOp.getId();
    }
    return 0;
}

static uint64_t getBlockAddress(mlir::Block* b)
{
    if (mlir::Plugin::CondOp oops = llvm::dyn_cast<mlir::Plugin::CondOp>(b->back())) {
        return oops.getAddressAttr().getInt();
    } else if (mlir::Plugin::FallThroughOp oops = llvm::dyn_cast<mlir::Plugin::FallThroughOp>(b->back())) {
        return oops.getAddressAttr().getInt();
    } else if (mlir::Plugin::RetOp oops = llvm::dyn_cast<mlir::Plugin::RetOp>(b->back())) {
        return oops.getAddressAttr().getInt();
    } else if (mlir::Plugin::GotoOp oops = llvm::dyn_cast<mlir::Plugin::GotoOp>(b->back())) {
        return oops.getAddressAttr().getInt();
    } else if (mlir::Plugin::TransactionOp oops = llvm::dyn_cast<mlir::Plugin::TransactionOp>(b->back())) {
        return oops.getAddressAttr().getInt();
    } else {
        abort();
    }
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
    return PluginServer::GetInstance()->GetBoolResult(funName, root.toStyledString());
}

vector<PhiOp> ControlFlowAPI::GetPhiOperationResult(const string &funName, const string& params)
{
    vector<PhiOp> retOps = PluginServer::GetInstance()->GetPhiOpsResult(funName, params);
    return retOps;
}

void ControlFlowAPI::GetDominatorSetOperationResult(const string &funName, const string& params)
{
    PluginServer::GetInstance()->RemoteCallClientWithAPI(funName, params);
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

vector<mlir::Operation*> ControlFlowAPI::GetAllOpsInsideBlock(mlir::Block *b)
{
    PluginServer *server = PluginServer::GetInstance();
    Json::Value root;
    string funName = __func__;
    root["bbAddr"] = std::to_string(server->FindBasicBlock(b));
    string params = root.toStyledString();
    vector<uint64_t> ids = server->GetIdsResult(funName, params);
    vector<mlir::Operation*> ops;
    for (auto id: ids) {
        mlir::Operation* op = server->FindOperation(id);
        ops.push_back(op);
    }
    return ops;
}

mlir::Block* ControlFlowAPI::CreateBlock(mlir::Block* b, FunctionOp *funcOp)
{
    Json::Value root;
    string funName = __func__;
    uint64_t funcAddr = funcOp->getIdAttr().getInt();
    assert(funcAddr);
    PluginServer *server = PluginServer::GetInstance();
    uint64_t bbAddr = server->FindBasicBlock(b);
    assert(bbAddr);
    root["funcaddr"] = std::to_string(funcAddr);
    root["bbaddr"] = std::to_string(bbAddr);
    string params = root.toStyledString();
    server->RemoteCallClientWithAPI(funName, params);
    uint64_t retId = server->GetBlockResult(b);
    return server->FindBlock(retId);
}

void ControlFlowAPI::DeleteBlock(mlir::Block* b, FunctionOp* funcOp)
{
    Json::Value root;
    string funName = __func__;
    uint64_t funcAddr = funcOp->getIdAttr().getInt();
    assert(funcAddr);
    uint64_t bbAddr = PluginServer::GetInstance()->FindBasicBlock(b);
    assert(bbAddr);
    root["funcaddr"] = std::to_string(funcAddr);
    root["bbaddr"] = std::to_string(bbAddr);
    string params = root.toStyledString();
    PluginServer::GetInstance()->RemoteCallClientWithAPI(funName, params);
    PluginServer::GetInstance()->EraseBlock(b);
}

/* dir: 1 or 2 */
void ControlFlowAPI::SetImmediateDominator(uint64_t dir, mlir::Block* bb, uint64_t domiAddr)
{
    Json::Value root;
    string funName = __func__;
    PluginServer *server = PluginServer::GetInstance();
    uint64_t bbAddr = server->FindBasicBlock(bb);
    if (!bbAddr || !domiAddr) return;
    assert(dir && bbAddr && domiAddr);
    root["dir"] = std::to_string(dir);
    root["bbaddr"] = std::to_string(bbAddr);
    root["domiaddr"] = std::to_string(domiAddr);
    string params = root.toStyledString();
    PluginServer::GetInstance()->RemoteCallClientWithAPI(funName, params);
}

void ControlFlowAPI::SetImmediateDominator(uint64_t dir, mlir::Block* bb,
                                           mlir::Block* domi)
{
    Json::Value root;
    string funName = __func__;
    PluginServer *server = PluginServer::GetInstance();
    uint64_t bbAddr = server->FindBasicBlock(bb);
    uint64_t domiAddr = server->FindBasicBlock(domi);
    if (!bbAddr || !domiAddr) return;
    assert(dir && bbAddr && domiAddr);
    root["dir"] = std::to_string(dir);
    root["bbaddr"] = std::to_string(bbAddr);
    root["domiaddr"] = std::to_string(domiAddr);
    string params = root.toStyledString();
    PluginServer::GetInstance()->RemoteCallClientWithAPI(funName, params);
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
    return PluginServer::GetInstance()->GetIdResult(funName, params);
}

/* dir: 1 or 2 */
uint64_t ControlFlowAPI::RecomputeDominator(uint64_t dir, mlir::Block* bb)
{
    Json::Value root;
    string funName = __func__;
    uint64_t bbAddr = PluginServer::GetInstance()->FindBasicBlock(bb);
    assert(dir && bbAddr);
    root["dir"] = std::to_string(dir);
    root["bbaddr"] = std::to_string(bbAddr);
    string params = root.toStyledString();
    return PluginServer::GetInstance()->GetIdResult(funName, params);
}

mlir::Value ControlFlowAPI::CreateNewDef(mlir::Value oldValue,
                                         mlir::Operation *op)
{
    Json::Value root;
    string funName = __func__;
    // FIXME: use baseOp.
    uint64_t opId = llvm::dyn_cast<PhiOp>(op).getIdAttr().getInt();
    root["opId"] = std::to_string(opId);
    uint64_t valueId = GetValueId(oldValue);
    root["valueId"] = std::to_string(valueId);
    uint64_t defId = 0;
    root["defId"] = std::to_string(defId);
    string params = root.toStyledString();
    return PluginServer::GetInstance()->GetValueResult(funName, params);
}

void ControlFlowAPI::CreateFallthroughOp(
    uint64_t address, uint64_t destaddr)
{
    Json::Value root;
    string funName = __func__;
    root["address"] = std::to_string(address);
    root["destaddr"] = std::to_string(destaddr);
    string params = root.toStyledString();
    PluginServer::GetInstance()->RemoteCallClientWithAPI(funName, params);
}

void ControlFlowAPI::RemoveEdge(mlir::Block* srcBB, mlir::Block* destBB)
{
    Json::Value root;
    string funName = __func__;
    uint64_t src = PluginServer::GetInstance()->FindBasicBlock(srcBB);
    uint64_t dest = PluginServer::GetInstance()->FindBasicBlock(destBB);
    root["src"] = std::to_string(src);
    root["dest"] = std::to_string(dest);
    string params = root.toStyledString();
    PluginServer::GetInstance()->RemoteCallClientWithAPI(funName, params);
}

} // namespace PluginAPI

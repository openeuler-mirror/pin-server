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
#include "PluginAPI/DataFlowAPI.h"

namespace PluginAPI {
using namespace PinServer;
using namespace mlir::Plugin;

static uint64_t GetValueId(mlir::Value v)
{
    mlir::Operation*op = v.getDefiningOp();
    if (auto mOp = llvm::dyn_cast<MemOp>(op)) {
        return mOp.id();
    } else if (auto ssaOp = llvm::dyn_cast<SSAOp>(op)) {
        return ssaOp.id();
    } else if (auto cstOp = llvm::dyn_cast<ConstOp>(op)) {
        return cstOp.id();
    } else if (auto treelistop = llvm::dyn_cast<ListOp>(op)) {
        return treelistop.id();
    } else if (auto strop = llvm::dyn_cast<StrOp>(op)) {
        return strop.id();
    } else if (auto arrayop = llvm::dyn_cast<ArrayOp>(op)) {
        return arrayop.id();
    } else if (auto declop = llvm::dyn_cast<DeclBaseOp>(op)) {
        return declop.id();
    } else if (auto fieldop = llvm::dyn_cast<FieldDeclOp>(op)) {
        return fieldop.id();
    } else if (auto addressop = llvm::dyn_cast<AddressOp>(op)) {
        return addressop.id();
    } else if (auto constructorop = llvm::dyn_cast<ConstructorOp>(op)) {
        return constructorop.id();
    } else if (auto vecop = llvm::dyn_cast<VecOp>(op)) {
        return vecop.id();
    } else if (auto blockop = llvm::dyn_cast<BlockOp>(op)) {
        return blockop.id();
    } else if (auto compop = llvm::dyn_cast<ComponentOp>(op)) {
        return compop.id();
    } else if (auto phOp = llvm::dyn_cast<PlaceholderOp>(op)) {
        return phOp.id();
    }
    return 0;
}

/* dir: 1 or 2 */
void DataFlowAPI::CalDominanceInfo(uint64_t dir, uint64_t funcId)
{
    Json::Value root;
    string funName = __func__;
    root["dir"] = std::to_string(dir);
    root["funcId"] = std::to_string(funcId);
    string params = root.toStyledString();
    PluginServer::GetInstance()->RemoteCallClientWithAPI(funName, params);
}

vector<mlir::Operation*> DataFlowAPI::GetImmUseStmts(mlir::Value v)
{
    Json::Value root;
    string funName =  __func__;
    uint64_t varId = GetValueId(v);
    root["varId"] = std::to_string(varId);
    string params = root.toStyledString();
    vector<uint64_t> retIds = PluginServer::GetInstance()->GetIdsResult(funName, params);
    vector<mlir::Operation*> ops;
    for (auto id : retIds) {
        ops.push_back(PluginServer::GetInstance()->FindOperation(id));
    }
    return ops;
}

mlir::Value DataFlowAPI::GetGimpleVuse(uint64_t opId)
{
    Json::Value root;
    string funName = __func__;
    root["opId"] = std::to_string(opId);
    string params = root.toStyledString();
    return PluginServer::GetInstance()->GetValueResult(funName, params);
}

mlir::Value DataFlowAPI::GetGimpleVdef(uint64_t opId)
{
    Json::Value root;
    string funName = __func__;
    root["opId"] = std::to_string(opId);
    string params = root.toStyledString();
    return PluginServer::GetInstance()->GetValueResult(funName, params);
}

vector<mlir::Value> DataFlowAPI::GetSsaUseOperand(uint64_t opId)
{
    Json::Value root;
    string funName = __func__;
    root["opId"] = std::to_string(opId);
    string params = root.toStyledString();
    return PluginServer::GetInstance()->GetValuesResult(funName, params);
}

vector<mlir::Value> DataFlowAPI::GetSsaDefOperand(uint64_t opId)
{
    Json::Value root;
    string funName = __func__;
    root["opId"] = std::to_string(opId);
    string params = root.toStyledString();
    return PluginServer::GetInstance()->GetValuesResult(funName, params);
}

vector<mlir::Value> DataFlowAPI::GetPhiOrStmtUse(uint64_t opId)
{
    Json::Value root;
    string funName = __func__;
    root["opId"] = std::to_string(opId);
    string params = root.toStyledString();
    return PluginServer::GetInstance()->GetValuesResult(funName, params);
}

vector<mlir::Value> DataFlowAPI::GetPhiOrStmtDef(uint64_t opId)
{
    Json::Value root;
    string funName = __func__;
    root["opId"] = std::to_string(opId);
    string params = root.toStyledString();
    return PluginServer::GetInstance()->GetValuesResult(funName, params);
}

/* flag : 0 or 1 */
bool DataFlowAPI::RefsMayAlias(mlir::Value v1, mlir::Value v2, uint64_t flag)
{
    Json::Value root;
    string funName = __func__;
    uint64_t id1 = GetValueId(v1);
    uint64_t id2 = GetValueId(v2);
    root["id1"] = std::to_string(id1);
    root["id2"] = std::to_string(id2);
    root["flag"] = std::to_string(flag);
    string params = root.toStyledString();
    return PluginServer::GetInstance()->GetBoolResult(funName, params);
}

bool DataFlowAPI::PTIncludesDecl(mlir::Value ptr, uint64_t declId)
{
    Json::Value root;
    string funName = __func__;
    root["ptrId"] = std::to_string(GetValueId(ptr));
    root["declId"] = std::to_string(declId);
    string params = root.toStyledString();
    return PluginServer::GetInstance()->GetBoolResult(funName, params);
}

bool DataFlowAPI::PTsIntersect(mlir::Value ptr_1, mlir::Value ptr_2)
{
    Json::Value root;
    string funName = __func__;
    root["ptrId_1"] = std::to_string(GetValueId(ptr_1));
    root["ptrId_2"] = std::to_string(GetValueId(ptr_2));
    string params = root.toStyledString();
    return PluginServer::GetInstance()->GetBoolResult(funName, params);
}

}
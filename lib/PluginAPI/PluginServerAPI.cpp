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

#include "PluginAPI/PluginServerAPI.h"
#include "PluginServer/PluginJson.h"

namespace PluginAPI {
using namespace PinServer;
using namespace mlir::Plugin;

static bool CheckAttribute(string &attribute)
{
    if (attribute == "NULL") {
        fprintf(stderr, "param attribute is NULL,check fail!\n");
        return false;
    }
    return true;
}

static bool CheckID(uintptr_t id)
{
    if (id == 0) {
        return false;
    }
    return true;
}

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

int64_t PluginServerAPI::GetInjectDataAddress()
{
    string funName = __func__;
    string params = "";

    return PluginServer::GetInstance()->GetIntegerDataResult(funName, params);
}

string PluginServerAPI::GetDeclSourceFile(int64_t clientDataAddr)
{
    string funName = __func__;
    string params = std::to_string(clientDataAddr);

    return PluginServer::GetInstance()->GetStringDataResult(funName, params);
}

string PluginServerAPI::VariableName(int64_t clientDataAddr)
{
    string funName = __func__;
    string params = std::to_string(clientDataAddr);

    return PluginServer::GetInstance()->GetStringDataResult(funName, params);
}

string PluginServerAPI::FuncName(int64_t clientDataAddr)
{
    string funName = __func__;
    string params = std::to_string(clientDataAddr);

    return PluginServer::GetInstance()->GetStringDataResult(funName, params);
}

string PluginServerAPI::GetIncludeFile()
{
    string funName = __func__;
    string params = "";

    return PluginServer::GetInstance()->GetStringDataResult(funName, params);
}

int PluginServerAPI::GetDeclSourceLine(int64_t clientDataAddr)
{
    string funName = __func__;
    string params = std::to_string(clientDataAddr);

    return PluginServer::GetInstance()->GetIntegerDataResult(funName, params);
}

int PluginServerAPI::GetDeclSourceColumn(int64_t clientDataAddr)
{
    string funName = __func__;
    string params = std::to_string(clientDataAddr);

    return PluginServer::GetInstance()->GetIntegerDataResult(funName, params);
}

bool PluginServerAPI::SetCurrentDefInSSA(uint64_t varId, uint64_t defId)
{
    Json::Value root;
    string funName = __func__;
    root["varId"] = std::to_string(varId);
    root["defId"] = std::to_string(defId);
    string params = root.toStyledString();
    return PluginServer::GetInstance()->GetBoolResult(funName, params);
}

mlir::Value PluginServerAPI::GetCurrentDefFromSSA(uint64_t varId)
{
    Json::Value root;
    string funName = __func__;
    root["varId"] = std::to_string(varId);
    string params = root.toStyledString();
    return PluginServer::GetInstance()->GetValueResult(funName, params);
}

mlir::Value PluginServerAPI::CopySSAOp(uint64_t id)
{
    Json::Value root;
    string funName = __func__;
    root["id"] = std::to_string(id);
    string params = root.toStyledString();
    return PluginServer::GetInstance()->GetValueResult(funName, params);
}

mlir::Value PluginServerAPI::CreateSSAOp(mlir::Type t)
{
    Json::Value root;
    string funName = __func__;
    auto baseTy = t.dyn_cast<PluginIR::PluginTypeBase>();
    PinJson::PluginJson json;
    root = json.TypeJsonSerialize(baseTy);
    string params = root.toStyledString();
    return PluginServer::GetInstance()->GetValueResult(funName, params);
}

// CGnodeOp ===============

vector<CGnodeOp> PluginServerAPI::GetAllCGnode()
{
    Json::Value root;
    string funName = "GetCGnodeIDs";
    string params = root.toStyledString();
    vector<CGnodeOp> res;
    vector<uint64_t> ids = PluginServer::GetInstance()->GetIdsResult(funName, params);
    for (auto id : ids) {
        res.push_back(GetCGnodeOpById(id));
    }
    return res;
}

CGnodeOp PluginServerAPI::GetCGnodeOpById(uint64_t id)
{
    Json::Value root;
    string funName = __func__;
    root["id"] = std::to_string(id);
    string params = root.toStyledString();
    CGnodeOp cgnodeop = PluginServer::GetInstance()->GetCGnodeOpResult(funName, params);
    return cgnodeop;
}

bool PluginServerAPI::IsRealSymbolOfCGnode(uint64_t id)
{
    Json::Value root;
    string funName = __func__;
    root["id"] = std::to_string(id);
    string params = root.toStyledString();  
    return   PluginServer::GetInstance()->GetBoolResult(funName, params);
}

// ========================

vector<FunctionOp> PluginServerAPI::GetAllFunc()
{
    Json::Value root;
    string funName = "GetFunctionIDs";
    string params = root.toStyledString();
    vector<FunctionOp> res;
    vector<uint64_t> ids = PluginServer::GetInstance()->GetIdsResult(funName, params);
    for (auto id : ids) {
        res.push_back(GetFunctionOpById(id));
    }
    return res;
}

FunctionOp PluginServerAPI::GetFunctionOpById(uint64_t id)
{
    Json::Value root;
    string funName = __func__;
    root["id"] = std::to_string(id);
    string params = root.toStyledString();
    vector<FunctionOp> funcOps = PluginServer::GetInstance()->GetFunctionOpResult(funName, params);
    FunctionOp funOp = nullptr;
    if (funcOps.size())
        funOp = funcOps[0];
    return funOp;
}

PhiOp PluginServerAPI::GetPhiOp(uint64_t id)
{
    Json::Value root;
    string funName = __func__;
    root["id"] = std::to_string(id);
    string params = root.toStyledString();
    vector<mlir::Operation*> opRet = PluginServer::GetInstance()->GetOpResult(funName, params);
    return llvm::dyn_cast<PhiOp>(opRet[0]);
}

CallOp PluginServerAPI::GetCallOp(uint64_t id)
{
    Json::Value root;
    string funName = __func__;
    root["id"] = std::to_string(id);
    string params = root.toStyledString();
    vector<mlir::Operation*> opRet = PluginServer::GetInstance()->GetOpResult(funName, params);
    return llvm::dyn_cast<CallOp>(opRet[0]);
}

bool PluginServerAPI::SetLhsInCallOp(uint64_t callId, uint64_t lhsId)
{
    Json::Value root;
    string funName = __func__;
    root["callId"] = std::to_string(callId);
    root["lhsId"] = std::to_string(lhsId);
    string params = root.toStyledString();
    return PluginServer::GetInstance()->GetBoolResult(funName, params);
}

uint32_t PluginServerAPI::AddArgInPhiOp(uint64_t phiId, uint64_t argId, uint64_t predId, uint64_t succId)
{
    Json::Value root;
    string funName = __func__;
    root["phiId"] = std::to_string(phiId);
    root["argId"] = std::to_string(argId);
    root["predId"] = std::to_string(predId);
    root["succId"] = std::to_string(succId);
    string params = root.toStyledString();
    return PluginServer::GetInstance()->GetIdResult(funName, params);
}

uint64_t PluginServerAPI::CreateCondOp(uint64_t blockId, IComparisonCode iCode,
                                       uint64_t lhs, uint64_t rhs,
                                       uint64_t tbaddr, uint64_t fbaddr)
{
    Json::Value root;
    string funName = __func__;
    root["blockId"] = std::to_string(blockId);
    root["condCode"] = std::to_string(static_cast<int32_t>(iCode));
    root["lhsId"] = std::to_string(lhs);
    root["rhsId"] = std::to_string(rhs);
    root["tbaddr"] = std::to_string(tbaddr);
    root["fbaddr"] = std::to_string(fbaddr);
    string params = root.toStyledString();
    return PluginServer::GetInstance()->GetIdResult(funName, params);
}

uint64_t PluginServerAPI::CreateAssignOp(uint64_t blockId, IExprCode iCode, vector<uint64_t> &argIds)
{
    Json::Value root;
    string funName = __func__;
    root["blockId"] = std::to_string(blockId);
    root["exprCode"] = std::to_string(static_cast<int32_t>(iCode));
    Json::Value item;
    size_t idx = 0;
    for (auto v : argIds) {
        string idStr = "id" + std::to_string(idx++);
        item[idStr] = std::to_string(v);
    }
    root["argIds"] = item;
    string params = root.toStyledString();
    return PluginServer::GetInstance()->GetIdResult(funName, params);
}

uint64_t PluginServerAPI::CreateCallOp(uint64_t blockId, uint64_t funcId,
                                       vector<uint64_t> &argIds)
{
    Json::Value root;
    string funName = __func__;
    root["blockId"] = std::to_string(blockId);
    root["funcId"] = std::to_string(funcId);
    Json::Value item;
    size_t idx = 0;
    for (auto v : argIds) {
        string idStr = "id" + std::to_string(idx++);
        item[idStr] = std::to_string(v);
    }
    root["argIds"] = item;
    string params = root.toStyledString();
    return PluginServer::GetInstance()->GetIdResult(funName, params);
}

mlir::Value PluginServerAPI::CreateConstOp(mlir::Attribute attr, mlir::Type type)
{
    Json::Value root;
    string funName = __func__;
    auto baseTy = type.dyn_cast<PluginIR::PluginTypeBase>();
    PinJson::PluginJson json;
    root = json.TypeJsonSerialize(baseTy);
    string valueStr;
    if (type.isa<PluginIR::PluginIntegerType>()) {
        valueStr = std::to_string(attr.cast<mlir::IntegerAttr>().getInt());
    }
    root["value"] = valueStr;
    string params = root.toStyledString();
    return PluginServer::GetInstance()->GetValueResult(funName, params);
}

mlir::Value PluginServerAPI::GetResultFromPhi(uint64_t phiId)
{
    Json::Value root;
    string funName = __func__;
    root["id"] = std::to_string(phiId);
    string params = root.toStyledString();
    return PluginServer::GetInstance()->GetValueResult(funName, params);
}

PhiOp PluginServerAPI::CreatePhiOp(uint64_t argId, uint64_t blockId)
{
    Json::Value root;
    string funName = __func__;
    root["blockId"] = std::to_string(blockId);
    root["argId"] = std::to_string(argId);
    string params = root.toStyledString();
    vector<mlir::Operation*> opRet = PluginServer::GetInstance()->GetOpResult(funName, params);
    return llvm::dyn_cast<PhiOp>(opRet[0]);
}

mlir::Value PluginServerAPI::ConfirmValue(mlir::Value v)
{
    Json::Value root;
    string funName = "ConfirmValue";
    uint64_t valId = GetValueId(v);
    root["valId"] = std::to_string(valId);
    string params = root.toStyledString();
    return PluginServer::GetInstance()->GetValueResult(funName, params);
}

bool PluginServerAPI::IsVirtualOperand(uint64_t id)
{
    Json::Value root;
    string funName = "IsVirtualOperand";
    root["id"] = std::to_string(id);
    string params = root.toStyledString();
    return PluginServer::GetInstance()->GetBoolResult(funName, params);
}

mlir::Value PluginServerAPI::BuildMemRef(PluginIR::PluginTypeBase type,
                                         mlir::Value base, mlir::Value offset)
{
    Json::Value root;
    string funName = "BuildMemRef";
    uint64_t baseId = GetValueId(base);
    uint64_t offsetId = GetValueId(offset);
    root["baseId"] = baseId;
    root["offsetId"] = offsetId;
    PinJson::PluginJson json;
    root["type"] = json.TypeJsonSerialize(type);
    string params = root.toStyledString();
    return PluginServer::GetInstance()->GetValueResult(funName, params);
}

PluginIR::PluginTypeID PluginServerAPI::GetTypeCodeFromString(string type)
{
    if (type == "VoidTy") {
        return PluginIR::PluginTypeID::VoidTyID;
    } else if (type == "UIntegerTy1") {
        return PluginIR::PluginTypeID::UIntegerTy1ID;
    } else if (type == "UIntegerTy8") {
        return PluginIR::PluginTypeID::UIntegerTy8ID;
    } else if (type == "UIntegerTy16") {
        return PluginIR::PluginTypeID::UIntegerTy16ID;
    } else if (type == "UIntegerTy32") {
        return PluginIR::PluginTypeID::UIntegerTy32ID;
    } else if (type == "UIntegerTy64") {
        return PluginIR::PluginTypeID::UIntegerTy64ID;
    } else if (type == "IntegerTy1") {
        return PluginIR::PluginTypeID::IntegerTy1ID;
    } else if (type == "IntegerTy8") {
        return PluginIR::PluginTypeID::IntegerTy8ID;
    } else if (type == "IntegerTy16") {
        return PluginIR::PluginTypeID::IntegerTy16ID;
    } else if (type == "IntegerTy32") {
        return PluginIR::PluginTypeID::IntegerTy32ID;
    } else if (type == "IntegerTy64") {
        return PluginIR::PluginTypeID::IntegerTy64ID;
    } else if (type == "BooleanTy") {
        return PluginIR::PluginTypeID::BooleanTyID;
    } else if (type == "FloatTy") {
        return PluginIR::PluginTypeID::FloatTyID;
    } else if (type == "DoubleTy") {
        return PluginIR::PluginTypeID::DoubleTyID;
    } else if (type == "PointerTy") {
        return PluginIR::PluginTypeID::PointerTyID;
    } else if (type == "ArrayTy") {
        return PluginIR::PluginTypeID::ArrayTyID;
    } else if (type == "VectorTy") {
        return PluginIR::PluginTypeID::VectorTyID;
    } else if (type == "FunctionTy") {
        return PluginIR::PluginTypeID::FunctionTyID;
    } else if (type == "StructTy") {
        return PluginIR::PluginTypeID::StructTyID;
    }
    
    return PluginIR::PluginTypeID::UndefTyID;
}

vector<LocalDeclOp> PluginServerAPI::GetDecls(uint64_t funcID)
{
    Json::Value root;
    string funName("GetLocalDecls");
    root["funcId"] = std::to_string(funcID);
    string params = root.toStyledString();

    return PluginServer::GetInstance()->GetLocalDeclResult(funName, params);
}

vector<mlir::Plugin::DeclBaseOp> PluginServerAPI::GetFuncDecls(uint64_t funcID)
{
    Json::Value root;
    string funName = __func__;
    root["funcId"] = std::to_string(funcID);
    string params = root.toStyledString();

    return PluginServer::GetInstance()->GetFuncDeclsResult(funName, params);
}

mlir::Plugin::FieldDeclOp PluginServerAPI::MakeNode(IDefineCode code)
{
    Json::Value root;
    string funName = __func__;
    root["defCode"] = std::to_string(static_cast<int32_t>(code));
    string params = root.toStyledString();

    return PluginServer::GetInstance()->GetMakeNodeResult(funName, params);
}

llvm::SmallVector<mlir::Plugin::FieldDeclOp> PluginServerAPI::GetFields(uint64_t declID)
{
    Json::Value root;
    string funName = __func__;
    root["declId"] = std::to_string(declID);
    string params = root.toStyledString();

    return PluginServer::GetInstance()->GetFieldsResult(funName, params);
}

void PluginServerAPI::SetDeclName(uint64_t newfieldId, uint64_t fieldId)
{
    Json::Value root;
    string funName = __func__;
    root["newfieldId"] = std::to_string(newfieldId);
    root["fieldId"] = std::to_string(fieldId);
    string params = root.toStyledString();
    PluginServer::GetInstance()->RemoteCallClientWithAPI(funName, params);
}

void PluginServerAPI::SetDeclType(uint64_t newfieldId, uint64_t fieldId)
{
    Json::Value root;
    string funName = __func__;
    root["newfieldId"] = std::to_string(newfieldId);
    root["fieldId"] = std::to_string(fieldId);
    string params = root.toStyledString();
    PluginServer::GetInstance()->RemoteCallClientWithAPI(funName, params);
}

void PluginServerAPI::SetDeclAlign(uint64_t newfieldId, uint64_t fieldId)
{
    Json::Value root;
    string funName = __func__;
    root["newfieldId"] = std::to_string(newfieldId);
    root["fieldId"] = std::to_string(fieldId);
    string params = root.toStyledString();
    PluginServer::GetInstance()->RemoteCallClientWithAPI(funName, params);
}

void PluginServerAPI::SetUserAlign(uint64_t newfieldId, uint64_t fieldId)
{
    Json::Value root;
    string funName = __func__;
    root["newfieldId"] = std::to_string(newfieldId);
    root["fieldId"] = std::to_string(fieldId);
    string params = root.toStyledString();
    PluginServer::GetInstance()->RemoteCallClientWithAPI(funName, params);
}

unsigned PluginServerAPI::GetDeclTypeSize(uint64_t declId)
{
    Json::Value root;
    string funName = __func__;
    root["declId"] = std::to_string(declId);
    string params = root.toStyledString();
    return PluginServer::GetInstance()->GetIntegerDataResult(funName, params);
}

void PluginServerAPI::SetSourceLocation(uint64_t newfieldId, uint64_t fieldId)
{
    Json::Value root;
    string funName = __func__;
    root["newfieldId"] = std::to_string(newfieldId);
    root["fieldId"] = std::to_string(fieldId);
    string params = root.toStyledString();
    PluginServer::GetInstance()->RemoteCallClientWithAPI(funName, params);
}

void PluginServerAPI::SetAddressable(uint64_t newfieldId, uint64_t fieldId)
{
    Json::Value root;
    string funName = __func__;
    root["newfieldId"] = std::to_string(newfieldId);
    root["fieldId"] = std::to_string(fieldId);
    string params = root.toStyledString();
    PluginServer::GetInstance()->RemoteCallClientWithAPI(funName, params);
}

void PluginServerAPI::SetNonAddressablep(uint64_t newfieldId, uint64_t fieldId)
{
    Json::Value root;
    string funName = __func__;
    root["newfieldId"] = std::to_string(newfieldId);
    root["fieldId"] = std::to_string(fieldId);
    string params = root.toStyledString();
    PluginServer::GetInstance()->RemoteCallClientWithAPI(funName, params);
}

void PluginServerAPI::SetVolatile(uint64_t newfieldId, uint64_t fieldId)
{
    Json::Value root;
    string funName = __func__;
    root["newfieldId"] = std::to_string(newfieldId);
    root["fieldId"] = std::to_string(fieldId);
    string params = root.toStyledString();
    PluginServer::GetInstance()->RemoteCallClientWithAPI(funName, params);
}

void PluginServerAPI::SetDeclContext(uint64_t newfieldId, uint64_t declId)
{
    Json::Value root;
    string funName = __func__;
    root["newfieldId"] = std::to_string(newfieldId);
    root["declId"] = std::to_string(declId);
    string params = root.toStyledString();
    PluginServer::GetInstance()->RemoteCallClientWithAPI(funName, params);
}

void PluginServerAPI::SetDeclChain(uint64_t newfieldId, uint64_t fieldId)
{
    Json::Value root;
    string funName = __func__;
    root["newfieldId"] = std::to_string(newfieldId);
    root["fieldId"] = std::to_string(fieldId);
    string params = root.toStyledString();
    PluginServer::GetInstance()->RemoteCallClientWithAPI(funName, params);
}

mlir::Plugin::DeclBaseOp PluginServerAPI::BuildDecl(IDefineCode code, llvm::StringRef name, PluginIR::PluginTypeBase type)
{
    Json::Value root;
    string funName = __func__;
    root["defCode"] = std::to_string(static_cast<int32_t>(code));
    root["name"] = name.str();
    PinJson::PluginJson json;
    root["type"] = json.TypeJsonSerialize(type);
    string params = root.toStyledString();

    return PluginServer::GetInstance()->GetBuildDeclResult(funName, params);
}

void PluginServerAPI::SetTypeFields(uint64_t declId, uint64_t fieldId)
{
    Json::Value root;
    string funName = __func__;
    root["declId"] = std::to_string(declId);
    root["fieldId"] = std::to_string(fieldId);
    string params = root.toStyledString();
    PluginServer::GetInstance()->RemoteCallClientWithAPI(funName, params);
}

void PluginServerAPI::LayoutType(uint64_t declId)
{
    Json::Value root;
    string funName = __func__;
    root["declId"] = std::to_string(declId);
    string params = root.toStyledString();
    PluginServer::GetInstance()->RemoteCallClientWithAPI(funName, params);
}

PluginIR::PluginTypeBase PluginServerAPI::GetDeclType(uint64_t declId)
{
    Json::Value root;
    string funName = __func__;
    root["declId"] = std::to_string(declId);
    string params = root.toStyledString();
    return PluginServer::GetInstance()->GetDeclTypeResult(funName, params);
}

void PluginServerAPI::LayoutDecl(uint64_t declId)
{
    Json::Value root;
    string funName = __func__;
    root["declId"] = std::to_string(declId);
    string params = root.toStyledString();
    PluginServer::GetInstance()->RemoteCallClientWithAPI(funName, params);
}

mlir::Block* PluginServerAPI::BlockResult(const string& funName, const string& params)
{
    uint64_t blockId =  PluginServer::GetInstance()->GetIdResult(funName, params);
    return PluginServer::GetInstance()->FindBlock(blockId);
}

vector<mlir::Block*> PluginServerAPI::BlocksResult(const string& funName, const string& params)
{
    vector<mlir::Block*> res;
    PluginServer *server = PluginServer::GetInstance();
    vector<uint64_t> blockIds = server->GetIdsResult(funName, params);
    for (auto b : blockIds) {
        res.push_back(server->FindBlock(b));
    }
    return res;
}

vector<LoopOp> PluginServerAPI::GetLoopsFromFunc(uint64_t funcID)
{
    Json::Value root;
    string funName("GetLoopsFromFunc");
    root["funcId"] = std::to_string(funcID);
    string params = root.toStyledString();

    return PluginServer::GetInstance()->LoopOpsResult(funName, params);
}

bool PluginServerAPI::IsDomInfoAvailable()
{
    Json::Value root;
    string funName("IsDomInfoAvailable");
    return GetDomInfoAvaiResult(funName);
}

bool PluginServerAPI::GetDomInfoAvaiResult(const string& funName)
{
    Json::Value root;
    return PluginServer::GetInstance()->GetBoolResult(funName, root.toStyledString());
}

LoopOp PluginServerAPI::AllocateNewLoop(uint64_t funcID)
{
    Json::Value root;
    string funName("AllocateNewLoop");
    root["funcId"] = std::to_string(funcID);
    string params = root.toStyledString();

    return PluginServer::GetInstance()->LoopOpResult(funName, params);
}

LoopOp PluginServerAPI::GetLoopById(uint64_t loopID)
{
    Json::Value root;
    string funName("GetLoopById");
    root["loopId"] = std::to_string(loopID);
    string params = root.toStyledString();

    return PluginServer::GetInstance()->LoopOpResult(funName, params);
}

void PluginServerAPI::DeleteLoop(uint64_t loopID)
{
    Json::Value root;
    string funName("DeleteLoop");
    root["loopId"] = std::to_string(loopID);
    string params = root.toStyledString();
    PluginServer::GetInstance()->RemoteCallClientWithAPI(funName, params);
}

void PluginServerAPI::AddLoop(uint64_t loopID, uint64_t outerID, uint64_t funcID)
{
    Json::Value root;
    string funName("AddLoop");
    root["loopId"] = loopID;
    root["outerId"] = outerID;
    root["funcId"] = funcID;
    string params = root.toStyledString();
    PluginServer::GetInstance()->RemoteCallClientWithAPI(funName, params);
}

void PluginServerAPI::AddBlockToLoop(mlir::Block* b, LoopOp *loop)
{
    Json::Value root;
    string funName("AddBlockToLoop");
    root["blockId"] = PluginServer::GetInstance()->FindBasicBlock(b);
    root["loopId"] = loop->getIdAttr().getInt();
    string params = root.toStyledString();
    PluginServer::GetInstance()->RemoteCallClientWithAPI(funName, params);
}

bool PluginServerAPI::IsBlockInLoop(uint64_t loopID, uint64_t blockID)
{
    Json::Value root;
    string funName("IsBlockInside");
    root["loopId"] = std::to_string(loopID);
    root["blockId"] = std::to_string(blockID);
    string params = root.toStyledString();

    return PluginServer::GetInstance()->GetBoolResult(funName, params);
}

mlir::Block* PluginServerAPI::GetHeader(uint64_t loopID)
{
    Json::Value root;
    string funName("GetHeader");
    root["loopId"] = std::to_string(loopID);
    string params = root.toStyledString();

    return BlockResult(funName, params);
}

mlir::Block* PluginServerAPI::GetLatch(uint64_t loopID)
{
    Json::Value root;
    string funName("GetLatch");
    root["loopId"] = std::to_string(loopID);
    string params = root.toStyledString();

    return BlockResult(funName, params);
}

void PluginServerAPI::SetHeader(LoopOp * loop, mlir::Block* b)
{
    Json::Value root;
    string funName("SetHeader");
    root["loopId"] = std::to_string(loop->getIdAttr().getInt());
    root["blockId"] = std::to_string(
        PluginServer::GetInstance()->FindBasicBlock(b));
    string params = root.toStyledString();
    PluginServer::GetInstance()->RemoteCallClientWithAPI(funName, params);
}

void PluginServerAPI::SetLatch(LoopOp * loop, mlir::Block* b)
{
    Json::Value root;
    string funName("SetLatch");
    root["loopId"] = std::to_string(loop->getIdAttr().getInt());
    root["blockId"] = std::to_string(
        PluginServer::GetInstance()->FindBasicBlock(b));
    string params = root.toStyledString();
    PluginServer::GetInstance()->RemoteCallClientWithAPI(funName, params);
}

pair<mlir::Block*, mlir::Block*> PluginServerAPI::LoopSingleExit(uint64_t loopID)
{
    Json::Value root;
    string funName("GetLoopSingleExit");
    root["loopId"] = std::to_string(loopID);
    string params = root.toStyledString();
    return PluginServer::GetInstance()->EdgeResult(funName, params);
}

vector<pair<mlir::Block*, mlir::Block*> > PluginServerAPI::GetLoopExitEdges(uint64_t loopID)
{
    Json::Value root;
    string funName("GetLoopExits");
    root["loopId"] = std::to_string(loopID);
    string params = root.toStyledString();
    return PluginServer::GetInstance()->EdgesResult(funName, params);
}

vector<mlir::Block*> PluginServerAPI::GetLoopBody(uint64_t loopID)
{
    Json::Value root;
    string funName("GetBlocksInLoop");
    root["loopId"] = std::to_string(loopID);
    string params = root.toStyledString();
    return BlocksResult(funName, params);
}

LoopOp PluginServerAPI::GetBlockLoopFather(mlir::Block* b)
{
    Json::Value root;
    string funName("GetBlockLoopFather");
    root["blockId"] = std::to_string(
        PluginServer::GetInstance()->FindBasicBlock(b));
    string params = root.toStyledString();
    return PluginServer::GetInstance()->LoopOpResult(funName, params);
}

LoopOp PluginServerAPI::FindCommonLoop(LoopOp* loop_1, LoopOp* loop_2)
{
    Json::Value root;
    string funName("FindCommonLoop");
    root["loopId_1"] = loop_1->getIdAttr().getInt();
    root["loopId_2"] = loop_2->getIdAttr().getInt();
    string params = root.toStyledString();
    return PluginServer::GetInstance()->LoopOpResult(funName, params);
}

mlir::Block* PluginServerAPI::FindBlock(uint64_t b)
{
    PluginServer *server = PluginServer::GetInstance();
    return server->FindBlock(b);
}

uint64_t PluginServerAPI::FindBasicBlock(mlir::Block* b)
{
    PluginServer *server = PluginServer::GetInstance();
    return server->FindBasicBlock(b);
}

bool PluginServerAPI::InsertValue(uint64_t id, mlir::Value v)
{
    PluginServer *server = PluginServer::GetInstance();
    return server->InsertValue(id, v);
}

bool PluginServerAPI::RedirectFallthroughTarget(
    FallThroughOp& fop, mlir::Block* src, mlir::Block* dest)
{
    Json::Value root;
    string funName = __func__;
    root["src"] = PluginServer::GetInstance()->FindBasicBlock(src);
    root["dest"] = PluginServer::GetInstance()->FindBasicBlock(dest);
    string params = root.toStyledString();
    PluginServer::GetInstance()->RemoteCallClientWithAPI(funName, params);
    // update server
    fop->setSuccessor(dest, 0);
    return true;
}

mlir::Operation* PluginServerAPI::GetSSADefOperation(uint64_t addr)
{
    return PluginServer::GetInstance()->FindOperation(addr);
}

void PluginServerAPI::InsertCreatedBlock(uint64_t id, mlir::Block* block)
{
    PluginServer::GetInstance()->InsertCreatedBlock(id, block);
}

void PluginServerAPI::DebugValue(uint64_t valId)
{
    Json::Value root;
    string funName = __func__;
    root["valId"] = valId;
    string params = root.toStyledString();
    PluginServer::GetInstance()->RemoteCallClientWithAPI(funName, params);
}

void PluginServerAPI::DebugOperation(uint64_t opId)
{
    Json::Value root;
    string funName = __func__;
    root["opId"] = opId;
    string params = root.toStyledString();
    PluginServer::GetInstance()->RemoteCallClientWithAPI(funName, params);
}

void PluginServerAPI::DebugBlock(mlir::Block* b)
{
    PluginServer *server = PluginServer::GetInstance();
    Json::Value root;
    string funName = __func__;
    root["bbAddr"] = std::to_string(server->FindBasicBlock(b));
    string params = root.toStyledString();
    server->RemoteCallClientWithAPI(funName, params);
}

bool PluginServerAPI::IsLtoOptimize()
{
    Json::Value root;
    string funName = __func__;
    string params = root.toStyledString();
    return PluginServer::GetInstance()->GetBoolResult(funName, params);
}

bool PluginServerAPI::IsWholeProgram()
{
    Json::Value root;
    string funName = __func__;
    string params = root.toStyledString();
    return PluginServer::GetInstance()->GetBoolResult(funName, params);
}

} // namespace Plugin_IR

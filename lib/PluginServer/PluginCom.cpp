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
    This file contains the implementation of the PluginCom class.
    主要完成功能：和client之间通信、数据解析、数据反序列化
*/

#include "PluginServer/PluginCom.h"
#include "PluginServer/PluginLog.h"

namespace PinCom {
int64_t PluginCom::GetIntegerDataResult(void)
{
    int64_t result = integerResult;
    integerResult = 0; // clear
    return result;
}

string PluginCom::GetStringDataResult(void)
{
    string result = stringResult;
    stringResult.clear(); // clear
    return result;
}

vector<mlir::Plugin::FunctionOp> PluginCom::GetFunctionOpResult(void)
{
    vector<mlir::Plugin::FunctionOp> retOps = this->funcOpData;
    this->funcOpData.clear();
    this->opData.clear();
    return retOps;
}

vector<mlir::Operation *> PluginCom::GetOpResult(void)
{
    vector<mlir::Operation *> retOps = opData;
    opData.clear();
    return retOps;
}

vector<mlir::Plugin::LocalDeclOp> PluginCom::GetLocalDeclResult(void)
{
    vector<mlir::Plugin::LocalDeclOp> retOps = decls;
    decls.clear();
    return retOps;
}

mlir::Plugin::CGnodeOp PluginCom::GetCGnodeOpResult(void)
{
    mlir::Plugin::CGnodeOp retop = cgnode;
    return retop;
}

vector<mlir::Plugin::LoopOp> PluginCom::LoopOpsResult(void)
{
    vector<mlir::Plugin::LoopOp> retLoops = loops;
    loops.clear();
    return retLoops;
}

mlir::Plugin::LoopOp PluginCom::LoopOpResult(void)
{
    mlir::Plugin::LoopOp retLoop = loop;
    return retLoop;
}

std::pair<mlir::Block*, mlir::Block*> PluginCom::EdgeResult()
{
    std::pair<mlir::Block*, mlir::Block*> e;
    e.first = edge.first;
    e.second = edge.second;
    return e;
}

vector<std::pair<mlir::Block*, mlir::Block*> > PluginCom::EdgesResult()
{
    vector<std::pair<mlir::Block*, mlir::Block*> > retEdges;
    retEdges = edges;
    edges.clear();
    return retEdges;
}

bool PluginCom::GetBoolResult()
{
    return this->boolResult;
}

uint64_t PluginCom::GetIdResult()
{
    return this->idResult;
}

vector<uint64_t> PluginCom::GetIdsResult()
{
    vector<uint64_t> retIds = idsResult;
    idsResult.clear();
    return retIds;
}

mlir::Value PluginCom::GetValueResult()
{
    return this->valueResult;
}

vector<mlir::Plugin::PhiOp> PluginCom::GetPhiOpsResult()
{
    vector<mlir::Plugin::PhiOp> retOps;
    for (auto item : opData) {
        mlir::Plugin::PhiOp p = llvm::dyn_cast<mlir::Plugin::PhiOp>(item);
        retOps.push_back(p);
    }
    opData.clear();
    return retOps;
}

void PluginCom::JsonDeSerialize(const string& key, const string& data)
{
    if (key == "FuncOpResult") {
        json.FuncOpJsonDeSerialize(data, this->funcOpData);
    } else if (key == "CGnodeOpResult") {
        this->cgnode = json.CGnodeOpJsonDeSerialize(data);
    } else if (key == "LocalDeclOpResult") {
        json.LocalDeclOpJsonDeSerialize(data, this->decls);
    } else if (key == "LoopOpResult") {
        this->loop = json.LoopOpJsonDeSerialize (data);
    } else if (key == "LoopOpsResult") {
        json.LoopOpsJsonDeSerialize (data, this->loops);
    } else if (key == "BoolResult") {
        this->boolResult = (bool)atol(data.c_str());
    } else if (key == "VoidResult") {
        ;
    } else if (key == "EdgeResult") {
        json.EdgeJsonDeSerialize(data, this->edge);
    } else if (key == "EdgesResult") {
        json.EdgesJsonDeSerialize(data, this->edges);
    } else if (key == "IdsResult") {
        json.IdsJsonDeSerialize(data, this->idsResult);
    } else if (key == "IdResult") {
        this->idResult = atol(data.c_str());
    } else if (key == "OpsResult") {
        json.OpJsonDeSerialize(data.c_str(), this->opData);
    } else if (key == "ValueResult") {
        Json::Value node;
        Json::Reader reader;
        reader.parse(data, node);
        this->valueResult = json.ValueJsonDeSerialize(node);
    } else if (key == "GetPhiOps") {
        json.GetPhiOpsJsonDeSerialize(data, this->opData);
    } else if (key == "IntegerResult") {
        json.IntegerDeSerialize(data, integerResult);
    } else if (key == "StringResult") {
        json.StringDeSerialize(data, stringResult);
    } else {
        PinLog::PluginLog::GetInstance()->LOGE("not Json,key:%s,value:%s\n", key.c_str(), data.c_str());
    }
}
} // namespace PinCom


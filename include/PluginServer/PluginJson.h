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
    This file contains the declaration of the PluginJson class.
    主要完成功能：将序列化数据进行反序列化
*/

#ifndef PLUGIN_JSON_H
#define PLUGIN_JSON_H

#include <json/json.h>
#include "Dialect/PluginOps.h"
#include "Dialect/PluginTypes.h"

using std::string;
using std::map;
using std::vector;

namespace PinJson {
class PluginJson {
public:
    void FuncOpJsonDeSerialize(const string&, vector<mlir::Plugin::FunctionOp>&);
    void LocalDeclOpJsonDeSerialize(const string&,
                                    vector<mlir::Plugin::LocalDeclOp>&);
    void LoopOpsJsonDeSerialize(const string&, vector<mlir::Plugin::LoopOp>&);
    void EdgesJsonDeSerialize(const string&,
                              vector<std::pair<mlir::Block*, mlir::Block*>>&);
    void EdgeJsonDeSerialize(const string&, std::pair<mlir::Block*, mlir::Block*>&);
    void IdsJsonDeSerialize(const string&, vector<uint64_t>&);
    mlir::Operation *CallOpJsonDeSerialize(const string&);
    mlir::Operation *CondOpJsonDeSerialize(const string&);
    mlir::Operation *RetOpJsonDeSerialize(const string&);
    mlir::Operation *FallThroughOpJsonDeSerialize(const string&);
    mlir::Operation *PhiOpJsonDeSerialize(const string&);
    mlir::Operation *AssignOpJsonDeSerialize(const string&);
    void GetPhiOpsJsonDeSerialize(const string&, vector<mlir::Operation *>&);
    mlir::Value SSAOpJsonDeSerialize(const string& data);
    mlir::Plugin::LoopOp LoopOpJsonDeSerialize(const string& data);
    PluginIR::PluginTypeBase TypeJsonDeSerialize(const string& data);
    void OpJsonDeSerialize(const string&, vector<mlir::Operation *>&);
    /* 将整形数据反序列化 */
    void IntegerDeSerialize(const string& data, int64_t& result);
    /* 将字符串数据反序列化 */
    void StringDeSerialize(const string& data, string& result);
    /* 将json格式数据解析成map<string, string>格式 */
    void GetAttributes(Json::Value node, map<string, string>& attributes);
    mlir::Value ValueJsonDeSerialize(Json::Value valueJson);
    Json::Value TypeJsonSerialize(PluginIR::PluginTypeBase& type);
    mlir::Value MemRefDeSerialize(const string& data);
    bool ProcessBlock(mlir::Block*, mlir::Region&, const Json::Value&);
};
} // namespace PinJson

#endif

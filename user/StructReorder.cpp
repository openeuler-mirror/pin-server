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
    This file contains the implementation of the ArrayWidenPass class.
*/

#include <iostream>
#include <map>
#include <set>
#include <vector>
#include <string>
#include <sstream>
#include "PluginAPI/PluginServerAPI.h"
#include "PluginServer/PluginLog.h"
#include "PluginAPI/ControlFlowAPI.h"
#include "user/StructReorder.h"

namespace PluginOpt {
using std::string;
using std::vector;
using std::cout;
using namespace mlir;
using namespace mlir::Plugin;
using namespace PluginAPI;
using namespace PinServer;
using namespace std;

mlir::MLIRContext *context;
mlir::OpBuilder* opBuilder = nullptr;
std::map<Block*, Value> defs_map;
std::map<uint64_t, std::string> opNameMap;


static void ProcessStructReorder(uint64_t *fun)
{
    fprintf(stderr, "Running first pass, structreoder\n");

    PluginServerAPI pluginAPI;
    vector<CGnodeOp> allnodes = pluginAPI.GetAllCGnode();
    fprintf(stderr, "allnodes size is %d\n", allnodes.size());
    for (auto &nodeOp : allnodes) {
        context = nodeOp.getOperation()->getContext();
        mlir::OpBuilder opBuilder_temp = mlir::OpBuilder(context);
        opBuilder = &opBuilder_temp;
        string name = nodeOp.symbolNameAttr().getValue().str();
        fprintf(stderr, "Now process symbol : %s \n", name.c_str());
        uint32_t order = nodeOp.orderAttr().getInt();
        fprintf(stderr, "Now process order : %d \n", order);
        if (nodeOp.IsRealSymbol())
            fprintf(stderr, "Now process IsRealSymbol  \n");
    }

    vector<FunctionOp> allFunction = pluginAPI.GetAllFunc();
    fprintf(stderr, "allfun size is %d\n", allFunction.size());
    for (auto &funcOp : allFunction) {
        context = funcOp.getOperation()->getContext();
        mlir::OpBuilder opBuilder_temp = mlir::OpBuilder(context);
        opBuilder = &opBuilder_temp;
        string name = funcOp.funcNameAttr().getValue().str();
        fprintf(stderr, "Now process func : %s \n", name.c_str());
    }
    
}

int StructReorderPass::DoOptimize(uint64_t *fun)
{
    ProcessStructReorder(fun);
    return 0;
}
}
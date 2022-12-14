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

#include <iostream>
#include <map>
#include <set>
#include <vector>
#include <string>
#include <sstream>
#include "PluginAPI/PluginServerAPI.h"
#include "PluginServer/PluginLog.h"

using std::string;
using std::vector;
using std::cout;
using namespace mlir;
using namespace mlir::Plugin;
using namespace PluginAPI;
using namespace PinServer;
using namespace std;

static void UserOptimizeFunc(void)
{
    PluginServerAPI pluginAPI;
    vector<FunctionOp> allFunction = pluginAPI.GetAllFunc();
    int count = 0;
    for (size_t i = 0; i < allFunction.size(); i++) {
        if (allFunction[i].declaredInlineAttr().getValue())
            count++;
    }
    printf("declaredInline have %d functions were declared.\n", count);
}

static uint64_t getBlockAddress(mlir::Block* b)
{
    if (mlir::Plugin::CondOp oops = dyn_cast<mlir::Plugin::CondOp>(b->back())) {
        return oops.addressAttr().getInt();
    } else if (mlir::Plugin::FallThroughOp oops = dyn_cast<mlir::Plugin::FallThroughOp>(b->back())) {
        return oops.addressAttr().getInt();
    } else if (mlir::Plugin::RetOp oops = dyn_cast<mlir::Plugin::RetOp>(b->back())) {
        return oops.addressAttr().getInt();
    } else {
        assert(false);
    }
}

static void printBlock(mlir::Block* b)
{
    printf("[bb%ld]", getBlockAddress(b));
    printf(": has op %ld\n", b->getOperations().size());
    for (unsigned i = 0; i < b->getNumSuccessors(); i++) {
        printf("  --> ");
        printf("[bb%ld]", getBlockAddress(b->getSuccessor(i)));
        printf("\n");
    }
}

static void LocalVarSummery(void)
{
    PluginServerAPI pluginAPI;
    vector<mlir::Plugin::FunctionOp> allFunction = pluginAPI.GetAllFunc();
    map<string, string> args = PluginServer::GetInstance()->GetArgs();
    for (size_t i = 0; i < allFunction.size(); i++) {
        uint64_t funcID = allFunction[i].idAttr().getValue().getZExtValue();
        printf("In the %ldth function:\n", i);
        vector<mlir::Plugin::LocalDeclOp> decls = pluginAPI.GetDecls(funcID);
        int64_t typeFilter = -1u;
        if (args.find("type_code") != args.end()) {
            typeFilter = (int64_t)pluginAPI.GetTypeCodeFromString(args["type_code"]);
        }
        for (size_t j = 0; j < decls.size(); j++) {
            auto decl = decls[j];
            string name = decl.symNameAttr().getValue().str();
            int64_t declTypeID = decl.typeIDAttr().getValue().getZExtValue();
            if (declTypeID == typeFilter) {
                printf("\tFind %ldth target type %s\n", j, name.c_str());
            }
        }
    }
}

static void PassManagerSetupFunc(void)
{
    printf("PassManagerSetupFunc in\n");
}

static bool
determineLoopForm(LoopOp loop)
{
    if (loop.innerLoopIdAttr().getInt() != 0 || loop.numBlockAttr().getInt() != 3)
    {
        printf ("\nWrong loop form, there is inner loop or redundant bb.\n");
        return false;
    }

    if (loop.GetSingleExit().first != 0 || !loop.GetLatch())
    {
        printf ("\nWrong loop form, only one exit or loop_latch does not exist.\n");
        return false;
    }
    return true;
}

static void
ProcessArrayWiden(void)
{
    std::cout << "Running first pass, awiden\n";

    PluginServerAPI pluginAPI;
    vector<FunctionOp> allFunction = pluginAPI.GetAllFunc();

    for (auto &funcOp : allFunction) {
        string name = funcOp.funcNameAttr().getValue().str();
        printf("Now process func : %s \n", name.c_str());
        vector<LoopOp> allLoop = funcOp.GetAllLoops();
        for (auto &loop : allLoop) {
            if (determineLoopForm(loop)) {
                printf("The %ldth loop form is success matched, and the loop can be optimized.\n", loop.indexAttr().getInt());
                return;
            }
        }
    }
}

void RegisterCallbacks(void)
{
    // PluginServer::GetInstance()->RegisterUserFunc(HANDLE_BEFORE_IPA, UserOptimizeFunc);
    // PluginServer::GetInstance()->RegisterUserFunc(HANDLE_BEFORE_IPA, LocalVarSummery);
    ManagerSetupData setupData;
    setupData.refPassName = PASS_PHIOPT;
    setupData.passNum = 1;
    setupData.passPosition = PASS_INSERT_AFTER;
    PluginServer::GetInstance()->RegisterPassManagerSetup(HANDLE_MANAGER_SETUP, setupData, ProcessArrayWiden);
}

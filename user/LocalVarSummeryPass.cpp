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
    This file contains the implementation of the LocalVarSummeryPass pass.
*/

#include "PluginAPI/ControlFlowAPI.h"
#include "PluginAPI/PluginServerAPI.h"
#include "user/LocalVarSummeryPass.h"

namespace PluginOpt {
using std::string;
using std::vector;
using std::cout;
using namespace mlir;
using namespace PluginAPI;

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
        mlir::Plugin::FunctionOp funcOp = allFunction[i];
        printf("func name is :%s\n", funcOp.funcNameAttr().getValue().str().c_str());
        mlir::Type dgyty = funcOp.type();
        if (auto ty = dgyty.dyn_cast<PluginIR::PluginFunctionType>()) {
            if(auto stTy = ty.getReturnType().dyn_cast<PluginIR::PluginStructType>()) {
                printf("func return type is PluginStructType\n");
                std::string tyName = stTy.getName();
                printf("    struct name is : %s\n", tyName.c_str());
                
                llvm::ArrayRef<std::string> paramsNames = stTy.getElementNames();
                for (auto name :paramsNames) {
                    std::string pName = name;
                    printf("\n    struct argname is : %s\n", pName.c_str());
                }
            }
            if(auto stTy = ty.getReturnType().dyn_cast<PluginIR::PluginVectorType>()) {
                printf("func return type is PluginVectorType\n");
                printf("    vector elem num : %d\n", stTy.getNumElements());
                printf("    vector elem type id : %d\n", stTy.getElementType().dyn_cast<PluginIR::PluginTypeBase>().getPluginTypeID());
            }
            size_t paramIndex = 0;
            llvm::ArrayRef<mlir::Type> paramsType = ty.getParams();
            for (auto ty : ty.getParams()) {
                printf("\n    Param index : %ld\n", paramIndex++);
                printf("\n    Param type id : %d\n", ty.dyn_cast<PluginIR::PluginTypeBase>().getPluginTypeID());
            }
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

int LocalVarSummeryPass::DoOptimize()
{
    LocalVarSummery();
    return 0;
}
}

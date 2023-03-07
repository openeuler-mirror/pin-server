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

static void dump_structtype(PluginIR::PluginTypeBase type)
{
    if (auto stTy = type.dyn_cast<PluginIR::PluginStructType>()) {
        std::string tyName = stTy.getName();
        fprintf(stderr, "    struct name is : %s\n", tyName.c_str());

        llvm::ArrayRef<std::string> paramsNames = stTy.getElementNames();
        for (auto name :paramsNames) {
            std::string pName = name;
            fprintf(stderr, "\n    struct argname is : %s\n", pName.c_str());
        }
    }
}


static void reorder_fields(FieldDeclOp& field, FieldDeclOp& newfield)
{
    if (!field) {
        field = newfield;
        unsigned size = newfield.GetTypeSize();
    } else {
        FieldDeclOp tmp = field;
        // unsigned size = tmp.getResultType().dyn_cast<PluginIR::PluginTypeBase>().getPluginTypeID();
        unsigned size = newfield.GetTypeSize();
        if (newfield.GetTypeSize() > tmp.GetTypeSize()) {
            newfield.SetDeclChain(tmp);
            field = newfield;
        }
    }
}

static void create_new_fields(mlir::Plugin::DeclBaseOp& decl, llvm::SmallVector<mlir::Plugin::FieldDeclOp> recordfields)
{
    PluginAPI::PluginServerAPI pluginAPI;
    FieldDeclOp fd;
    for (auto &fielddecl : recordfields) {
        FieldDeclOp field = pluginAPI.MakeNode(IDefineCode::FieldDecl);
        field.SetName(fielddecl);
        field.SetType(fielddecl);
        field.SetDeclAlign(fielddecl);

        field.SetSourceLocation(fielddecl); 
        field.SetUserAlign(fielddecl);
        field.SetAddressable(fielddecl);
        field.SetNonAddressablep(fielddecl);
        field.SetVolatile(fielddecl);
        field.SetDeclContext(decl.idAttr().getInt());

        reorder_fields(fd, field);

    }
    pluginAPI.SetTypeFields(decl.idAttr().getInt(), fd.idAttr().getInt());
    pluginAPI.LayoutType(decl.idAttr().getInt());
    pluginAPI.LayoutDecl(decl.idAttr().getInt());
    fprintf(stderr, "reorder struct type after :>>>\n");
    dump_structtype(pluginAPI.GetDeclType(decl.idAttr().getInt()));
}

static void create_new_type(mlir::Plugin::DeclBaseOp& decl, llvm::SmallVector<mlir::Plugin::FieldDeclOp> recordfields)
{
    create_new_fields(decl, recordfields);
}

static void create_new_types(llvm::SmallVector<PluginIR::PluginTypeBase> recordTypes,
            llvm::SmallVector<DeclBaseOp> recordDecls, llvm::SmallVector<mlir::Plugin::FieldDeclOp> recordFields)
{
    for (int i = 0; i < recordTypes.size(); i++) {

        auto type = recordTypes[i].dyn_cast<PluginIR::PluginStructType>();
        mlir::MLIRContext m_context;
        m_context.getOrLoadDialect<PluginDialect>();
        PluginIR::PluginTypeBase rPluginType = PluginIR::PluginUndefType::get(&m_context);
        StringRef name = type.getName();
        StringRef newName = name.str() + ".reorg." + to_string(i);
        PluginAPI::PluginServerAPI pluginAPI;
        DeclBaseOp decl = pluginAPI.BuildDecl(IDefineCode::TYPEDECL, newName, rPluginType);

        create_new_type(decl, recordFields);  

    }

}

static void record_decl(mlir::Plugin::DeclBaseOp decl, llvm::SmallVector<mlir::Plugin::DeclBaseOp>& recordDecls)
{
    if (llvm::find(recordDecls, decl) == recordDecls.end())
    {
        recordDecls.push_back(decl);
    }
}

static void record_fields(DeclBaseOp decl, llvm::SmallVector<mlir::Plugin::FieldDeclOp>& recordFields)
{
    PluginAPI::PluginServerAPI pluginAPI;
    llvm::SmallVector<mlir::Plugin::FieldDeclOp> fields = pluginAPI.GetFields(decl.idAttr().getInt());
    recordFields.insert(recordFields.end(), fields.begin(), fields.end());
}

static PluginIR::PluginTypeBase record_type(PluginIR::PluginTypeBase type, llvm::SmallVector<PluginIR::PluginTypeBase>& recordTypes,
                                            DeclBaseOp decl, llvm::SmallVector<mlir::Plugin::FieldDeclOp>& recordFields)
{
    if (llvm::find(recordTypes, type) == recordTypes.end())
    {
        recordTypes.push_back(type);
    }
    record_fields(decl, recordFields);
    return type;
}

static PluginIR::PluginTypeBase inner_type(PluginIR::PluginTypeBase type)
{
    while(type.isa<PluginIR::PluginPointerType>() || type.isa<PluginIR::PluginArrayType>()) {
        if (auto t = type.dyn_cast<PluginIR::PluginPointerType>()) {
            type = t.getElementType().dyn_cast<PluginIR::PluginTypeBase>();
        } else if (auto t = type.dyn_cast<PluginIR::PluginArrayType>()) {
            type = t.getElementType().dyn_cast<PluginIR::PluginTypeBase>();
        }
    }
    return type;
}


static bool handle_type(PluginIR::PluginTypeBase type)
{
    type = inner_type(type);
    if (type.isa<PluginIR::PluginStructType>()) {
        fprintf(stderr, "handle struct type :>>>\n");
        dump_structtype(type);
        return true;
    }
    return false;
}

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
        uint32_t order = nodeOp.orderAttr().getInt();
        if (nodeOp.IsRealSymbol())
            fprintf(stderr, "process RealSymbol : %s/%d \n", name.c_str(), order);
    }

    vector<FunctionOp> allFunction = pluginAPI.GetAllFunc();
    llvm::SmallVector<PluginIR::PluginTypeBase> recordTypes;
    llvm::SmallVector<mlir::Plugin::DeclBaseOp> recordDecls;
    llvm::SmallVector<mlir::Plugin::FieldDeclOp> recordFields;
    fprintf(stderr, "allfun size is %d\n", allFunction.size());
    for (auto &funcOp : allFunction) {
        context = funcOp.getOperation()->getContext();
        mlir::OpBuilder opBuilder_temp = mlir::OpBuilder(context);
        opBuilder = &opBuilder_temp;
        string name = funcOp.funcNameAttr().getValue().str();
        fprintf(stderr, "Now process func : %s \n", name.c_str());
        uint64_t funcID = funcOp.idAttr().getValue().getZExtValue();

        vector<mlir::Plugin::DeclBaseOp> decls = pluginAPI.GetFuncDecls(funcID);

        for (auto &decl : decls) {
            auto type = decl.getResultType().dyn_cast<PluginIR::PluginTypeBase>();
            if (!handle_type(type)) continue;
            type = record_type(inner_type(type), recordTypes, decl, recordFields);

            record_decl(decl, recordDecls);
        }

        create_new_types(recordTypes, recordDecls, recordFields);
    }
    
}

int StructReorderPass::DoOptimize(uint64_t *fun)
{
    ProcessStructReorder(fun);
    return 0;
}

bool StructReorderPass::Gate()
{
    PluginServerAPI pluginAPI;
    if (pluginAPI.IsLtoOptimize()) {
        fprintf(stderr, "\n The LTO flag is open \n");
        return true;
    }
    if (pluginAPI.IsWholeProgram()) {
        fprintf(stderr, "\n The whole program flag is open \n");
        return true;
    }
    return false;
}
}
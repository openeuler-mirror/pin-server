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
#include "PluginAPI/PluginAPI_Server.h"
#include "Plugin_Log.h"
#include "PluginServer.h"

using std::string;
using std::vector;
using std::cout;
using namespace Plugin_API;
using namespace Plugin_Server_LOG;
using namespace std;

static void PrintOperation(vector<Operation> &p)
{
    int i = 0;
    for (auto& v : p) {
        printf("operation%d:\n", i++);
        cout << "id:" << v.GetID() << "\n";
        cout << "Opcode:" << v.GetOpcode() << "\n";
        cout << "attributes:" << "\n";
        for (auto m = v.GetAttributes().rbegin(); m != v.GetAttributes().rend(); m++) {
            cout << "    " << m->first << ":" << m->second << "\n";
        }

        cout << "resultType:" << "\n";
        cout << "    id:" << v.GetResultTypes().GetID() << "\n";
        cout << "    typeCode:" << v.GetResultTypes().GetTypeCode() << "\n";
        for (auto m = v.GetResultTypes().GetAttributes().rbegin();
            m != v.GetResultTypes().GetAttributes().rend(); m++) {
            cout << "    " << m->first << ":" << m->second << "\n";
        }

        cout << "operands:" << "\n";
        int j = 0;
        for (auto m = v.GetOperands().rbegin(); m != v.GetOperands().rend(); m++) {
            cout << "    " << m->first << ":" << "\n";
            Decl decl = m->second;
            cout << "        id:" << decl.GetID() << "\n";
            cout << "        declCode:" << decl.GetDeclCode() << "\n";
            cout << "        attributes:" << "\n";
            for (auto n = decl.GetAttributes().rbegin(); n != decl.GetAttributes().rend(); n++) {
                cout << "            " << n->first << ":" << n->second << "\n";
            }
            cout << "        declType:" << "\n";
            Type type = decl.GetType();
            cout << "            id:" << type.GetID() << "\n";
            cout << "            typeCode:" << type.GetTypeCode() << "\n";
            cout << "            attributes:" << "\n";
            for (auto n = type.GetAttributes().rbegin(); n != type.GetAttributes().rend(); n++) {
                cout << "                " << n->first << ":" << n->second << "\n";
            }
        }
    }
}

static void UserOptimizeFunc(void)
{
    PluginAPI_Server pluginAPI;
    vector<Operation> inlineFunction = pluginAPI.SelectOperation(OP_FUNCTION, "declaredInline");
    LOGI("declaredInline have %ld functions were declared inline.\n", inlineFunction.size());
    printf("declaredInline have %ld functions were declared inline.\n", inlineFunction.size());
    vector<Operation> allFunction = pluginAPI.GetAllFunc("name");
    LOGI("allfunction have %ld functions were declared\n", allFunction.size());
    printf("allfunction have %ld functions were declared\n", allFunction.size());
}

static void Split(const string& s, vector<string>& token, const string& delimiters = " ")
{
    istringstream iss(s);
    string str;
    while (getline(iss, str, delimiters.c_str()[0])) {
        token.push_back(str);
    }
}

static void VariablesSummery(void)
{
    PluginAPI_Server pluginAPI;
    vector<Operation> allFunction = pluginAPI.SelectOperation(OP_FUNCTION, "");
    map<string, string> args = PluginServer::GetInstance()->GetArgs();

    for (auto& f : allFunction) {
        printf("\nvariables_summary for %s: \n", f.GetAttribute("name").c_str());
        if (f.GetAttribute("localDecl") == "") {
            printf("0");
            continue;
        }
        vector<string> localDeclStr;
        Split(f.GetAttribute("localDecl"), localDeclStr, ",");
        for (auto& s : localDeclStr) {
            if (s == "") continue;
            Decl decl = pluginAPI.SelectDeclByID(stol(s.c_str()));
            if (args.find("type_code") != args.end()) {
                if ((decl.GetType().GetTypeCode() == pluginAPI.GetTypeCodeFromString(args["type_code"]))) {
                    printf("%s %s;", decl.GetAttribute("name").c_str(), args["type_code"].c_str());
                }
            } else {
                if ((decl.GetType().GetTypeCode() == TC_I32) || (decl.GetType().GetTypeCode() == TC_I16)) {
                    printf("%s;", decl.GetAttribute("name").c_str());
                }
            }
        }
    }
}

static void AllFunc(void)
{
    PluginAPI_Server pluginAPI;
    vector<Operation> allFunction = pluginAPI.GetAllFunc("name");
    printf("allfunction have %ld functions were declared\n", allFunction.size());
}

string UserInit(void)
{
    string ret = "HANDLE_BEFORE_IPA";
    PluginServer::GetInstance()->RegisterUserOptimize(UserOptimizeFunc);
    PluginServer::GetInstance()->RegisterUserOptimize(VariablesSummery);
    PluginServer::GetInstance()->RegisterUserOptimize(AllFunc);
    return ret;
}

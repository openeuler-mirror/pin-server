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
#include "PluginServer/PluginLog.h"

using std::string;
using std::vector;
using std::cout;
using namespace Plugin_API;
using namespace PinServer;
using namespace std;

static void UserOptimizeFunc(void)
{
    PluginAPI_Server pluginAPI;
    vector<Operation> inlineFunction = pluginAPI.SelectOperation(OP_FUNCTION, "declaredInline");
    LOGI("declaredInline have %ld functions were declared inline.\n", inlineFunction.size());
    printf("declaredInline have %ld functions were declared inline.\n", inlineFunction.size());
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
    printf("\n");
}

static void AllFunc(void)
{
    PluginAPI_Server pluginAPI;
    vector<Operation> allFunction = pluginAPI.GetAllFunc("name");
    printf("allfunction have %ld functions were declared\n", allFunction.size());
}

void RegisterCallbacks(void)
{
    PluginServer::GetInstance()->RegisterUserFunc(HANDLE_BEFORE_IPA, "UserOptimizeFunc", UserOptimizeFunc);
    PluginServer::GetInstance()->RegisterUserFunc(HANDLE_AFTER_IPA, "VariablesSummery", VariablesSummery);
    PluginServer::GetInstance()->RegisterUserFunc(HANDLE_AFTER_IPA, "AllFunc", AllFunc);
}

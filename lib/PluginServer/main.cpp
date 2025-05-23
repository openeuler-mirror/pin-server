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
    This file contains the implementation of the main.
*/

#include "PluginServer/PluginServer.h"

using namespace PinServer;

int main(int argc, char** argv)
{
    const int argcNum = 7;
    if (argc != argcNum) {
        printf("param num:%d, should be:%d\n", argc, argcNum);
        return -1;
    }
    std::string port = argv[0];
    LogPriority priority = (LogPriority)atoi(argv[1]);
    PluginServer server(priority, port);
    server.SetServerCommand(argv[2], argv[3], argv[4], argv[5], argv[6]);
    server.RunServer();
    return 0;
}

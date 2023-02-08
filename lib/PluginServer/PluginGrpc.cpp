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
    This file contains the implementation of the PluginGrpc class.
    主要完成功能：完成grpc server服务的注册，提供server和client之间grpc收发接口
*/

#include <thread>
#include "PluginServer/PluginServer.h"

namespace PinGrpc {
void PluginGrpc::ServerSend(const string& key, const string& value)
{
    ServerMsg serverMsg;
    serverMsg.set_attribute(key);
    serverMsg.set_value(value);
    grpcStream->Write(serverMsg);
}

Status PluginGrpc::ReceiveSendMsg(ServerContext* context, ServerReaderWriter<ServerMsg, ClientMsg>* stream)
{
    ClientMsg clientMsg;
    grpcStream = stream;

    while (stream->Read(&clientMsg)) {
        PinServer::PluginServer::GetInstance()->ClientMsgProc(clientMsg.attribute(), clientMsg.value());
    }
    return Status::OK;
}

void PluginGrpc::ServerMonitorThread()
{
    int delay = 100000; // 100ms
    pid_t initPid = 1;
    while (1) {
        if (shutdown || (getppid() == initPid)) {
            grpcServer->Shutdown();
            break;
        }
        usleep(delay);
    }
}

bool PluginGrpc::RegisterServer(const string& port)
{
    string serverAddress = "0.0.0.0:" + port;
    ServerBuilder builder;
    int serverPort = 0;
    // Listen on the given address without any authentication mechanism.
    builder.AddListeningPort(serverAddress, grpc::InsecureServerCredentials(), &serverPort);
    builder.RegisterService(this);
    grpcServer = std::unique_ptr<Server>(builder.BuildAndStart());
    if (serverPort != atoi(port.c_str())) {
        return false;
    }
    return true;
}

void PluginGrpc::Run()
{
    std::thread serverExtiThread(&PluginGrpc::ServerMonitorThread, this);
    serverExtiThread.join();

    // Wait for the server to shutdown. Note that some other thread must be
    // responsible for shutting down the server for this call to ever return.
    grpcServer->Wait();
}
} // namespace PinGrpc

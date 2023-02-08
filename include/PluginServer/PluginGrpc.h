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
    This file contains the declaration of the GrpcService class.
    主要完成功能：完成grpc server服务的注册，提供server和client之间grpc收发接口
*/

#ifndef PLUGIN_GRPC_H
#define PLUGIN_GRPC_H

#include <grpcpp/grpcpp.h>
#include "plugin.grpc.pb.h"

namespace PinGrpc {
using grpc::Server;
using grpc::ServerBuilder;
using grpc::ServerContext;
using grpc::ServerReaderWriter;
using grpc::Status;

using plugin::PluginService;
using plugin::ClientMsg;
using plugin::ServerMsg;
using std::string;

class PluginGrpc final : public PluginService::Service {
public:
    PluginGrpc()
    {
        shutdown = false;
    }
    /* 定义的grpc服务端和客户端通信的接口函数 */
    Status ReceiveSendMsg(ServerContext* context, ServerReaderWriter<ServerMsg, ClientMsg>* stream) override;
    /* 服务端发送数据给client接口 */
    void ServerSend(const string& key, const string& value);
    bool RegisterServer(const string& port);
    void Run();
    void ShutDown()
    {
        shutdown = true;
    }

private:
    void ServerMonitorThread(); // 监听线程,shutdown为true时,grpc server退出
    bool shutdown; // 是否关闭grpc server
    ServerReaderWriter<ServerMsg, ClientMsg> *grpcStream; // 保存server和client通信的grpc stream指针
    std::unique_ptr<Server> grpcServer;
};
} // namespace PinGrpc

#endif

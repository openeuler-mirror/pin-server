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
    This file contains the declaration of the PluginOptBase class.
    主要完成功能：提供优化基类，gate为进入条件，DoOptimize为执行函数，RegisterCallbacks为注册函数
*/

#ifndef PLUGIN_OPTBASE_H
#define PLUGIN_OPTBASE_H

#include "PluginServer/ManagerSetup.h"
#include "Dialect/PluginDialect.h"
#include "Dialect/PluginOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Builders.h"

namespace PluginOpt {
enum InjectPoint : uint8_t {
    HANDLE_PARSE_TYPE = 0,
    HANDLE_PARSE_DECL,
    HANDLE_PRAGMAS,
    HANDLE_PARSE_FUNCTION,
    HANDLE_BEFORE_IPA,
    HANDLE_AFTER_IPA,
    HANDLE_BEFORE_EVERY_PASS,
    HANDLE_AFTER_EVERY_PASS,
    HANDLE_BEFORE_ALL_PASS,
    HANDLE_AFTER_ALL_PASS,
    HANDLE_COMPILE_END,
    HANDLE_MANAGER_SETUP,
    HANDLE_INCLUDE_FILE,
    HANDLE_MAX,
};

class PluginOptBase {
public:
    PluginOptBase(InjectPoint inject)
    {
        this->inject = inject;
        context.getOrLoadDialect<mlir::Plugin::PluginDialect>();
    }
    virtual ~PluginOptBase() = default;
    virtual bool Gate() = 0;
    virtual int DoOptimize() = 0;
    InjectPoint GetInject()
    {
        return inject;
    }
    void SetFuncAddr(uint64_t add)
    {
        func = add;
    }
    uint64_t GetFuncAddr(void)
    {
        return func;
    }
    mlir::MLIRContext *GetContext()
    {
        return &(this->context);
    }

private:
    mlir::MLIRContext context;
    InjectPoint inject;
    uint64_t func; // 保存managerSetup fun参数指针
};
} // namespace PluginOpt
#endif

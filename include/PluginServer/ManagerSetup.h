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
    This file contains the declaration of the ManagerSetup class.
    主要完成功能：提供managerSetup注册方法
*/

#ifndef MANAGER_SETUP_H
#define MANAGER_SETUP_H

namespace PluginOpt {
enum RefPassName {
    PASS_CFG,
    PASS_PHIOPT,
    PASS_SSA,
    PASS_LOOP,
    PASS_LAD,
    PASS_MAC,
};

enum PassPosition {
    PASS_INSERT_AFTER,
    PASS_INSERT_BEFORE,
    PASS_REPLACE,
};

class ManagerSetup {
public:
    ManagerSetup(RefPassName name, int num, PassPosition position)
    {
        refPassName = name;
        passNum = num;
        passPosition = position;
    }
    RefPassName GetPassName()
    {
        return refPassName;
    }
    int GetPassNum()
    {
        return passNum;
    }
    PassPosition GetPassPosition()
    {
        return passPosition;
    }

private:
    RefPassName refPassName;
    int passNum; // 指定passName的第几次执行作为参考点
    PassPosition passPosition; // 指定pass是添加在参考点之前还是之后
};
} // namespace PinOpt

#endif
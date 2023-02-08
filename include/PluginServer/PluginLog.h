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
    This file contains the declaration of the Plugin_Log class.
    主要完成功能：提供LOGE、LOGW、LOGI、LOGD四个log保存接口，并提供SetLogPriority接口
    设置log级别
*/

#ifndef PLUGIN_LOG_H
#define PLUGIN_LOG_H

#include <string>
#include <memory>

namespace PinLog {
using std::string;

enum LogPriority : uint8_t {
    PRIORITY_ERROR = 0,
    PRIORITY_WARN,
    PRIORITY_INFO,
    PRIORITY_DEBUG
};

constexpr int LOG_BUF_SIZE = 102400;
constexpr int BASE_DATE = 1900;
class PluginLog {
public:
    PluginLog();
    ~PluginLog()
    {
        CloseLog();
    }
    void CloseLog();
    bool SetPriority(LogPriority pri);
    void SetFileSize(unsigned int size)
    {
        logFileSize = size;
    }
    void LOGE(const char *fmt, ...);
    void LOGW(const char *fmt, ...);
    void LOGI(const char *fmt, ...);
    void LOGD(const char *fmt, ...);
    static PluginLog *GetInstance();

private:
    void LogPrint(LogPriority priority, const char *tag, const char *fmt);
    void LogWrite(const char *tag, const char *msg);
    void LogWriteFile(const string& data);
    void GetLogFileName(string& fileName);
    LogPriority priority;
    unsigned int logFileSize;
    std::shared_ptr<std::fstream> logFs;
    char logBuf[LOG_BUF_SIZE];
};
} // namespace PinLog

#endif

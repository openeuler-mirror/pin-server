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
    This file contains the implementation of the Plugin_Log class.
*/

#include <cstring>
#include <cstdarg>
#include <iostream>
#include <ctime>
#include <fstream>
#include <mutex>
#include <csignal>
#include <unistd.h>
#include "PluginServer/PluginLog.h"

namespace PinLog {
static std::mutex g_mutex; // 线程锁
PluginLog g_pluginLog;
const int LOG_DEFAULT_SIZE = 10 * 1024 * 1024;

PluginLog::PluginLog()
{
    priority = PRIORITY_WARN;
    logFileSize = LOG_DEFAULT_SIZE;
}

PluginLog *PluginLog::GetInstance()
{
    return &g_pluginLog;
}

void PluginLog::GetLogFileName(string& fileName)
{
    time_t nowTime = time(nullptr);
    if (nowTime == -1) {
        fprintf(stderr, "%s fail\n", __func__);
    }
    struct tm *t = localtime(&nowTime);
    char buf[100];
    int ret = sprintf(buf, "/tmp/pin_server%d_%4d%02d%02d_%02d_%02d_%02d.log", getppid(),
        t->tm_year + BASE_DATE, t->tm_mon + 1, t->tm_mday, t->tm_hour, t->tm_min, t->tm_sec);
    if (ret < 0) {
        fprintf(stderr, "%s sprintf fail\n", __func__);
    }
    fileName = buf;
}

void PluginLog::LogWriteFile(const string& data)
{
    string fileName;
    if (logFs == nullptr) {
        logFs = std::make_shared<std::fstream>();
        GetLogFileName(fileName);
        logFs->open(fileName.c_str(), std::ios::app);
    }

    if (logFs->tellg() > logFileSize) {
        logFs->close();
        GetLogFileName(fileName);
        logFs->open(fileName.c_str(), std::ios::app);
    }

    logFs->write(data.c_str(), data.size());
}

void PluginLog::CloseLog()
{
    if (logFs) {
        if (logFs->is_open()) {
            logFs->close();
            logFs = nullptr;
        }
    }
}

void PluginLog::LogWrite(const char *tag, const char *msg)
{
    time_t nowTime = time(nullptr);
    if (nowTime == -1) {
        fprintf(stderr, "%s fail\n", __func__);
    }
    struct tm *t = localtime(&nowTime);
    char buf[30];
    int ret = sprintf(buf, "%4d-%02d-%02d %02d:%02d:%02d ", t->tm_year + BASE_DATE, t->tm_mon + 1, t->tm_mday,
        t->tm_hour, t->tm_min, t->tm_sec);
    if (ret < 0) {
        fprintf(stderr, "%s sprintf fail\n", __func__);
    }
    string stag = tag;
    string smsg = msg;
    string data = buf + stag + smsg;
    LogWriteFile(data);
}

void PluginLog::LogPrint(LogPriority pri, const char *tag, const char *buf)
{
    if (pri <= priority) {
        fprintf(stderr, "%s%s", tag, buf);
    }

    g_mutex.lock();
    LogWrite(tag, buf);
    g_mutex.unlock();
}

void PluginLog::LOGE(const char *fmt, ...)
{
    va_list ap;
    va_start(ap, fmt);
    int ret = vsnprintf(logBuf, LOG_BUF_SIZE, fmt, ap);
    if (ret < 0) {
        fprintf(stderr, "%s vsnprintf fail\n", __func__);
    }
    va_end(ap);
    
    LogPrint(PRIORITY_ERROR, "ERROR:", logBuf);
}

void PluginLog::LOGW(const char *fmt, ...)
{
    va_list ap;
    va_start(ap, fmt);
    int ret = vsnprintf(logBuf, LOG_BUF_SIZE, fmt, ap);
    if (ret < 0) {
        fprintf(stderr, "%s vsnprintf fail\n", __func__);
    }
    va_end(ap);
    LogPrint(PRIORITY_WARN, "WARN:", logBuf);
}

void PluginLog::LOGI(const char *fmt, ...)
{
    va_list ap;
    va_start(ap, fmt);
    int ret = vsnprintf(logBuf, LOG_BUF_SIZE, fmt, ap);
    if (ret < 0) {
        fprintf(stderr, "%s vsnprintf fail\n", __func__);
    }
    va_end(ap);
    LogPrint(PRIORITY_INFO, "INFO:", logBuf);
}

void PluginLog::LOGD(const char *fmt, ...)
{
    va_list ap;
    va_start(ap, fmt);
    int ret = vsnprintf(logBuf, LOG_BUF_SIZE, fmt, ap);
    if (ret < 0) {
        fprintf(stderr, "%s vsnprintf fail\n", __func__);
    }
    va_end(ap);
    LogPrint(PRIORITY_DEBUG, "DEBUG:", logBuf);
}

bool PluginLog::SetPriority(LogPriority pri)
{
    if (pri > PRIORITY_DEBUG) {
        return false;
    }
    priority = pri;
    return true;
}
} // namespace PinLog

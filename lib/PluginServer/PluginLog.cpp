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

#include <cstdarg>
#include <iostream>
#include <ctime>
#include <fstream>
#include <memory>
#include <mutex>
#include <csignal>
#include "PluginServer/PluginLog.h"

namespace PinServer {
using namespace std;
using std::string;
constexpr int LOG_BUF_SIZE = 102400;
constexpr int BASE_DATE = 1900;
static LogPriority g_priority = PRIORITY_WARN; // log打印的级别控制
static std::mutex g_mutex; // 线程锁
static char g_buf[LOG_BUF_SIZE];

shared_ptr<fstream> g_fs;
static void LogWriteInit(const string& data);
static void (*g_writeToLog)(const string& data) = LogWriteInit;

static void GetLogFileName(string& fileName)
{
    time_t nowTime = time(nullptr);
    struct tm *t = localtime(&nowTime);
    char buf[100];
    sprintf(buf, "/tmp/pin_server%d_%4d%02d%02d_%02d_%02d_%02d.log", getppid(),
        t->tm_year + BASE_DATE, t->tm_mon + 1, t->tm_mday, t->tm_hour, t->tm_min, t->tm_sec);
    fileName = buf;
}

static void LogWriteFile(const string& data)
{
    if (g_fs->tellg() > LOG_FILE_SIZE) {
        g_fs->close();
        string fileName;
        GetLogFileName(fileName);
        g_fs->open(fileName.c_str(), ios::app);
    }

    g_fs->write(data.c_str(), data.size());
}

static void LogWriteInit(const string& data)
{
    if (g_writeToLog == LogWriteInit) {
        g_fs = std::make_shared<fstream>();
        string fileName;
        GetLogFileName(fileName);
        g_fs->open(fileName.c_str(), ios::app);
        g_writeToLog = LogWriteFile;
    }
    g_writeToLog(data);
}

void CloseLog(void)
{
    if (g_fs) {
        if (g_fs->is_open()) {
            g_fs->close();
        }
    }
}

static void LogWrite(const char *tag, const char *msg)
{
    time_t nowTime = time(nullptr);
    struct tm *t = localtime(&nowTime);
    char buf[30];
    sprintf(buf, "%4d-%02d-%02d %02d:%02d:%02d ", t->tm_year + BASE_DATE, t->tm_mon + 1, t->tm_mday,
        t->tm_hour, t->tm_min, t->tm_sec);

    string stag = tag;
    string smsg = msg;
    string data = buf + stag + smsg;
    g_writeToLog(data);
}

void LogPrint(LogPriority priority, const char *tag, const char *fmt, ...)
{
    va_list ap;

    va_start(ap, fmt);
    vsnprintf(g_buf, LOG_BUF_SIZE, fmt, ap);
    va_end(ap);

    if (priority <= g_priority) {
        printf("%s%s", tag, g_buf);
    }

    g_mutex.lock();
    LogWrite(tag, g_buf);
    g_mutex.unlock();
}

bool SetLogPriority(LogPriority priority)
{
    if (priority > PRIORITY_DEBUG) {
        return false;
    }
    g_priority = priority;
    return true;
}
} // namespace PinServer

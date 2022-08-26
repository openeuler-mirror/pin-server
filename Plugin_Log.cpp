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
#include "Plugin_Log.h"

namespace Plugin_Server_LOG {
using namespace std;
using std::string;
constexpr int LOG_BUF_SIZE = 10240;
constexpr int BASE_DATE = 1900;

static int g_priority = 0;

std::shared_ptr<fstream> g_fsServer;
std::shared_ptr<fstream> g_fsUser;
static void LogWriteInit(LogId id, const string& data);
static void (*g_writeToLog)(LogId id, const string& data) = LogWriteInit;

static void LogWriteFile(LogId id, const string& data)
{
    if (id == LOG_ID_SERVER) {
        if (g_fsServer->tellg() > LOG_FILE_SIZE) {
            g_fsServer->close();
            time_t nowTime = time(nullptr);
            struct tm *t = localtime(&nowTime);
            char cmd[100];
            sprintf(cmd, "mv server.log server%4d-%02d-%02d_%02d_%02d_%02d.log",
                t->tm_year + BASE_DATE, t->tm_mon, t->tm_mday, t->tm_hour, t->tm_min, t->tm_sec);
            system(cmd);
            g_fsServer->open("server.log", ios::app);
        }
        g_fsServer->write(data.c_str(), data.size());
    } else if (id == LOG_ID_USER) {
        if (g_fsUser->tellg() > LOG_FILE_SIZE) {
            g_fsUser->close();
            time_t nowTime = time(nullptr);
            struct tm *t = localtime(&nowTime);
            char cmd[100];
            sprintf(cmd, "mv user.log user%4d-%02d-%02d_%02d_%02d_%02d.log",
                t->tm_year + BASE_DATE, t->tm_mon, t->tm_mday, t->tm_hour, t->tm_min, t->tm_sec);
            system(cmd);
            g_fsUser->open("user.log", ios::app);
        }
        g_fsUser->write(data.c_str(), data.size());
    }
}

static void LogWriteInit(LogId id, const string& data)
{
    if (g_writeToLog == LogWriteInit) {
        g_fsServer = std::make_shared<fstream>();
        g_fsUser = std::make_shared<fstream>();
        g_fsServer->open("server.log", ios::app);
        g_fsUser->open("user.log", ios::app);
        g_writeToLog = LogWriteFile;
    }
    g_writeToLog(id, data);
}

void CloseLog(void)
{
    g_fsServer->close();
    g_fsUser->close();
}

static void LogWrite(int logId, const char *tag, const char *msg)
{
    time_t nowTime = time(nullptr);
    struct tm *t = localtime(&nowTime);
    char buf[30];
    sprintf(buf, "%4d-%02d-%02d %02d:%02d:%02d ", t->tm_year + BASE_DATE, t->tm_mon, t->tm_mday,
        t->tm_hour, t->tm_min, t->tm_sec);

    string stag = tag;
    string smsg = msg;
    string data = buf + stag + smsg;
    g_writeToLog((LogId)logId, data);
}

void LogPrint(int logId, int priority, const char *tag, const char *fmt, ...)
{
    va_list ap;
    char buf[LOG_BUF_SIZE];

    va_start(ap, fmt);
    vsnprintf(buf, LOG_BUF_SIZE, fmt, ap);
    va_end(ap);

    if (priority <= g_priority) {
        LogWrite(logId, tag, buf);
    }
}

void SetLogPriority(int pri)
{
    g_priority = pri;
}
} // namespace Plugin_Server_LOG

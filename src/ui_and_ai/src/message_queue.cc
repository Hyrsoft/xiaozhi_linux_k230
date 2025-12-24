// SPDX-License-Identifier: GPL-3.0-only
/*
 * Copyright (c) 2025, Canaan Bright Sight Co., Ltd
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */
#include <pthread.h>
#include <mqueue.h>
#include "lv_100ask_xz_ai/src/cJSON.h"
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <fcntl.h>
#include <unistd.h>
#include <mutex>
#include "message_queue.h"

#include <iostream>
#include <thread>
#include "utils.h"
#include "vi_vo.h"
#include "text_paint.h"

#include <time.h>
#include <sys/time.h>

using std::cerr;
using std::cout;
using std::endl;
using std::thread;
using std::string;

#define SEND_QUEUE_NAME "/send_queue"   // 发送队列名称
#define RECV_QUEUE_NAME "/recv_queue"   // 接收队列名称
#define MAX_MSG_SIZE 1024               // 单个消息最大大小（字节）
#define MAX_MSG_COUNT 10                // 队列最大消息数

MessageQueue* MessageQueue::instance = nullptr;
std::mutex MessageQueue::mutex_;
static mqd_t ai_to_ui;
static mqd_t ui_to_ai;

// 定义消息结构体（包含JSON字符串和方向标识）
typedef struct {
    char data[MAX_MSG_SIZE];
} mq_message_t;

// JSON数据生成函数
static char* create_ui_json(ui_cmd_message_t* message) {
    static char empty_str[] = "";
    if (message == NULL) {
        std::cout << "Invalid input parameters." << std::endl;
        return empty_str;
    }
    cJSON *root = cJSON_CreateObject();
    cJSON_AddStringToObject(root, "registered_name", message->registered_name);
    cJSON_AddNumberToObject(root, "command_type", message->cmd);
    char *json_str = cJSON_Print(root);
    cJSON_Delete(root);
    return json_str;
}

static char* create_ai_json(ai_cmd_message_t* message) {
    static char empty_str[] = "";
    if (message == NULL) {
        std::cout << "Invalid input parameters." << std::endl;
        return empty_str;
    }
    cJSON *root = cJSON_CreateObject();
    cJSON_AddStringToObject(root, "error_message", message->error_message);
    cJSON_AddNumberToObject(root, "data", message->data);
    cJSON_AddNumberToObject(root, "success", message->success);
    char *json_str = cJSON_Print(root);
    cJSON_Delete(root);
    return json_str;
}

static bool parse_ai_json(char* message, ai_cmd_message_t* cmd_message){
    char *temp_message;
    if (message == NULL || cmd_message == NULL) {
        std::cout << "Invalid input parameters." << std::endl;
        return false;
    }
    cJSON *root = cJSON_Parse(message);
    if(root == NULL){
        std::cout << "Fail to parse JSON" << std::endl;
        return false;
    }
    cmd_message->data = cJSON_GetObjectItem(root, "data")->valueint;
    cmd_message->success = cJSON_GetObjectItem(root, "success")->valueint;
    temp_message = cJSON_GetObjectItem(root, "error_message")->valuestring;
    if(temp_message != ""){
        size_t len = strlen(temp_message) + 1;
        cmd_message->error_message = new char[len];
        strcpy(cmd_message->error_message, temp_message);
    }
    else{
        cmd_message->error_message = NULL;
    }
    cJSON_Delete(root);
    return true;
}

static bool parse_ui_json(char* message, ui_cmd_message_t* cmd_message){
    char *temp_name;
    if (message == NULL || cmd_message == NULL) {
        std::cout << "Invalid input parameters." << std::endl;
        return false;
    }
    cJSON *root = cJSON_Parse(message);
    if(root == NULL){
        std::cout << "Fail to parse JSON" << std::endl;
        return false;
    }
    cmd_message->cmd = cJSON_GetObjectItem(root, "command_type")->valueint;
    temp_name = cJSON_GetObjectItem(root, "registered_name")->valuestring;
    if(temp_name != ""){
        size_t len = strlen(temp_name) + 1;
        cmd_message->registered_name = new char[len];
        strcpy(cmd_message->registered_name, temp_name);
    }
    cJSON_Delete(root);
    return true;
}

int MessageQueue::SendFaceRecRunCmd(void){
    mq_message_t msg;
    char empty_str[] = "";
    ui_cmd_message_t cmd = {1, empty_str};
    ssize_t read_size = 0;
    ai_cmd_message_t ai_message;
    bool is_success;
    char* json_data = create_ui_json(&cmd);
    if(json_data == "")
        return 0;
    snprintf(msg.data, MAX_MSG_SIZE, "%s", json_data);
    cJSON_free(json_data);
    mq_send(ui_to_ai, msg.data, strlen(msg.data)+1, 0);

    //read_size = mq_receive(ai_to_ui, (char*)&msg, sizeof(msg), NULL);
    //if(read_size <= 0)
   //     return 0;
    //is_success = parse_ai_json(msg.data, &ai_message);
    //if(is_success&&ai_message.success == 1)
    //    return ai_message.data;
    return 0;
}

int MessageQueue::SendRegRegisterCmd(const char* register_name, char*& error_message){
    mq_message_t msg;
    ssize_t read_size = 0;
    bool is_success;
    size_t len;
    char* name_copy = strdup(register_name); 
    ui_cmd_message_t cmd = {5, name_copy};
    ai_cmd_message_t ai_message;
    char* json_data = create_ui_json(&cmd);
    if(json_data == nullptr || strcmp(json_data, "") == 0)
        goto error_handle;
    snprintf(msg.data, MAX_MSG_SIZE, "%s", json_data);
    cJSON_free(json_data);
    if(mq_send(ui_to_ai, msg.data, strlen(msg.data)+1, 0) == -1){
        goto error_handle;
    }
    read_size = mq_receive(ai_to_ui, (char*)&msg, sizeof(msg), NULL);
    if(read_size <= 0)
        goto error_handle;
    is_success = parse_ai_json(msg.data, &ai_message);
    if(is_success&&ai_message.success == 1){
        error_message = nullptr;
        return ai_message.success;
    }
    len = strlen(ai_message.error_message) + 1;
    error_message = new char[len];
    strcpy(error_message, ai_message.error_message);
    delete ai_message.error_message;
    return 0;

error_handle:
    char error_message_tmp[] = "连接错误，请重试！";
    len = strlen(error_message_tmp) + 1;
    error_message = new char[len];
    strcpy(error_message, error_message_tmp);
    return 0;
}

int MessageQueue::SendClearDataBaseCmd(void){
    mq_message_t msg;
    char empty_str[] = "";
    ui_cmd_message_t cmd = {6, empty_str};
    ssize_t read_size = 0;
    ai_cmd_message_t ai_message;
    bool is_success;
    char* json_data = create_ui_json(&cmd);
    if(json_data == "")
        return 0;
    snprintf(msg.data, MAX_MSG_SIZE, "%s", json_data);
    cJSON_free(json_data);
    mq_send(ui_to_ai, msg.data, strlen(msg.data)+1, 0);

    read_size = mq_receive(ai_to_ui, (char*)&msg, sizeof(msg), NULL);
    if(read_size <= 0)
        return 0;
    is_success = parse_ai_json(msg.data, &ai_message);
    if(is_success&&ai_message.success == 1)
        return ai_message.data;
    return 0;
}

void MessageQueue::SendAIMsg(ai_cmd_message_t* message){
    mq_message_t msg;
    char* json_data = create_ai_json(message);
    if(json_data == "")
        return;
    snprintf(msg.data, MAX_MSG_SIZE, "%s", json_data);
    cJSON_free(json_data);
    mq_send(ai_to_ui, msg.data, strlen(msg.data)+1, 0);
    return;
}

int MessageQueue::SendRegRegisterNumCmd(void){
    mq_message_t msg;
    ssize_t read_size = 0;
    char empty_str[] = "";
    ui_cmd_message_t cmd = {2, empty_str};
    ai_cmd_message_t ai_message;
    bool is_success;
    char* json_data = create_ui_json(&cmd);
    if(json_data == "")
        return 0;
    snprintf(msg.data, MAX_MSG_SIZE, "%s", json_data);
    cJSON_free(json_data);
    mq_send(ui_to_ai, msg.data, strlen(msg.data)+1, 0);
    read_size = mq_receive(ai_to_ui, (char*)&msg, sizeof(msg), NULL);
    if(read_size <= 0)
        return 0;
    is_success = parse_ai_json(msg.data, &ai_message);
    if(is_success&&ai_message.success == 1)
        return ai_message.data;
    return 0;
}

int MessageQueue::SendRegImgDropCmd(void){
    mq_message_t msg;
    ssize_t read_size = 0;
    char empty_str[] = "";
    ui_cmd_message_t cmd = {4, empty_str};
    ai_cmd_message_t ai_message;
    bool is_success;
    char* json_data = create_ui_json(&cmd);
    if(json_data == "")
        return 0;
    snprintf(msg.data, MAX_MSG_SIZE, "%s", json_data);
    cJSON_free(json_data);
    mq_send(ui_to_ai, msg.data, strlen(msg.data)+1, 0);
    read_size = mq_receive(ai_to_ui, (char*)&msg, sizeof(msg), NULL);
    if(read_size <= 0)
        return 0;
    is_success = parse_ai_json(msg.data, &ai_message);
    if(is_success&&ai_message.success == 1)
       return 1;
    return 0;
}

int MessageQueue::SendGetRegImgCmd(void){
    mq_message_t msg;
    ssize_t read_size = 0;
    char empty_str[] = "";
    ui_cmd_message_t cmd = {3, empty_str};
    ai_cmd_message_t ai_message;
    bool is_success;
    char* json_data = create_ui_json(&cmd);
    if(json_data == "")
        return 0;
    snprintf(msg.data, MAX_MSG_SIZE, "%s", json_data);
    cJSON_free(json_data);
    mq_send(ui_to_ai, msg.data, strlen(msg.data)+1, 0);
    read_size = mq_receive(ai_to_ui, (char*)&msg, sizeof(msg), NULL);
    if(read_size <= 0)
        return 0;
    is_success = parse_ai_json(msg.data, &ai_message);
    if(is_success&&ai_message.success == 1)
       return 1;
    return 0;
}

bool MessageQueue::AIQueryCmd(ui_cmd_message_t *message){
    mq_message_t msg;
    ssize_t read_size = 0;
    read_size = mq_receive(ui_to_ai, (char*)&msg, sizeof(msg), NULL);
    if(read_size <= 0)
        return false;
    return parse_ui_json(msg.data, message);
}

bool MessageQueue::UIQueryCmd(ai_cmd_message_t *message){
    mq_message_t msg;
    ssize_t read_size = 0;
    read_size = mq_receive(ai_to_ui, (char*)&msg, sizeof(msg), NULL);
    if(read_size <= 0)
        return false;
    return parse_ai_json(msg.data, message);
}

MessageQueue* MessageQueue::getInstance(){
    std::lock_guard<std::mutex> lock(mutex_);
    if (instance == nullptr) {
        instance = new MessageQueue();
    }
    return instance;
}

MessageQueue::MessageQueue() {
    struct mq_attr mq_attr = {
        .mq_flags = 0,
        .mq_maxmsg = MAX_MSG_COUNT,
        .mq_msgsize = MAX_MSG_SIZE
    };

    // 创建双向队列
    ai_to_ui = mq_open(SEND_QUEUE_NAME, O_RDWR | O_CREAT, 0666, &mq_attr);
    ui_to_ai = mq_open(RECV_QUEUE_NAME, O_RDWR | O_CREAT, 0666, &mq_attr);
    if (ai_to_ui == -1 || ui_to_ai == -1) {
        perror("Queue creation failed");
        return;
    }
    return;
}


MessageQueue::~MessageQueue() {
    mq_close(ai_to_ui);
    mq_close(ui_to_ai);
    mq_unlink(SEND_QUEUE_NAME);
    mq_unlink(RECV_QUEUE_NAME);
    return;
}

extern "C"{
    void * createMessageQueue(){
        return MessageQueue::getInstance();
    }

    int callSendFaceRecRunCmd(void* obj){
        MessageQueue* myObj = static_cast<MessageQueue*>(obj);
        return myObj->SendFaceRecRunCmd();
    }

    int callSendRegRegisterCmd(void* obj, const char* register_name, char** error_message){
        MessageQueue* myObj = static_cast<MessageQueue*>(obj);
        return myObj->SendRegRegisterCmd(register_name, *error_message);
    }

    int callSendRegRegisterNumCmd(void* obj){
        MessageQueue* myObj = static_cast<MessageQueue*>(obj);
        return myObj->SendRegRegisterNumCmd();
    }

    int callSendRegImgDropCmd(void* obj){
        MessageQueue* myObj = static_cast<MessageQueue*>(obj);
        return myObj->SendRegImgDropCmd();
    }

    int callSendGetRegImgCmd(void* obj){
        MessageQueue* myObj = static_cast<MessageQueue*>(obj);
        return myObj->SendGetRegImgCmd();
    }

    int callSendClearDataBaseCmd(void* obj){
        MessageQueue* myObj = static_cast<MessageQueue*>(obj);
        return myObj->SendClearDataBaseCmd();
    }
}
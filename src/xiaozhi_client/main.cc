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
#include "control_center.h"
#include "acap.h"
#include <stdio.h>
#include <sys/mman.h>
#include <unistd.h>
#include <fcntl.h>
#include <string.h>
#include <iostream>
#include <pthread.h>
#include <stdbool.h>
#include <stdlib.h>
#include "json.hpp"

#include "acap.h"
#include "cfg.h"
#include "ipc_udp.h"
#include "wake_word_generator.h"
#include "webttsctrl.h"

extern WebttsConfig g_webtts_config;

using json = nlohmann::json;

extern int g_websocket_start;
static WakeupMethodMode g_wakeup_method = kWakeupMethodModeUnknown; // 默认值
extern int g_audio_upload_enable;
SpeechInteractionMode g_speech_interaction_mode = kSpeechInteractionModeAuto;
int g_wakeup_word_start = 0;

p_ipc_endpoint_t g_ipc_wakeup_detect_audio_ep;
p_ipc_endpoint_t g_ipc_wakeup_detect_control_ep;

#define VIDEO_WAKEUP_KEYWORD "你好小凡，"
static std::string g_video_wakeup_person_name;

static void audio_data_upload_cb(const unsigned char* data, int length)
{
    send_audio(data,length,NULL);
}

static void wakeup_data_upload_cb(const unsigned char* data, int length)
{
    send_audio(data,length,NULL);
}

static void generate_speech_cb(const char* pcm_data, size_t data_size)
{
    send_audio((const unsigned char*)pcm_data,data_size,NULL);
}

static void audio_data_download_cb(const unsigned char* data, int length)
{
    //printf("==========audio_data_download_cb===========:%d\n",length);
    play_opus_stream(data, length);
}

static void audio_data_tts_state_cb(int tts_state)
{
    //printf("========audio tts state===========:%d\n",tts_state);
    //set_tts_state(tts_state);
}

static  void ws_work_state_cb(bool work_state)
{
    if(!work_state)
    {
        json j;
        j["type"] = "wake-up";
        j["status"] = "stop";
        j["wake-up_method"] = g_wakeup_method == kWakeupMethodModeVoice ? "voice" : "video";
        std::string wakeupString = j.dump();

        g_ipc_wakeup_detect_control_ep->send(g_ipc_wakeup_detect_control_ep, wakeupString.data(), wakeupString.size());

    }

}

//解析命令参数
static int parse_command_args(int argc, char **argv)
{
    if (argc > 1) {
        std::string mode_arg = argv[1];
        if (mode_arg == "auto") {
            g_speech_interaction_mode = kSpeechInteractionModeAuto;
        } else if (mode_arg == "manual") {
            g_speech_interaction_mode = kSpeechInteractionModeManual;
        } else if (mode_arg == "realtime") {
            g_speech_interaction_mode = kSpeechInteractionModeRealtime;
        } else if (mode_arg == "wakeup") {
            g_speech_interaction_mode = kSpeechInteractionModeAutoWithWakeupWord;
        } else {
            std::cerr << "Invalid interaction mode: " << mode_arg << std::endl;
            return -1;
        }
        std::cout << "Speech interaction mode set to: " << mode_arg << std::endl;
    } else {
        std::cerr << "Usage: xiaozhi_client [auto|manual|realtime|wakeup]" << std::endl;
        std::cerr << "  auto: Automatic speech interaction mode" << std::endl;
        std::cerr << "  manual: Manual speech interaction mode" << std::endl;
        std::cerr << "  realtime: Real-time speech interaction mode" << std::endl;
        std::cerr << "  wakeup: Automatic speech interaction mode with wakeup word" << std::endl;
        return -1;
    }

    return 0;
}


static int process_wakeup_word_audio_info(char *buffer, size_t size, void *user_data)
{

    return 0;
}

static int _do_wakeup_word_detected()
{
    //设置唤醒词开始标志
    g_wakeup_word_start = 1;

    //唤醒小智
    if (!g_websocket_start)
    {
        connect_to_xiaozhi_server();
        usleep(1 * 1000 * 500);
    }
    else
    {
        //终止当前对话
        g_audio_upload_enable = 0;
        usleep(1000*500);
        abort_cur_session();
        usleep(1000*500);
        g_audio_upload_enable = 1;
    }


    //发送唤醒词
    if (g_wakeup_method == kWakeupMethodModeVoice)
    {
        //发送唤醒词
        char cwd[1024];
        if (getcwd(cwd, sizeof(cwd)) != NULL) {
            std::string filePath = std::string(cwd) + "/wakeup_audio.pcm";
            printf("Wakeup word file path: %s\n", filePath.c_str());
            //发送唤醒词
            wake_word_file(filePath.c_str());
        } else {
            perror("getcwd() error");
            return -1;
        }
    }
    else if (g_wakeup_method == kWakeupMethodModeVideo)
    {
        //发送唤醒词
        //wake_word_file("/usr/bin/wakeup_video.pcm");
        string wakeup_text = VIDEO_WAKEUP_KEYWORD + g_video_wakeup_person_name;
        webtts_send_text(wakeup_text.c_str());
    }
    else
    {
        printf("wakeup method is unknown\n");
        return -1;
    }

    usleep(1000 * 500);

    g_wakeup_word_start = 0;

    return 0;
}

static int process_wakeup_word_control_info(char *buffer, size_t size, void *user_data)
{
    printf("Received wake-up message:%s\n",buffer);
    if (g_speech_interaction_mode != kSpeechInteractionModeAutoWithWakeupWord)
    {
        printf("current mode is not kSpeechInteractionModeAutoWithWakeupWord\n");
        return 0;
    }

    json j = json::parse(buffer);
    if (!j.contains("type"))
        return 0;

    if (j["type"] == "wake-up")
    {
        if (!j.contains("wake-up_method"))
            return 0;
        if (!j.contains("status"))
            return 0;

        //开始唤醒
        if (j["status"] == "start")
        {
            std::string wakeup_method = j["wake-up_method"];
            if (wakeup_method == "voice")
            {
                g_wakeup_method = kWakeupMethodModeVoice;
            }
            else if (wakeup_method == "video")
            {
                g_wakeup_method = kWakeupMethodModeVideo;
                if (!j.contains("wake-up_text"))
                {
                    printf("wakeup method is video, but no wakeup text\n");
                    return 0;
                }

                g_video_wakeup_person_name = j["wake-up_text"];
                //printf("wakeup method is video, wakeup text is %s\n", g_video_wakeup_person_name.c_str());

            }
            else
            {
                printf("wakeup method is unknown\n");
                return 0;
            }

            _do_wakeup_word_detected();
        }

    }

    return 0;
}

static void init_wakeup_detect()
{
    //语音，视频唤醒词检测
    g_ipc_wakeup_detect_audio_ep = ipc_endpoint_create_udp(WAKEUP_WORD_DETECTION_AUDIO_PORT_DOWN, WAKEUP_WORD_DETECTION_AUDIO_PORT_UP, process_wakeup_word_audio_info, NULL);
    //唤醒词检测控制
    g_ipc_wakeup_detect_control_ep = ipc_endpoint_create_udp(WAKEUP_WORD_DETECTION_CONTROL_PORT_DOWN, WAKEUP_WORD_DETECTION_CONTROL_PORT_UP, process_wakeup_word_control_info, NULL);

    //本地文件语音合成检测
    init_wake_word_generator(wakeup_data_upload_cb);

    g_webtts_config.audio_callback = generate_speech_cb;

    int ret2 = webtts_init(&g_webtts_config);
    if (ret2 != 0) {
        std::cerr << "Failed to initialize WebTTS" << std::endl;
        return ;
    }

}

static int _test_wakeup()
{
    while(1)
    {
        printf("Press Enter to continue...\n");
        while (getchar() != '\n'); // 等待用户按下回车键

        static int count = 0;
        count ++;
        printf("[%d]========test wakeup word=========\n",count);
        if (count % 2 == 0)
            g_wakeup_method = kWakeupMethodModeVideo;
        else
            g_wakeup_method = kWakeupMethodModeVoice;

        _do_wakeup_word_detected();
    }
    return 0;
}




int main(int argc, char **argv)
{
    int ret = 0;
    //解析命令参数
    ret = parse_command_args(argc, argv);
    if (ret != 0)
    {
        return -1;
    }

    //初始化设备
    ret = init_device(audio_data_download_cb,audio_data_tts_state_cb,ws_work_state_cb);
    if (ret != 0)
    {
        printf("init_device failed\n");
        return -1;
    }

    //激活设备
    ret = active_device();
    if (ret != 0)
    {
        printf("active_device failed\n");
        return -1;
    }

    //初始化唤醒词检测
    init_wakeup_detect();

    //初始化音频
    acap_init(audio_data_upload_cb);

    //启动音频
    acap_start();

    //连接到小智服务器
    if (g_speech_interaction_mode != kSpeechInteractionModeAutoWithWakeupWord) {
        connect_to_xiaozhi_server();
    }
    else
    {
        //测试唤醒词
        // printf("=========test wakeup\n");
        // _test_wakeup();

    }

    while (1)
    {
        sleep(1);
    }

    acap_stop();
    acap_deinit();

    return 0;
}
// SPDX-License-Identifier: GPL-3.0-only
/*
 * Copyright (c) 2025, Canaan Bright Sight Co., Ltd
 */

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <mutex>
#include <string>
#include <sys/time.h>
#include <thread>
#include <unordered_map>
#include <vector>
#include <unistd.h>

#include "cfg.h"
#include "face_detection.h"
#include "face_emotion.h"
#include "ipc_udp.h"
#include "json.hpp"
#include "mmz.h"
#include "sensor_buf_manager.h"
#include "setting.h"
#include "utils.h"
#include "vi_vo.h"

using json = nlohmann::json;
using std::cerr;
using std::cout;
using std::endl;
using std::string;
using std::thread;
using std::unordered_map;
using std::vector;

static std::mutex result_mutex;
static vector<FaceEmotionInfo> face_emotion_results;
static vector<FaceDetectionInfo> face_det_results;
static vector<FaceEmotionInfo> cached_emotion_results; // reuse recent emotion results when skipping frames
static constexpr unsigned BUFFER_NUM = 3;
static constexpr int kEmotionCooldownMs = 3000;
static constexpr int kEmotionIntervalMs = 1000; // run emotion at most once per second
std::atomic<bool> ai_stop(false);
std::atomic<bool> display_stop(false);
static volatile unsigned kpu_frame_count = 0;
static struct timeval tv, tv2;
struct display *display = nullptr;
struct display_buffer *draw_buffer = nullptr;

p_ipc_endpoint_t g_ipc_wakeup_detect_control_ep = nullptr;

static std::atomic<long long> g_last_emotion_trigger_ms{0};

static const unordered_map<string, string> kEmotionLabelZh = {
    {"Anger", "生气"},
    {"Disgust", "厌恶"},
    {"Fear", "恐惧"},
    {"Happiness", "高兴"},
    {"Neutral", "平静"},
    {"Sadness", "伤心"},
    {"Surprise", "惊讶"},
};

static long long now_ms()
{
    return std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

void print_usage(const char *name)
{
    cout << "Usage: " << name << " <kmodel_det> <obj_thres> <nms_thres> <kmodel_femo> <input_mode> <debug_mode>" << endl
         << "Options:" << endl
         << "  kmodel_det               人脸检测kmodel路径\n"
         << "  obj_thres                人脸检测阈值\n"
         << "  nms_thres                人脸检测nms阈值\n"
         << "  kmodel_femo              人脸情感识别kmodel路径\n"
         << "  input_mode               输入模式（仅支持None摄像头输入）\n"
         << "  debug_mode               调试开关\n"
         << endl;
}

void ai_proc(char *argv[], int video_device)
{
    struct v4l2_drm_context context;
    v4l2_drm_default_context(&context);
    context.device = video_device;
    context.display = false;
    context.width = SENSOR_WIDTH;
    context.height = SENSOR_HEIGHT;
    context.video_format = v4l2_fourcc('B', 'G', '3', 'P'); // BGR planar
    context.buffer_num = BUFFER_NUM;

    if (v4l2_drm_setup(&context, 1, NULL)) {
        cerr << "v4l2_drm_setup error" << endl;
        return;
    }
    if (v4l2_drm_start(&context)) {
        cerr << "v4l2_drm_start error" << endl;
        return;
    }

    FaceDetection face_det(argv[1], atof(argv[2]), atof(argv[3]), {SENSOR_CHANNEL, SENSOR_HEIGHT, SENSOR_WIDTH}, atoi(argv[6]));
    FaceEmotion face_emo(argv[4], {SENSOR_CHANNEL, SENSOR_HEIGHT, SENSOR_WIDTH}, atoi(argv[6]));

    std::vector<std::tuple<int, void *>> tensors;
    for (unsigned i = 0; i < BUFFER_NUM; i++) {
        tensors.push_back({context.buffers[i].fd, context.buffers[i].mmap});
    }
    SensorBufManager sensor_buf({SENSOR_CHANNEL, SENSOR_HEIGHT, SENSOR_WIDTH}, tensors);

    long long last_emotion_infer_ms = 0;
    size_t frame_count = 0;
    while (!ai_stop) {
        long long loop_now = now_ms();
        bool run_emotion_this_frame = (loop_now - last_emotion_infer_ms) >= kEmotionIntervalMs;
        int ret = v4l2_drm_dump(&context, 1000);
        if (ret) {
            perror("v4l2_drm_dump error");
            continue;
        }
        runtime_tensor &img_data = sensor_buf.get_buf_for_index(context.vbuffer.index);
        face_det.pre_process(img_data);
        face_det.inference();

        result_mutex.lock();
        face_det_results.clear();
        face_det.post_process({SENSOR_WIDTH, SENSOR_HEIGHT}, face_det_results);

        face_emotion_results.clear();
        if (run_emotion_this_frame) {
            cached_emotion_results.clear();
        }
        bool need_wakeup = false;
        string wakeup_text;
        long long now = now_ms();

        for (size_t i = 0; i < face_det_results.size(); ++i) {
            FaceEmotionInfo emo_result{};
            if (run_emotion_this_frame) {
                face_emo.pre_process(img_data, face_det_results[i].sparse_kps.points);
                face_emo.inference();
                face_emo.post_process(emo_result);
                cached_emotion_results.push_back(emo_result);
                last_emotion_infer_ms = now;
            }
            else {
                if (i < cached_emotion_results.size()) {
                    emo_result = cached_emotion_results[i];
                }
                else {
                    emo_result = FaceEmotionInfo{0, 0.0f, "Neutral"};
                    cached_emotion_results.push_back(emo_result);
                }
            }

            face_emotion_results.push_back(emo_result);

            if (run_emotion_this_frame && !need_wakeup && emo_result.label != "Neutral" && now - g_last_emotion_trigger_ms.load() >= kEmotionCooldownMs) {
                auto zh = kEmotionLabelZh.find(emo_result.label);
                string zh_label = (zh != kEmotionLabelZh.end()) ? zh->second : emo_result.label;
                wakeup_text = "我当前情绪是" + zh_label;
                need_wakeup = true;
                g_last_emotion_trigger_ms.store(now);
            }
        }
        if (run_emotion_this_frame) {
            last_emotion_infer_ms = now;
        }
        result_mutex.unlock();

        if (need_wakeup && g_ipc_wakeup_detect_control_ep) {
            json j;
            j["type"] = "wake-up";
            j["status"] = "start";
            j["wake-up_method"] = "video";
            j["wake-up_text"] = wakeup_text;
            std::string textString = j.dump();
            g_ipc_wakeup_detect_control_ep->send(g_ipc_wakeup_detect_control_ep, textString.data(), textString.size());
        }

        kpu_frame_count += 1;
        frame_count += 1;
        v4l2_drm_dump_release(&context);
    }
    v4l2_drm_stop(&context);
}

int frame_handler(struct v4l2_drm_context *context, bool displayed)
{
    static unsigned response = 0, display_frame_count = 0;
    response += 1;

    if (displayed && context[0].buffer_hold[context[0].wp] >= 0) {
        static struct display_buffer *last_drawed_buffer = nullptr;
        auto buffer = context[0].display_buffers[context[0].buffer_hold[context[0].wp]];
        if (buffer != last_drawed_buffer) {
            if (draw_buffer->width > draw_buffer->height) {
                cv::Mat temp_img(draw_buffer->height, draw_buffer->width, CV_8UC4);
                temp_img.setTo(cv::Scalar(0, 0, 0, 0));
                result_mutex.lock();
                for (size_t i = 0; i < face_det_results.size(); ++i) {
                    FaceEmotion::draw_result_video(temp_img, face_det_results[i].bbox, face_emotion_results[i]);
                }
                result_mutex.unlock();
                memcpy(draw_buffer->map, temp_img.data, draw_buffer->size);
            }
            else {
                cv::Mat temp_img(draw_buffer->width, draw_buffer->height, CV_8UC4);
                temp_img.setTo(cv::Scalar(0, 0, 0, 0));
                result_mutex.lock();
                for (size_t i = 0; i < face_det_results.size(); ++i) {
                    FaceEmotion::draw_result_video(temp_img, face_det_results[i].bbox, face_emotion_results[i]);
                }
                result_mutex.unlock();
                cv::rotate(temp_img, temp_img, cv::ROTATE_90_CLOCKWISE);
                memcpy(draw_buffer->map, temp_img.data, draw_buffer->size);
            }
            last_drawed_buffer = buffer;
            thead_csi_dcache_clean_invalid_range(buffer->map, buffer->size);
            display_update_buffer(draw_buffer, 0, 0);
        }
        display_frame_count += 1;
    }

    gettimeofday(&tv2, NULL);
    uint64_t duration = 1000000 * (tv2.tv_sec - tv.tv_sec) + tv2.tv_usec - tv.tv_usec;
    if (duration >= 1000000) {
        fprintf(stderr, " poll: %.2f, ", response * 1000000. / duration);
        response = 0;
        if (display) {
            fprintf(stderr, "display: %.2f, ", display_frame_count * 1000000. / duration);
            display_frame_count = 0;
        }
        fprintf(stderr, "camera: %.2f, ", context[0].frame_count * 1000000. / duration);
        context[0].frame_count = 0;
        fprintf(stderr, "KPU: %.2f", kpu_frame_count * 1000000. / duration);
        kpu_frame_count = 0;
        fprintf(stderr, "          \r");
        fflush(stderr);
        gettimeofday(&tv, NULL);
    }

    if (display_stop) {
        return 'q';
    }
    return 0;
}

void display_proc(int video_device)
{
    struct v4l2_drm_context context;
    v4l2_drm_default_context(&context);
    context.device = video_device;
    if (display->width > display->height) {
        context.width = display->width;
        context.height = (display->width * SENSOR_HEIGHT / SENSOR_WIDTH) & 0xfff8;
        context.video_format = V4L2_PIX_FMT_NV12;
        context.display_format = 0;
        context.drm_rotation = rotation_0;
    }
    else {
        context.width = display->height;
        context.height = display->width;
        context.video_format = V4L2_PIX_FMT_NV12;
        context.display_format = 0;
        context.drm_rotation = rotation_90;
    }

    if (v4l2_drm_setup(&context, 1, &display)) {
        std::cerr << "v4l2_drm_setup error" << std::endl;
        return;
    }

    struct display_plane *plane = display_get_plane(display, DRM_FORMAT_ARGB8888);
    draw_buffer = display_allocate_buffer(plane, display->width, display->height);
    display_commit_buffer(draw_buffer, 0, 0);

    gettimeofday(&tv, NULL);
    v4l2_drm_run(&context, 1, frame_handler);

    if (display) {
        display_free_plane(plane);
        display_exit(display);
    }
}

void __attribute__((destructor)) cleanup()
{
    std::cout << "Cleaning up memory..." << std::endl;
    shrink_memory_pool();
    kd_mpi_mmz_deinit();
}

int main(int argc, char *argv[])
{
    std::cout << "case " << argv[0] << " built at " << __DATE__ << " " << __TIME__ << std::endl;
    if (argc != 7) {
        print_usage(argv[0]);
        return -1;
    }

    if (strcmp(argv[5], "None") != 0) {
        std::cerr << "Image input mode is not supported in this build." << std::endl;
        return -1;
    }

    display = display_init(0);
    if (!display) {
        cerr << "display_init error, exit" << endl;
        return -1;
    }

    g_ipc_wakeup_detect_control_ep = ipc_endpoint_create_udp(WAKEUP_WORD_DETECTION_CONTROL_PORT_UP, WAKEUP_WORD_DETECTION_CONTROL_PORT_DOWN, NULL, NULL);

    std::thread ai_thread(ai_proc, argv, 2);
    std::thread display_thread(display_proc, 1);

    std::cout << "输入 'q'回车退出" << std::endl;

    string input;
    while (std::getline(std::cin, input)) {
        if (input == "q") {
            display_stop.store(true);
            usleep(100000);
            ai_stop.store(true);
            break;
        }
        usleep(100000);
    }

    display_thread.join();
    ai_thread.join();

    return 0;
}

/* Copyright (c) 2025, Canaan Bright Sight Co., Ltd
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 * 1. Redistributions of source code must retain the above copyright
 * notice, this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright
 * notice, this list of conditions and the following disclaimer in the
 * documentation and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND
 * CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES,
 * INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
 * MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 * SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 * BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
 * WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
 * NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */
#ifndef _FACE_EMOTION_H
#define _FACE_EMOTION_H

#include <vector>
#include "utils.h"
#include "ai_base.h"

using std::vector;

typedef struct FaceEmotionInfo
{
    int idx;                     //人脸情感识别结果对应类别id
    float score;                 //人脸情感识别结果对应类别得分
    string label;                //人脸情感识别结果对应类别
} FaceEmotionInfo;

/**
 * @brief 人脸情感识别
 * 主要封装了对于每一帧图片，从预处理、运行到后处理给出结果的过程
 */
class FaceEmotion : public AIBase
{
public:
    FaceEmotion(const char *kmodel_file, const int debug_mode);
    FaceEmotion(const char *kmodel_file, FrameCHWSize isp_shape, const int debug_mode);
    ~FaceEmotion();

    void pre_process(cv::Mat ori_img, float* sparse_points);
    void pre_process(runtime_tensor& img_data, float* sparse_points);
    void inference();
    void post_process(FaceEmotionInfo& result);
    void draw_result(cv::Mat& src_img,Bbox& bbox,FaceEmotionInfo& result, bool pic_mode=true);
    static void draw_result_video(cv::Mat& src_img,Bbox& bbox,FaceEmotionInfo& result);
private:
    void svd22(const float a[4], float u[4], float s[2], float v[4]);
    void image_umeyama_224(float* src, float* dst);
    void get_affine_matrix(float* sparse_points);
    void softmax(vector<float>& input,vector<float>& output);

    std::unique_ptr<ai2d_builder> ai2d_builder_; // ai2d构建器
    runtime_tensor ai2d_in_tensor_;              // ai2d输入tensor
    runtime_tensor ai2d_out_tensor_;             // ai2d输出tensor
    FrameCHWSize isp_shape_;                     // isp对应的地址大小
    float matrix_dst_[10];                       // 人脸affine的变换矩阵
    vector<string> label_list_;                   // 情感分类标签列表
};
#endif

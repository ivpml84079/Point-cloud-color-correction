#include <iostream>
#include <omp.h>
#include <iomanip>
#include <filesystem>
#include <cmath>
#include <Eigen/Eigen>
#include <Eigen\Dense>
#include <fstream>
#include <limits>
#include <vector>
#include <string> 
#include <chrono>
#include <stdio.h>
#include <cstdio>
#include <stdlib.h>
#include <pcl/common/common_headers.h>
#include <pcl/common/colors.h>
#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/point_types_conversion.h>
#include <pcl/registration/icp.h>
#include <pcl/registration/transformation_estimation_svd.h>
#include <pcl/features/shot.h>
#include <pcl/features/pfh.h>
#include <pcl/features/fpfh.h>
#include <pcl/features/normal_3d.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/ModelCoefficients.h>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace cv;
using namespace Eigen;
using namespace Eigen::internal;
using namespace Eigen::Architecture;
using namespace std;
using namespace pcl;
using namespace registration;
namespace fs = std::filesystem;

void saveMatrix(int** matrix, const string& filename){
    ofstream outputFile(filename);
    if (outputFile.is_open()) {
        for (int ch = 0; ch < 3; ch++) {            
            for (int i = 0; i < 256; i++) {
                outputFile << "[" << ch << "][" << i << "] = " << matrix[ch][i] << endl;
            }
            for (int i = 0; i < 256; i++) {
                outputFile << matrix[ch][i] << endl;
            }
            outputFile << endl;
        }
        outputFile.close();
        //cout << "Array data saved to " << filename << std::endl;
    }
    else {
        cout << "Unable to open the file " << filename << std::endl;
    }
}

void drawLineChart(const int* data, int sizeY, cv::Scalar color, cv::Mat& chart) {
    int height = chart.rows;
    int width = chart.cols;
    double x_scale = (double)width / 256;
    double y_scale = (double)1 / sizeY;

    for (int i = 0; i < 255; i++) {
        cv::Point p1(i * x_scale, height * (1 - data[i] * y_scale));
        cv::Point p2((i + 1) * x_scale, height * (1 - data[i + 1] * y_scale));
        if ((p1.y != height) && (p2.y != height))
            cv::line(chart, p1, p2, color, 1, cv::LINE_AA);
    }
}

void drawHistogram(int** tar_matrix, int** src_matrix, const string& path) {
    // 创建一个白色背景的图像，作为绘制折线图的画布
    int chart_width = 800;
    int chart_height = 600;
    cv::Mat chart[3];
    for (int ch = 0; ch < 3; ch++) {
        chart[ch] = cv::Mat(chart_height, chart_width, CV_8UC3, cv::Scalar(255, 255, 255));
        // 绘制目标累积直方图的折线图，使用蓝色表示
        cv::Scalar tar_color(255, 0, 0); // 蓝色
        drawLineChart(tar_matrix[ch], tar_matrix[ch][255], tar_color, chart[ch]);

        // 绘制源累积直方图的折线图，使用红色表示
        cv::Scalar src_color(0, 0, 255); // 红色
        drawLineChart(src_matrix[ch], src_matrix[ch][255], src_color, chart[ch]);
    }    
    imwrite(path + "/HM" + "/cul_graph_0.jpg", chart[0]);
    imwrite(path + "/HM" + "/cul_graph_1.jpg", chart[1]);
    imwrite(path + "/HM" + "/cul_graph_2.jpg", chart[2]);
}

void drawHistogram(int** function, const string& path, const string& name) {
    // 创建一个白色背景的图像，作为绘制折线图的画布
    int chart_width = 800;
    int chart_height = 600;
    cv::Mat chart[3];
    for (int ch = 0; ch < 3; ch++) {
        chart[ch] = cv::Mat(chart_height, chart_width, CV_8UC3, cv::Scalar(255, 255, 255));
        // 绘制function的折线图，使用綠色表示
        cv::Scalar func_color(0, 255, 0); // 綠色
        drawLineChart(function[ch], 255, func_color, chart[ch]);        
    }
    imwrite(path + "/CMF_" + name + "_0.jpg", chart[0]);
    imwrite(path + "/CMF_" + name + "_1.jpg", chart[1]);
    imwrite(path + "/CMF_" + name + "_2.jpg", chart[2]);
}

void gaussian_color_change(int k, int* init, int* blend, long double* k_dist, int* color_r, int* color_g, int* color_b) {
    int print = 0;
    if (k > 1) {
        for (int ch = 0; ch < 3; ch++) {
            // 計算 variance_d(距離之 variance), variance_c(色彩差異之 variance) for weight
            long double variance_d = 0.0, variance_c = 0.0, avg_d = 0.0, avg_c = 0.0;
            int color = 0;
            // 對k個鄰近匹配點計算所需之 Avg_d, Avg_c
            for (int i = 0; i < k; i++) {
                color = ch == 0 ? color_r[i] : ch == 1 ? color_g[i] : color_b[i];
                avg_d += k_dist[i];
                avg_c += abs(init[ch] - color);
            }
            avg_d /= k;
            avg_c /= k;
            //if (print) cout << "avg_d = " << avg_d << "\t avg_c = " << avg_c << endl;
            //variance_c = 0.5;  // test : 0.1, 0.5, 0.9, 3.0 越大對顏色差異越不敏感
            // 計算 variance 
            for (int i = 0; i < 5; i++) {
                color = ch == 0 ? (abs(init[0] - color_r[i])) : ch == 1 ? (abs(init[1] - color_g[i])) : (abs(init[2] - color_b[i]));
                variance_d += pow((k_dist[i] - avg_d), 2);
                variance_c += pow((color - avg_c), 2);
            }
            variance_d /= k;
            //variance_c /= k;
            //variance_d = 0.5;
            variance_c = avg_c * avg_c;
            if (print) cout << " variance_d = " << variance_d << "\t variance_c = " << variance_c << endl;
            // 計算 weight 
            double* weight = new double[k] {};
            double* weight_c = new double[k] {};
            double* weight_d = new double[k] {};
            for (int i = 0; i < k; i++) {
                color = ch == 0 ? (abs(init[0] - color_r[i])) : ch == 1 ? (abs(init[1] - color_g[i])) : (abs(init[2] - color_b[i]));
                weight_d[i] = exp((double)(-1) * pow(k_dist[i], 2) / (variance_d * 1));
                weight_c[i] = exp((double)(-1) * pow(color, 2) / (variance_c * 1));
                weight[i] = weight_c[i] * weight_d[i];
                if (print) {
                    cout << "\n[BEFORE]" << "\n---------------------" << i << endl;
                    cout << "dist_dif = " << k_dist[i] << "\t color_dif = " << color << endl;
                    cout << " weight_d = " << weight_d[i]
                        << "\t weight_c = " << weight_c[i]
                        << "\n weight = " << weight[i] << "\n---------------------" << i << endl;
                }
            }
            bool allZero = true;
            for (int i = 0; i < k; i++) {
                if (weight_c[i] != 0.0) {
                    allZero = false;
                    break;
                }
            }
            if (allZero) {
                double val = 1.0 / k;
                for (int i = 0; i < k; i++) {
                    weight_c[i] = val;
                }
            }
            allZero = true;
            for (int i = 0; i < k; i++) {
                if (weight_d[i] != 0.0) {
                    allZero = false;
                    break;
                }
            }
            if (allZero) {
                double val = 1.0 / k;
                for (int i = 0; i < k; i++) {
                    weight_d[i] = val;
                }
            }
            for (int i = 0; i < k; i++) {
                weight[i] = weight_c[i] * weight_d[i];
            }
            for (int i = 0; i < k; i++) {
                if (print) {
                    cout << "\n[AFTER]" << "\n---------------------" << i << endl;
                    cout << " weight_d = " << weight_d[i]
                        << "\t weight_c = " << weight_c[i]
                        << "\n weight = " << weight[i] << "\n---------------------" << i << endl;
                }
            }

            // 執行調色
            // 計算weight加總
            double total_weight = 0.0;
            double temp = 0.0;
            for (int i = 0; i < k; i++)
                total_weight += weight[i];
            // 執行 k 個鄰居的 joint bilateral interpolation
            for (int i = 0; i < k; i++) {
                //color = ch == 0 ? color_r[i] : ch == 1 ? color_g[i] : color_b[i];
                color = ch == 0 ? color_r[i] - init[0] : ch == 1 ? color_g[i] - init[1] : color_b[i] - init[2];
                temp += ((double)1 / total_weight) * weight[i] * color;
                if (print) cout << "color[" << i << "] = " << color << "\t weight = " << weight[i] << endl;
            }
            blend[ch] = static_cast<int>(std::round(temp));
        }
    }
    else if(k == 1){
        blend[0] = color_r[0] - init[0];
        blend[1] = color_g[0] - init[1];
        blend[2] = color_b[0] - init[2];
    }
    else if(k == 0){
        blend[0] = 256;
        blend[1] = 256;
        blend[2] = 256;
    }
}

double CD_color(int* check_list, int* correspond, pcl::PointCloud<pcl::PointXYZRGB>::Ptr source, pcl::PointCloud<pcl::PointXYZRGB>::Ptr target) {
    cout << "\n[CD] Calculate" << endl;
    double CD[3] = { 0 };
    for (int ch = 0; ch < 3; ch++) {
        int src_hist[256] = { 0 };
        int tar_hist[256] = { 0 };
        int interCount = 0;
        // 計算 source 直方圖
        for (int i = 0; i < source->size(); i++) {
            for (int j = 0; j < target->size(); j++) {
                if (correspond[j] == i && check_list[j] == 1)
                    switch (ch) {
                        case 0:
                            src_hist[source->points[i].r]++;
                            break;
                        case 1:
                            src_hist[source->points[i].g]++;
                            break;
                        case 2:
                            src_hist[source->points[i].b]++;
                            break;
                    }
            }
        }
        for (int i = 0; i < 256; i++) {
            //cout << ch << " src[ " << i << "] = " << src_hist[i] << endl;
        }
        // 計算 target 直方圖 & interCount 數量
        for (int i = 0; i < target->size(); i++) {
            if (check_list[i] == 1) {
                switch (ch) {
                    case 0:
                        tar_hist[target->points[i].r]++;
                        break;
                    case 1:
                        tar_hist[target->points[i].g]++;
                        break;
                    case 2:
                        tar_hist[target->points[i].b]++;
                        break;
                    }
                interCount++;
            }
        }          
        for (int i = 0; i < 256; i++) {
            //cout << ch << " tar[ " << i << "] = " << tar_hist[i] << endl;
        }
        // 計算 CD 
        for (int i = 0; i < 256; i++) {            
            CD[ch] += ((double)tar_hist[i] / interCount) * ((double)abs(tar_hist[i] - src_hist[i]) / 256);            
        }
        cout << "CD[" << ch << "] = " << CD[ch] << endl;
    }
    return (CD[0] + CD[1] + CD[2]) / 3;
}

void mean_cal(pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud, float* mean) {
    
    for (int ch = 0; ch < 3; ch++) {
        float temp = 0.0;
        // 計算 cloud 各 point r/g/b 求 mean
        for (int i = 0; i < cloud->size(); i++) {
            switch (ch) {
            case 0:
                temp += cloud->points[i].r;
                break;
            case 1:
                temp += cloud->points[i].g;
                break;
            case 2:
                temp += cloud->points[i].b;
                break;
            }
        }
        temp /= cloud->size();
        mean[ch] = temp;        
    }
}

double* Mean_Varience_color(pcl::PointCloud<pcl::PointXYZRGB>::Ptr source, pcl::PointCloud<pcl::PointXYZRGB>::Ptr target) {
    cout << "\n[Mean] Calculate" << endl;
    //double MV[3][2] = {0}; [ch][mean/var]
    double* MV = (double*)malloc(4 * 2 * sizeof(double));
    for (int ch = 0; ch < 3; ch++) {
        double src_avg = 0;
        double tar_avg = 0;
        // 計算 source 各 point r/g/b 求平均
        for (int i = 0; i < source->size(); i++) {
            switch (ch) {
                case 0:
                    src_avg += source->points[i].r;
                    break;
                case 1:
                    src_avg += source->points[i].g;
                    break;
                case 2:
                    src_avg += source->points[i].b;
                    break;
            }
        }
        src_avg /= source->size();
        // 計算 target 各 point r/g/b 求平均
        for (int i = 0; i < target->size(); i++) {
            switch (ch) {
                case 0:
                    tar_avg += target->points[i].r;
                    break;
                case 1:
                    tar_avg += target->points[i].g;
                    break;
                case 2:
                    tar_avg += target->points[i].b;
                    break;
            }
        }
        tar_avg /= target->size();
        //MV[ch][0] = abs(((double)src_sum / source->size()) - ((double)tar_sum / target->size()));
        *(MV + ch * 2 + 0) = abs(src_avg - tar_avg);
        //cout << "Mean_dif[" << ch << "] = " << *(MV + ch * 2 + 0) << endl;

        double src_var = 0;
        double tar_var = 0;
        for (int i = 0; i < source->size(); i++) {
            switch (ch) {
                case 0:
                    src_var += pow((double)source->points[i].r - src_avg, 2) / source->size();
                    break;
                case 1:
                    src_var += pow((double)source->points[i].g - src_avg, 2) / source->size();
                    break;
                case 2:
                    src_var += pow((double)source->points[i].b - src_avg, 2) / source->size();
                    break;
            }            
        }
        for (int i = 0; i < target->size(); i++) {
            switch (ch) {
                case 0:
                    tar_var += pow((double)target->points[i].r - tar_avg, 2) / target->size();
                    break;
                case 1:
                    tar_var += pow((double)target->points[i].g - tar_avg, 2) / target->size();
                    break;
                case 2:
                    tar_var += pow((double)target->points[i].b - tar_avg, 2) / target->size();
                    break;
            }
        }
        //MV[ch][1] = abs(src_var - tar_var);
        *(MV + ch * 2 + 1) = abs(src_var - tar_var);
        //cout << "Varience_dif[" << ch << "] = " << *(MV + ch * 2 + 1) << endl;
    }    
    *(MV + 3 * 2 + 0) = (*(MV + 0 * 2 + 0) + *(MV + 1 * 2 + 0) + *(MV + 2 * 2 + 0)) / 3;
    *(MV + 3 * 2 + 1) = (*(MV + 0 * 2 + 1) + *(MV + 1 * 2 + 1) + *(MV + 2 * 2 + 1)) / 3;
    return MV;
}

void Gamma_color_change(pcl::PointCloud<pcl::PointXYZRGB>::Ptr origin, pcl::PointCloud<pcl::PointXYZRGB>::Ptr result, float gamma, const string& path) {
    copyPointCloud(*origin,*result);
    for (int i = 0; i < origin->size(); i++) {
        float fpixel;
        for (int ch = 0; ch < 3; ch++) {
            switch (ch) {
            case 0:
                fpixel = (float)(result->points[i].r) / 255.0;
                result->points[i].r = saturate_cast<uchar>(pow(fpixel, gamma) * 255.0f);
                break;
            case 1:
                fpixel = (float)(result->points[i].g) / 255.0;
                result->points[i].g = saturate_cast<uchar>(pow(fpixel, gamma) * 255.0f);
                break;
            case 2:
                fpixel = (float)(result->points[i].b) / 255.0;
                result->points[i].b = saturate_cast<uchar>(pow(fpixel, gamma) * 255.0f);
                break;
            }
        }
    }
    pcl::PLYWriter writer;
    writer.write<pcl::PointXYZRGB>(path + "/gamma_target.ply", *result, false, false);
}

void ShowHist(int* hist,int size, Mat& hist_img, int scale = 2,int color=0)
{
    int bins = size;
    double max_val = 0;
    for (int i = 0; i < bins; i++)
        max_val = hist[i] > max_val ? hist[i] : max_val;
    hist_img = Mat::zeros(bins, bins * scale, CV_8UC3);
    hist_img.setTo(Scalar(255,255,255));    
    for (int i = 0; i < bins; i++)
    {
        float bin_val = (float)hist[i];
        int intensity = cvRound(bin_val * bins / max_val);  //要绘制的高度
        cv::rectangle(hist_img, cv::Point(i * scale, bins - 1),
            cv::Point((i + 1) * scale - 1, bins - intensity),
            color == 0 ? CV_RGB(255, 0, 0) : color == 1 ? CV_RGB(0, 255, 0) : CV_RGB(0, 0, 255));
    }

    //cv::copyMakeBorder(hist_img,hist_img,10,10,10,10,cv::BORDER_CONSTANT,cv::Scalar(0,0,0));
}

int* thresh_biclass(int pointCount, float* difArray, const string& path, double* th_low, double* th_high) {
    const int intervalCount = 99999; // 期望分成多少分 = intervalCount + 1
    //int pointCount = _msize(correspond) / sizeof(correspond[0]);
    int* hist = new int[intervalCount + 1] {};  // 紀錄直方圖各區塊包含點數
    int* hist_pos = new int[pointCount] {};     // 記錄各點所在直方圖區塊位置
    int* biClass = new int[pointCount] {};  // 紀錄經 biClass 法後是屬於 close(1) ,moderate(2) 

    // 找出最大值最小值
    double min = 999.9, max = 0.0;
    for (int i = 0; i < pointCount; i++) {
        min = sqrt(difArray[i]) < min ? sqrt(difArray[i]) : min;
        max = sqrt(difArray[i]) > max ? sqrt(difArray[i]) : max;
        /*min = difArray[i] < min ? difArray[i] : min;
        max = difArray[i] > max ? difArray[i] : max;*/
    }
    // 計算相鄰直方圖距離差異
    double intervalVal = (max - min) / intervalCount;

    // 填入直方圖統計數量
    int pos; // 計算個別 difArray 所對應直方圖位置
    for (int i = 0; i < pointCount; i++) {
        pos = (int)((sqrt(difArray[i]) - min) / intervalVal);
        //pos = (int)((difArray[i] - min) / intervalVal);
        hist[pos] += 1;
        hist_pos[i] = pos;
    }

    // 計算 Otsu 最佳 threshold
    float b_max = -1;   // max between-variance
    int th = -1;        // max between-variance's correspond threshold 
    for (int k = 0; k < intervalCount + 1; k++) {   // 遍歷直方圖找 between-variance 最大位置
        // 計算 num(數量), sum(數值*數量,用來計算平均)
        // 第一群 (0~k)
        int num0 = 0;
        float sum0 = 0;

        // 第二群 (k+1~final)
        int num1 = 0;
        float sum1 = 0;

        for (int i = 0; i <= k; i++) {
            num0 += hist[i];
            sum0 += i * hist[i];
        }
        for (int i = k + 1; i < intervalCount + 1; i++) {
            num1 += hist[i];
            sum1 += i * hist[i];
        }

        // 計算 w(佔總體比例), u(平均數值)
        double w0 = (double)num0 / pointCount;
        double w1 = (double)num1 / pointCount;

        double u0 = (double)sum0 / num0;
        double u1 = (double)sum1 / num1;

        // 計算 between varience = w0 * w1 * (u0 - u1)^2
        // 參考 : https://blog.csdn.net/leonardohaig/article/details/120269341
        double sigma = w0 * w1 * pow((u0 - u1), 2);
        //cout << "k = " << k << "\t, sigma = " << w0 << " * " << w1 << " * " << u0-u1 << "^2 = " << sigma << endl;
        if (sigma >= b_max) {
            th = k;
            b_max = sigma;
        }
    }

    // 計算出 lower_mean 將距離較近之分群再做細分
    double lower_mean = 0;
    int lower_count = 0;
    int new_th = -1;
    for (int i = 0; i < pointCount; i++) {
        if (sqrt(difArray[i]) <= min + th * intervalVal) {
            lower_mean += sqrt(difArray[i]);
            lower_count++;
        }
    }
    lower_mean /= lower_count;
    new_th = (int)((lower_mean - min) / intervalVal);   // lower_mean 所在直方圖位置

    // 將分群結果賦予 target 點雲集: 1 -> close: 0 ~ new_th, 2 -> moderate: new_th+1 ~ final
    int class_1 = 0, class_2 = 0;
    for (int i = 0; i < pointCount; i++) {
        if (hist_pos[i] <= new_th) {
            biClass[i] = 1;
            class_1++;
        }
        else {
            biClass[i] = 2;
            class_2++;
        }
    }

    // 繪製 Otsu 法後的分群後直方圖
    Mat hist_img;
    int scale = 2;
    int size = (new_th * 2 / 500 + 1) * 500;
    
    ShowHist(hist, size, hist_img, scale);
    imwrite(path + "/biClass_" + std::to_string(size) + "_.jpg", hist_img);
    line(hist_img, Point(new_th * scale, 0), Point(new_th * scale, size), Scalar(0, 0, 0), 2);
    imwrite(path + "/biClass_" + std::to_string(size) + "_th.jpg", hist_img);
    
    // 輸出至 txt
    string filename = path + "/biClass_" + std::to_string(size) + "_.txt";
    ofstream outputFile(filename);
    if (outputFile.is_open()) {
        outputFile << "[min] = " << min << "\t[max] = " << max << "[interval] = " << intervalVal << endl;
        outputFile << "[th] = " << min + new_th * intervalVal << " (" << new_th << ")" << endl;
        for (int i = 0; i < size; i++) {
            outputFile << hist[i] << endl;
        }
    }

    printf("\n[close] : %d \t (%2.3f%)\n", class_1, 100 * (float)class_1 / pointCount);
    printf("[moderate] : %d \t (%2.3f%)\n", class_2, 100 * (float)class_2 / pointCount);
    printf("Dist(th_biClass) : %f (%d)\t\n", min + intervalVal * new_th , new_th);



    // 將 th 對應由 hist_pos 更改為實際數值
    * th_low = min + intervalVal * new_th;
    * th_high = max;

    return biClass;
}

void otsuLUT(int* his, float** result) {
    int size = 20000;
    for (int u = 0; u < size; u++)
        for (int v = 0; v < size; v++)
            result[u][v] = 0;
    float sum = 0, prob;
    for (int v = 0; v < size; v++)
        sum += his[v];
    if (sum <= 0) return;
    float** P = new float* [size];
    float** S = new float* [size];
    for (int i = 0; i < size; i++) {
        P[i] = new float[size] {};
        S[i] = new float[size] {};
    }

    P[0][0] = his[0];
    S[0][0] = his[0];
    for (int v = 1; v < size; v++) {
        prob = his[v] / sum;
        P[0][v] = P[0][v - 1] + prob;
        S[0][v] = S[0][v - 1] + (v + 1) * prob;
    }
    for (int u = 1; u < size; u++) {
        for (int v = 1; v < size; v++) {
            P[u][v] = P[0][v] - P[0][u - 1];
            S[u][v] = S[0][v] - S[0][u - 1];
        }
    }

    // result is equal (29) from Liao
    for (int u = 0; u < size; u++) {
        for (int v = 0; v < size; v++) {
            if (P[u][v] == 0)   // avoid divide by zero error
                result[u][v] = 0;
            else
                result[u][v] = pow(S[u][v], 2) / P[u][v];
        }
    }
}

void otsuCostFunc3(int* his, int* low, int* high) {
    int size = 20000;
    float v, max;
    float** h2d = new float* [size];
    for (int i = 0; i < size; i++)
        h2d[i] = new float[size] {};
    cout << "IN otsuLUT" << endl;
    otsuLUT(his, h2d);
    cout << "Out otsuLUT" << endl;
    int lo = size / 3;
    int hi = size * (float)(2 / 3);
    // default solution
    max = h2d[0][lo] + h2d[lo + 1][hi] + h2d[hi + 1][size - 1];
    // brutle force search
    for (int l = 0; l < (size - 2); l++) {
        for (int h = l + 1; h < (size - 1); h++) {
            v = h2d[0][l] + h2d[l + 1][h] + h2d[h + 1][size - 1];
            if (v > max) {
                lo = l;
                hi = h;
                max = v;
            }
        }
    }
    *low = lo;
    *high = hi;
}

int* thresh_triclass_(int pointCount, float* difArray, const string& path, double* th_low, double* th_high) {
    const int intervalCount = 19999; // 期望分成多少分 = intervalCount + 1    
    int* hist = new int[intervalCount + 1] {};  // 紀錄直方圖各區塊包含點數
    int* hist_pos = new int[pointCount] {};     // 記錄各點所在直方圖區塊位置
    int* triClass = new int[pointCount] {};  // 紀錄經 triClass 法後是屬於 close(1) ,moderate(2) ,distant(3)

    // 找出最大值最小值
    double min = 999.9, max = 0.0;
    for (int i = 0; i < pointCount; i++) {
        min = difArray[i] < min ? difArray[i] : min;
        max = difArray[i] > max ? difArray[i] : max;
    }
    // 計算相鄰直方圖距離差異
    double intervalVal = (max - min) / intervalCount;

    // 填入直方圖統計數量
    int pos; // 計算個別 difArray 所對應直方圖位置
    for (int i = 0; i < pointCount; i++) {
        pos = (int)((difArray[i] - min) / intervalVal);
        hist[pos] += 1;
        hist_pos[i] = pos;
    }

    int low, high;
    int close = 0, moderate = 0;
    otsuCostFunc3(hist,&low,&high);

    // 計算出 lower_mean 將距離較近之分群再做細分
    double lower_mean = 0;
    int lower_count = 0;
    int new_low = -1;
    for (int i = 0; i < pointCount; i++) {
        if (difArray[i] <= min + low * intervalVal) {
            lower_mean += difArray[i];
            lower_count++;
        }
    }
    lower_mean /= lower_count;
    new_low = (int)((lower_mean - min) / intervalVal);   // lower_mean 所在直方圖位置

    for (int i = 0; i < pointCount; i++) {
        if (hist_pos[i] <= new_low) {
            triClass[i] = 1;
            close++;
        }
        else if (hist_pos[i] <= high) {
            triClass[i] = 2;
            moderate++;
        }
        else {
            triClass[i] = 3;
        }
    }
    // 繪製 Otsu 法後的分群後直方圖
    Mat hist_img;
    int scale = 2;
    int size = (high / 500 + 1) * 500;
    ShowHist(hist, size, hist_img, scale);
    imwrite(path + "/triClass_" + std::to_string(size) + "_.jpg", hist_img);
    line(hist_img, Point(low * scale, 0), Point(low * scale, size), Scalar(0, 0, 0), 2);
    line(hist_img, Point(new_low * scale, 0), Point(new_low * scale, size), Scalar(0, 0, 0), 2);
    line(hist_img, Point(high * scale, 0), Point(high * scale, size), Scalar(0, 0, 0), 2);
    imwrite(path + "/triClass_" + std::to_string(size) + "_th.jpg", hist_img);

    // 輸出至 txt
    string filename = path + "/triClass_"+ std::to_string(size) +"_.txt";
    ofstream outputFile(filename);
    if (outputFile.is_open()) {
        outputFile << "[min] = " << min << "\t[max] = " << max << "[interval] = " << intervalVal << endl;
        outputFile << "[low] = " << min + low * intervalVal << " (" << low << ")" << endl; 
        outputFile << "[new_low] = " << min + new_low * intervalVal << " (" << new_low << ")" << endl;
        outputFile << "[high] = " << min + high * intervalVal << " (" << high << ")" << endl;
        for (int i = 0; i < 500; i++) {
            outputFile << hist[i] << endl;
        }
    }

    printf("\n[close] : %d \t (%2.3f%)\n", close, 100 * (float)close / pointCount);
    printf("[moderate] : %d \t (%2.3f%)\n", moderate, 100 * (float)moderate / pointCount);
    printf("[distant] : %d \t (%2.3f%)\n", (pointCount - close - moderate), 100 * (float)(pointCount - close - moderate) / pointCount);
    printf("Dist(th_triClass_) : %f (%d)\t %f (%d)\n", min + intervalVal * low, low,
        min + intervalVal * high, high);    

    cout << "\n[close] : " << close << "\t(" <<  100 * (float)close / pointCount << "%)\n";
    cout << "[moderate] : " << moderate << "\t(" << 100 * (float)moderate / pointCount << "%)\n";
    cout << "[distant] : " << (pointCount - close - moderate) << "\t(" << 100 * (float)(pointCount - close - moderate) / pointCount << "%)\n";
    cout << "Dist(th_triClass_) : " << min + intervalVal * low << " (" << low << ")\t " << min + intervalVal * high << " ("  << high << ")\n";

	// 將 th 對應由 hist_pos 更改為實際數值
	*th_low = min + intervalVal * new_low;
	*th_high = min + intervalVal * high;
    return triClass;
}

void swap(float* a, float* b) {
    float temp = *a;
    *a = *b;
    *b = temp;
}

void swap(int* a, int* b) {
    int temp = *a;
    *a = *b;
    *b = temp;
}

int Partition_big(float* arr, int* rank, int front, int end) {
    float pivot = arr[end];
    int i = front - 1;
    for (int j = front; j < end; j++) {
        if (arr[j] > pivot) {
            i++;
            swap(&arr[i], &arr[j]);
            swap(&rank[i], &rank[j]);
        }
    }
    i++;
    swap(&arr[i], &arr[end]);
    swap(&rank[i], &rank[end]);
    return i;
}

int Partition_big(int* arr, int* rank, int front, int end) {
    int pivot = arr[end];
    int i = front - 1;
    for (int j = front; j < end; j++) {
        if (arr[j] > pivot) {
            i++;
            swap(&arr[i], &arr[j]);
            swap(&rank[i], &rank[j]);
        }
    }
    i++;
    swap(&arr[i], &arr[end]);
    swap(&rank[i], &rank[end]);
    return i;
}

int Partition_small(float* arr, int* rank, int front, int end) {
    float pivot = arr[end];
    int i = front - 1;
    for (int j = front; j < end; j++) {
        if (arr[j] < pivot) {
            i++;
            swap(&arr[i], &arr[j]);
            swap(&rank[i], &rank[j]);
        }
    }
    i++;
    swap(&arr[i], &arr[end]);
    swap(&rank[i], &rank[end]);
    return i;
}

int Partition_small(int* arr, int* rank, int front, int end) {
    int pivot = arr[end];
    int i = front - 1;
    for (int j = front; j < end; j++) {
        if (arr[j] < pivot) {
            i++;
            swap(&arr[i], &arr[j]);
            swap(&rank[i], &rank[j]);
        }
    }
    i++;
    swap(&arr[i], &arr[end]);
    swap(&rank[i], &rank[end]);
    return i;
}

void QuickSort(int big, float* arr, int* rank, int front, int end) {
    if (front < end) {
        if (big == 1) {
            int pivot = Partition_big(arr, rank, front, end);
            QuickSort(1, arr, rank, front, pivot - 1);
            QuickSort(1, arr, rank, pivot + 1, end);
        }
        else {
            int pivot = Partition_small(arr, rank, front, end);
            QuickSort(0, arr, rank, front, pivot - 1);
            QuickSort(0, arr, rank, pivot + 1, end);
        }
    }
}

void QuickSort(int big, int* arr, int* rank, int front, int end) {
    if (front < end) {
        if (big == 1) {
            int pivot = Partition_big(arr, rank, front, end);
            QuickSort(1, arr, rank, front, pivot - 1);
            QuickSort(1, arr, rank, pivot + 1, end);
        }
        else {
            int pivot = Partition_small(arr, rank, front, end);
            QuickSort(0, arr, rank, front, pivot - 1);
            QuickSort(0, arr, rank, pivot + 1, end);
        }
    }
}

void write_array(int* rank, float* val, int size, const string& arrName, const string& path) {
    string filename = path + "/" + arrName + ".txt";
    ofstream outputFile(filename);

    if (outputFile.is_open()) {
        for (int i = 0; i < size; i++) {
            outputFile << arrName << "[" << i << "] = " << rank[i] << "\tval = " << val[i] << endl;
        }
    }
    else {
        cout << "Unable to open the file [ " << filename << " ]" << endl;
    }
}

void establish_correspondence(pcl::PointCloud<pcl::PointXYZRGB>::Ptr target, pcl::PointCloud<pcl::PointXYZRGB>::Ptr source, 
                                int* correspond, float* dist) {
    // 建立 k-d tree 
    pcl::KdTreeFLANN<pcl::PointXYZRGB> kdtree;
    kdtree.setInputCloud(source); 

    // 定義參數，只找最近鄰居 knn, k = 1  
    std::vector<int> pointIdxNKNSearch(1);
    std::vector<float> pointNKNSquaredDistance(1);    
    
    // 搜尋最鄰近之 source 點
    pcl::PointXYZRGB searchPoint;
    for (size_t i = 0; i < target->size(); i++) {
        searchPoint.x = target->points[i].x;
        searchPoint.y = target->points[i].y;
        searchPoint.z = target->points[i].z;
        searchPoint.r = (int)target->points[i].r;
        searchPoint.g = (int)target->points[i].g;
        searchPoint.b = (int)target->points[i].b;

        if (kdtree.nearestKSearch(searchPoint, 1, pointIdxNKNSearch, pointNKNSquaredDistance) > 0) {
            dist[i] = pointNKNSquaredDistance[0]; // 儲存距離
            correspond[i] = pointIdxNKNSearch[0]; // 儲存關係      
        }
    }
}

int** copyMatrix(int** matrix) {
    int** copy = new int* [3];
    for (int ch = 0; ch < 3; ch++)
        copy[ch] = new int[256];
    for (int ch = 0; ch < 3; ch++)
        for (int i = 0; i < 256; i++)
            copy[ch][i] = matrix[ch][i];
    return copy;
}

int** color_mapping_function(int** tar_cul_his, int** src_cul_his) {
    int** function = new int* [3];
    for (int ch = 0; ch < 3; ch++)
        function[ch] = new int[256] {};

    for (int ch = 0; ch < 3; ch++) {
        for (int i = 0; i < 256; i++) {
            for (int j = 0; j < 256; j++) {
                if (tar_cul_his[ch][i] <= src_cul_his[ch][j]) {
                    function[ch][i] = j;
                    j = 256;
                }
            }
        }
    }

    return function;
}

void function_delete(int** function, int** threshold, int** tar_his, int** src_his, const string& path,int*** re_fun) {    
    int index[3][256] = { 1 };    
    int gradient[3][256] = { 0 };    
    int** function_del_ma = copyMatrix(function);
    int** function_del_he = copyMatrix(function);
    int** function_del = copyMatrix(function);
    int min = 256, max = -1;
    int p1, p2;
    
    for (int ch = 0; ch < 3; ch++) {     
        // 刪除 src 過少情況
        for (int i = 0; i < 256; i++) {
            if (src_his[ch][i] < threshold[1][ch]) {
                for (int j = 0; j < 256; j++) {
                    if (function_del_ma[ch][j] == i) {
                        function_del_ma[ch][j] = 0;
                        function_del_he[ch][j] = 0;
                        function_del[ch][j] = 0;
                        //cout << "\n[del-source] i = " << j << " -> " << function[ch][j] << " to " << function_del[ch][j] << endl;
                    }                                            
                }                
            }
            if (src_his[ch][i] != 0) {
                min = i < min ? i : min;
                max = i > max ? i : max;
            }
        }
        // 刪除 tar 過少情況
        for (int i = 0; i < 256; i++) {
            if (tar_his[ch][i] < threshold[0][ch]) {
                function_del_ma[ch][i] = 0;
                function_del_he[ch][i] = 0;
                function_del[ch][i] = 0;
                //cout << "\n[del-target] i = " << i << " -> " << function_del[ch][i] << endl;
            }            
        }
        
        // 處理 src&tar，用 straigntening
        //cout << "min = " << min << "\nmax = " << max << endl;
        for (int i = 0; i < 256; i++) {
            p1 = -1, p2 = -1;
            if (function_del_ma[ch][i] == 0) {
                p1 = i - 1;
                for (int j = i + 1; j < 256; j++) {
                    if (function_del_ma[ch][j] != 0) {
                        p2 = j;
                        j = 256;
                    }
                }                
                //cout << "\nfrom : " << p1 << " [ " << p1 + 1 << " ~ " << p2 - 1 << " ] " << p2 << endl;
                if (p1 == -1) {
                    function_del_ma[ch][0] = min;                    
                    p1 = 0;
                }
                if (p2 == -1) {
                    function_del_ma[ch][255] = max;                    
                    p2 = 255;
                }    
                int count = 1;                
                for (int j = p1 + 1; j < p2; j++) {
                    function_del_ma[ch][j] = ((float)(function_del_ma[ch][p2] - function_del_ma[ch][p1]) / (p2 - p1)) * count + function_del_ma[ch][p1];
                    count++;
                    //cout << "\t[ " << j << " ] -> " << function_del_ma[ch][j] << endl;
                }
            }
        }
         
        // 處理 src&tar，用 Histogram Equalization
        for (int i = 0; i < 256; i++) {
            p1 = -1, p2 = -1;
            if (function_del_he[ch][i] == 0) {
                if (tar_his[ch][i] != 0) {
                    p1 = i - 1;
                    for (int j = i + 1; j < 256; j++) {
                        if (function_del_he[ch][j] != 0) {
                            p2 = j;
                            j = 256;
                        }
                    }
                    //cout << "\nfrom : " << p1 << " [ " << p1 + 1 << " ~ " << p2 - 1 << " ] " << p2 << endl;
                    if (p1 == -1) {
                        function_del_he[ch][0] = min;
                    }
                    if (p2 == -1) {
                        function_del_he[ch][255] = max;
                        p2 = 255;
                        
                        int total = 0;
                        for (int j = p1 + 1; j <= p2; j++) {
                            total += tar_his[ch][j];
                        }
                        int cul = 0;
                        for (int j = p1 + 1; j < p2; j++) {
                            cul += tar_his[ch][j];
                            function_del_he[ch][j] = ((double)(function_del_he[ch][p2] - function_del_he[ch][p1]) / total) * cul + function_del_he[ch][p1];
                        }
                        //cout << "\t[ " << i << " ] -> " << function_del_he[ch][i] << endl;
                    }
                    else {
                        int total = 0;
                        for (int j = p1 + 1; j <= p2; j++) {
                            total += tar_his[ch][j];
                        }
                        int cul = 0;
                        for (int j = p1 + 1; j < p2; j++) {
                            cul += tar_his[ch][j];
                            function_del_he[ch][j] = ((double)(function_del_he[ch][p2] - function_del_he[ch][p1]) / total) * cul + function_del_he[ch][p1];
                            //cout << "\t[ " << j << " ] -> " << function_del_he[ch][j] << endl;
                        }
                    }
                }
                else {
                    function_del_he[ch][i] = function_del_he[ch][i - 1];
                }
            }
        }
        
    }

    saveMatrix(function_del_ma ,path + "/HM" + "/CMF_MA.txt");
    drawHistogram(function_del_ma, path + "/HM", "MA");
    saveMatrix(function_del_he, path + "/HM" + "/CMF_HE.txt");
    drawHistogram(function_del_he, path + "/HM", "HE");
    saveMatrix(function_del, path + "/HM" + "/CMF_del.txt");
    drawHistogram(function_del, path + "/HM", "del");
    re_fun[1] = function_del_ma;
    re_fun[2] = function_del_he;
    //re_fun[3] = function;
}

void createFolder(const string& path, const string& name) {
    if (!fs::exists(path + "/" + name)) {
        if (fs::create_directory(path + "/" + name)) {
            cout << "Folder [ " << path + "/" + name << " ] create success." << endl;
        }
        else {
            cout << "Can't create folder [ " << path + "/" + name << " ]." << endl;
        }
    }
    else {
        cout << "Folder [ " << path + "/" + name << " ] already exist." << endl;
    }
}

int*** cal_histogram(pcl::PointCloud<pcl::PointXYZRGB>::Ptr target, pcl::PointCloud<pcl::PointXYZRGB>::Ptr source,
    int* inter_intra, int* correspond, const string& path, int which) { // which: [1~3]c/m/d, [4]c+m, [5]c+m+d
    int** tar_his = new int* [3];
    int** src_his = new int* [3];
    int** tar_cul_his = new int* [3];
    int** src_cul_his = new int* [3];
    int*** function = new int** [4]; // function[0] = Fecker, function[1] = Ma, function[2] = He, function[3] = function_del
    //int** function_del;
    int** tar_cul_rank;
    int** src_cul_rank;    
    int** tar_rank = new int* [3];
    int** src_rank = new int* [3];
    for (int ch = 0; ch < 3; ch++) {
        tar_his[ch] = new int[256] {};
        src_his[ch] = new int[256] {};
        tar_cul_his[ch] = new int[256] {};
        src_cul_his[ch] = new int[256] {};
        tar_rank[ch] = new int[256] {};
        src_rank[ch] = new int[256] {};
        for (int j = 0; j < 256; j++) {
            tar_rank[ch][j] = j;
            src_rank[ch][j] = j;
        }
    }

    //String createFolderName = folderPath + "/HM";    
    createFolder(path,"HM");

    // 計算 src 和 tar 各自的直方圖
    for (int ch = 0; ch < 3; ch++) {
        for (int i = 0; i < target->size(); i++) {
            if (which == 4) {
                if ((inter_intra[i] == 1) || (inter_intra[i] == 2)) {
                    switch (ch) {
                    case 0:
                        tar_his[0][target->points[i].r]++;
                        src_his[0][source->points[correspond[i]].r]++;
                        break;
                    case 1:
                        tar_his[1][target->points[i].g]++;
                        src_his[1][source->points[correspond[i]].g]++;
                        break;
                    case 2:
                        tar_his[2][target->points[i].b]++;
                        src_his[2][source->points[correspond[i]].b]++;
                        break;
                    }
                }
            }
            else if (which == 5) {
                switch (ch) {
                case 0:
                    tar_his[0][target->points[i].r]++;
                    src_his[0][source->points[correspond[i]].r]++;
                    break;
                case 1:
                    tar_his[1][target->points[i].g]++;
                    src_his[1][source->points[correspond[i]].g]++;
                    break;
                case 2:
                    tar_his[2][target->points[i].b]++;
                    src_his[2][source->points[correspond[i]].b]++;
                    break;
                }
            }                
            else if (inter_intra[i] == which) {
                switch (ch) {
                case 0:
                    tar_his[0][target->points[i].r]++;
                    src_his[0][source->points[correspond[i]].r]++;
                    break;
                case 1:
                    tar_his[1][target->points[i].g]++;
                    src_his[1][source->points[correspond[i]].g]++;
                    break;
                case 2:
                    tar_his[2][target->points[i].b]++;
                    src_his[2][source->points[correspond[i]].b]++;
                    break;
                }
            }
        }
    }

    Mat tar_hist_img, src_hist_img;
    ShowHist(tar_his[0], 256, tar_hist_img, 1, 0);
    imwrite(path + "/HM" + "/tar_his_r.jpg", tar_hist_img);

    ShowHist(src_his[0], 256, src_hist_img, 1, 0);
    imwrite(path + "/HM" + "/src_his_r.jpg", src_hist_img);

    // 計算 src 和 tar 各自累積直方圖   
    for (int ch = 0; ch < 3; ch++) {
        tar_cul_his[ch][0] = tar_his[ch][0];
        src_cul_his[ch][0] = src_his[ch][0];
        for (int i = 1; i < 256; i++) {
            tar_cul_his[ch][i] = tar_cul_his[ch][i - 1] + tar_his[ch][i];
            src_cul_his[ch][i] = src_cul_his[ch][i - 1] + src_his[ch][i];
        }
    }

    // 處存為 txt 以供查閱
    saveMatrix(tar_his, path + "/HM" + "/tar_his.txt");
    saveMatrix(tar_cul_his, path + "/HM" + "/tar_cul_his.txt");
    saveMatrix(src_his, path + "/HM" + "/src_his.txt");
    saveMatrix(src_cul_his, path + "/HM" + "/src_cul_his.txt");
    
    // 畫出累積直方圖以做比較
    drawHistogram(tar_cul_his, src_cul_his, path);

    // 計算 color mapping function
    function[0] = color_mapping_function(tar_cul_his, src_cul_his);
    saveMatrix(function[0], path + "/HM" + "/CMF_Fecker.txt");
    drawHistogram(function[0], path + "/HM", "Fecker");

    // 計算累積直方圖前5%最少的色彩值: 由小到大重新排列 -> 轉換成累積直方圖 -> 刪除最小的前5%色彩數值 
    tar_cul_rank = copyMatrix(tar_his);
    src_cul_rank = copyMatrix(src_his);
    int** function_del = copyMatrix(function[0]);
    
    // 由小到大重新排列
    int** th = new int* [2]; // 0 : tar, 1 : src
    th[0] = new int[3] {}, th[1] = new int[3] {};
    for (int ch = 0; ch < 3; ch++) {        
        QuickSort(0, tar_cul_rank[ch], tar_rank[ch], 0, 255);
        QuickSort(0, src_cul_rank[ch], src_rank[ch], 0, 255);
        // 換算成由小到大排列的累積直方圖
        for (int i = 1; i < 256; i++) {
            tar_cul_rank[ch][i] = tar_cul_rank[ch][i - 1] + tar_cul_rank[ch][i];
            src_cul_rank[ch][i] = src_cul_rank[ch][i - 1] + src_cul_rank[ch][i];
        }
        // 尋找 src& tar 各自 threshold -> 交給 function_delete 計算出回填數值 
        float rate = 0.05;
        for (int i = 0; i < 256; i++) {
            if (tar_cul_rank[ch][i] > tar_cul_rank[ch][255] * rate) {
                th[0][ch] = tar_his[ch][tar_rank[ch][i]];
                i = 256;
                cout << "th[tar][" << ch << "]" << th[0][ch] << endl;
            }            
        }
        for (int i = 0; i < 256; i++) {
            if (src_cul_rank[ch][i] > src_cul_rank[ch][255] * rate) {
                th[1][ch] = src_his[ch][src_rank[ch][i]];
                i = 256;
                cout << "th[src][" << ch << "]" << th[1][ch] << endl;
            }
        }
        cout << endl;
    }
    saveMatrix(tar_cul_rank, path + "/HM" + "/tar_cul_rank.txt");
    // 將function的對應刪除
    function_delete(function_del, th, tar_his, src_his, path, function);
    
    function[3] = function_del;    
    // function[0] = Fecker, function[1] = Ma, function[2] = He, function[3] = function_del    
    
    return function;
}

float gamma(float x) {
    return x > 0.04045 ? powf((x + 0.055f) / 1.055f, 2.4f) : (x / 12.92);
}

void RGB2Lab(int* color, float* Lab) {
    // RGB to XYZ
    float X, Y, Z;
    float RR = gamma(color[0] / 255.0);
    float GG = gamma(color[1] / 255.0);
    float BB = gamma(color[2] / 255.0);

    X = 0.4124564f * RR + 0.3575761f * GG + 0.1804375f * BB;
    Y = 0.2126729f * RR + 0.7151522f * GG + 0.0721750f * BB;
    Z = 0.0193339f * RR + 0.1191920f * GG + 0.9503041f * BB;

    // XYZ to Lab
    float fX, fY, fZ;
    const float param_13 = 1.0f / 3.0f;
    const float param_16116 = 16.0f / 116.0f;
    const float Xn = 0.950456f;
    const float Yn = 1.0f;
    const float Zn = 1.088754f;

    X /= (Xn);
    Y /= (Yn);
    Z /= (Zn);

    if (Y > 0.008856f)
        fY = pow(Y, param_13);
    else
        fY = 7.787f * Y + param_16116;

    if (X > 0.008856f)
        fX = pow(X, param_13);
    else
        fX = 7.787f * X + param_16116;

    if (Z > 0.008856)
        fZ = pow(Z, param_13);
    else
        fZ = 7.787f * Z + param_16116;

    Lab[0] = 116.0f * fY - 16.0f;
    Lab[0] = Lab[0] > 0.0f ? Lab[0] : 0.0f;
    Lab[1] = 500.0f * (fX - fY);
    Lab[2] = 200.0f * (fY - fZ);
}

float color_dif(int* init, int* ref) {
    float* Lab_init = new float[3] {};
    float* Lab_ref = new float[3] {};

    RGB2Lab(init, Lab_init);
    RGB2Lab(ref, Lab_ref);

    return sqrt(pow(abs(Lab_init[0] - Lab_ref[0]), 2) +
        pow(abs(Lab_init[1] - Lab_ref[1]), 2) +
        pow(abs(Lab_init[2] - Lab_ref[2]), 2));
}

pcl::PointCloud<pcl::PointXYZRGB>::Ptr white_mask(pcl::PointCloud<pcl::PointXYZRGB>::Ptr target) {
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr out_cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
    copyPointCloud(*target, *out_cloud);

    for (int i = 0; i < out_cloud->size(); i++) {
        out_cloud->points[i].r = 255;
        out_cloud->points[i].g = 255;
        out_cloud->points[i].b = 255;
    }

    return out_cloud;
}

void color_white(pcl::PointCloud<pcl::PointXYZRGB>::Ptr target, int num) {
    target->points[num].r = 255;
    target->points[num].g = 255;
    target->points[num].b = 255;
}

void color_red(pcl::PointCloud<pcl::PointXYZRGB>::Ptr target, int num) {
    target->points[num].r = 255;
    target->points[num].g = 0;
    target->points[num].b = 0;
}

void color_blue(pcl::PointCloud<pcl::PointXYZRGB>::Ptr target, int num) {
    target->points[num].r = 0;
    target->points[num].g = 0;
    target->points[num].b = 255;
}

pcl::PointCloud<pcl::PointXYZRGB>::Ptr triclass_color_correction_(
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr target,
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr source,
    int* triclass,
    int* correspond,
    float* dif,
    double low,
    double high,
    const string& path) {
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr out_cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
    copyPointCloud(*target, *out_cloud);

    // 建立 mask 供辨識 knn 中參考顏色差異過大點
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr knn_0_c(new pcl::PointCloud<pcl::PointXYZRGB>);
    copyPointCloud(*target, *knn_0_c);
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr knn_0_m(new pcl::PointCloud<pcl::PointXYZRGB>);
    copyPointCloud(*target, *knn_0_m);

    pcl::PLYWriter writer;
    // 計算 MHM
    // [0][ch][*] = Fecker, [1][ch][*] = Ding. [2][ch][*] = He, [3][ch][*] = Fecker_del, * for color value.
    // int*** CMF = cal_histogram(target, source, triclass, correspond, path, 1); // (1~3)c/m/d, (4)c+m, (5)all
    
    int count = 0;
    int print = 0;
    float close_color_constrain = 20.0;
    float moderate_color_constrain = 20.0;
    // [Step 1: close] -> (KNN + JBI)
    cout << "\nIn [CLOSE]..." << endl;
    #pragma omp parallel for
    for (int i = 0; i < out_cloud->size(); i++) {
        if (triclass[i] == 1) {
            count++;
            if (count % 1000 == 0)
                cout << "\n==================== " << count  << " ====================" << endl;
            int K = 10;
            pcl::KdTreeFLANN<pcl::PointXYZRGB> kdtree;  // 建立 k-d tree 
            kdtree.setInputCloud(source);   // 設定用來比較的點雲
            std::vector<int> pointIdxNKNSearch(K);
            std::vector<float> pointNKNSquaredDistance(K);
            pcl::PointXYZRGB searchPoint;
            // 搜索點
            searchPoint.x = out_cloud->points[i].x;
            searchPoint.y = out_cloud->points[i].y;
            searchPoint.z = out_cloud->points[i].z;
            searchPoint.r = (int)out_cloud->points[i].r;
            searchPoint.g = (int)out_cloud->points[i].g;
            searchPoint.b = (int)out_cloud->points[i].b;
            if (kdtree.nearestKSearch(searchPoint, K, pointIdxNKNSearch, pointNKNSquaredDistance) > 0) {
                int* r = new int[K] {}, * g = new int[K] {}, * b = new int[K] {};    // 鄰近點的RGB值
                long double* dist = new long double[K] {};    // K個鄰近點的距離
                int blend_color[3] = { 0 };	// 最後計算之顏色結果
                int init_color[3] = { searchPoint.r, searchPoint.g, searchPoint.b };	// target 點初始顏色值
                int real_K = 0;
                for (int j = 0; j < K; j++) {   // 讀入色彩數值以及距離差異    
                    if (real_K < 5) {
                        int ref_color[3] = { (int)source->points[pointIdxNKNSearch[j]].r, (int)source->points[pointIdxNKNSearch[j]].g, (int)source->points[pointIdxNKNSearch[j]].b };
                        if (color_dif(init_color, ref_color) < close_color_constrain) {
                            r[real_K] = (int)source->points[pointIdxNKNSearch[j]].r;
                            g[real_K] = (int)source->points[pointIdxNKNSearch[j]].g;
                            b[real_K] = (int)source->points[pointIdxNKNSearch[j]].b;
                            dist[real_K] = sqrt(pointNKNSquaredDistance[j]);
                            real_K++;
                            if (count % 1000 == 0 && print) {
                                cout << "\n[ " << init_color[0] << " " << init_color[1] << " " << init_color[2] << " ] -> "
                                    << "[ " << ref_color[0] << " " << ref_color[1] << " " << ref_color[2] << " ]" << endl;
                                cout << "cor: (" << searchPoint.x << " " << searchPoint.y << " " << searchPoint.z << ") vs "
                                    << "(" << source->points[pointIdxNKNSearch[j]].x << " " << source->points[pointIdxNKNSearch[j]].y << " " << source->points[pointIdxNKNSearch[j]].z << ")" << endl;
                                cout << "[ " << j << " ] color dif = " << color_dif(init_color, ref_color) << endl;
                                cout << "[ " << j << " ] dist dif = " << dist[real_K - 1] << endl;
                            }
                        }
                    }
                    else
                        j = K;
					
                }    
                if (real_K == 0) {
                    // 紀錄 knn_0_c [白色] = moderate 純 NN 調色
                    color_white(knn_0_c, i);
                }
                else {
                    // 紀錄 knn_0_m [紅色] = moderate 純 KBI 調色
                    color_red(knn_0_c, i);
                }
                gaussian_color_change(real_K, init_color, blend_color, dist, r, g, b);
                                
                if (blend_color[0] == 256) { // case: real_K == 0
                    out_cloud->points[i].r = (int)source->points[pointIdxNKNSearch[0]].r;
                    out_cloud->points[i].g = (int)source->points[pointIdxNKNSearch[0]].g;
                    out_cloud->points[i].b = (int)source->points[pointIdxNKNSearch[0]].b;
                }
                else {
                    out_cloud->points[i].r += blend_color[0];
                    out_cloud->points[i].g += blend_color[1];
                    out_cloud->points[i].b += blend_color[2];
                }                
                
                if (count % 1000 == 0 && print) {
                    cout << "b [ " << (int)blend_color[0] << " " << (int)blend_color[1] << " " << (int)blend_color[2] << " ] " << endl;
                    cout << "\ncolor : [ " << init_color[0] << " " << init_color[1] << " " << init_color[2] << " ] -> [ "
                        << (int)out_cloud->points[i].r << " " << (int)out_cloud->points[i].g << " " << (int)out_cloud->points[i].b << " ] ";
                }
            }
            // 紀錄 knn_0_m 紅色 = close
            color_red(knn_0_m,i);
        }
    }
    writer.write<pcl::PointXYZRGB>(path + "/close.ply", *out_cloud, false, false);   

    // [Step 2: moderate] -> (KNN + JBI) * w1 + MHM * (1 - w1) 
    cout << "In [MODERATE]..." << endl;
    count = 0;
    print = 0;

    int*** CMF = cal_histogram(target, source, triclass, correspond, path, 1); // (1~3)c/m/d, (4)c+m, (5)all

    #pragma omp parallel for
    for (int i = 0; i < out_cloud->size(); i++) {
        if (triclass[i] == 2) {
            count++;
            if (count % 1000 == 0)
                cout << "\n************************ " << count << " ************************" << endl;
            int K = 10;
            pcl::KdTreeFLANN<pcl::PointXYZRGB> kdtree;  // 建立 k-d tree 
            kdtree.setInputCloud(source);   // 設定用來比較的點雲
            std::vector<int> pointIdxNKNSearch(K);
            std::vector<float> pointNKNSquaredDistance(K);
            pcl::PointXYZRGB searchPoint;
            // 搜索點
            searchPoint.x = out_cloud->points[i].x;
            searchPoint.y = out_cloud->points[i].y;
            searchPoint.z = out_cloud->points[i].z;
            searchPoint.r = (int)out_cloud->points[i].r;
            searchPoint.g = (int)out_cloud->points[i].g;
            searchPoint.b = (int)out_cloud->points[i].b;
            int blend_color[3] = { 0 };	// 最後計算之顏色補償值結果
            if (kdtree.nearestKSearch(searchPoint, K, pointIdxNKNSearch, pointNKNSquaredDistance) > 0) {
                int* r = new int[K] {}, * g = new int[K] {}, * b = new int[K] {};    // 鄰近點的RGB值
                long double* dist = new long double[K] {};    // K個鄰近點的距離    
                int init_color[3] = { searchPoint.r, searchPoint.g, searchPoint.b };	// target 點初始顏色值                
                int real_K = 0;
                for (int j = 0; j < K; j++) {   // 讀入色彩數值以及距離差異
                    if (real_K < 5) {
                        int ref_color[3] = { (int)source->points[pointIdxNKNSearch[j]].r, (int)source->points[pointIdxNKNSearch[j]].g, (int)source->points[pointIdxNKNSearch[j]].b };                        
                        if (count % 1000 == 0 && print) {
                            cout << "\n[Ref info] " << j << endl;
                            cout << "[ " << ref_color[0] << " " << ref_color[1] << " " << ref_color[2] << " ]" << endl;
                            cout << "(" << source->points[pointIdxNKNSearch[j]].x << " " << source->points[pointIdxNKNSearch[j]].y << " " << source->points[pointIdxNKNSearch[j]].z << ")" << endl;
                            cout << "[ " << j << " ] color dif = " << color_dif(init_color, ref_color) << endl;
                            cout << "[ " << j << " ] dist dif = " << dist[real_K - 1] << endl;
                        }
                        if (color_dif(init_color, ref_color) < moderate_color_constrain) {
                            r[real_K] = (int)source->points[pointIdxNKNSearch[j]].r;
                            g[real_K] = (int)source->points[pointIdxNKNSearch[j]].g;
                            b[real_K] = (int)source->points[pointIdxNKNSearch[j]].b;
                            dist[real_K] = sqrt(pointNKNSquaredDistance[j]);
                            real_K++;
                            if (count % 1000 == 0 && print) {
                                cout << "\n[Ref] " << j << endl;
                                cout << "\n[ " << init_color[0] << " " << init_color[1] << " " << init_color[2] << " ] -> "
                                    << "[ " << ref_color[0] << " " << ref_color[1] << " " << ref_color[2] << " ]" << endl;
                                cout << "cor: (" << searchPoint.x << " " << searchPoint.y << " " << searchPoint.z << ") vs "
                                    << "(" << source->points[pointIdxNKNSearch[j]].x << " " << source->points[pointIdxNKNSearch[j]].y << " " << source->points[pointIdxNKNSearch[j]].z << ")" << endl;
                                cout << "[ " << j << " ] color dif = " << color_dif(init_color, ref_color) << endl;
                                cout << "[ " << j << " ] dist dif = " << dist[real_K-1] << endl;
                            }
                        }
                    }
                    else
                        j = K;
                }    
                if (real_K == 0) {
                    // 紀錄 knn_0_m [白色] = moderate 純 MHM 調色
                    color_white(knn_0_m,i);
                }
                else {
                    // 紀錄 knn_0 [藍色] = moderate 混合 KNN+JBI 和 MHM
                    color_blue(knn_0_m, i);
                }
                gaussian_color_change(real_K, init_color, blend_color, dist, r, g, b);
            }
            // 利用距離百分比決定 w1, w2
            double w2 = (dif[i] - low) / (high - low); // weight for MHM
            double w1 = 1 - w2; // weight for JBI         

            if (count % 1000 == 0 && print) {
                cout << "high = " << high << "\tlow = " << low << endl;
                cout << "w1 = " << w1 << "\tw2 = " << w2 << endl;
                cout << "b [ " << (int)blend_color[0] << " " << (int)blend_color[1] << " " << (int)blend_color[2] << " ] " << endl;
                cout << "c [ " << (CMF[2][0][out_cloud->points[i].r] - out_cloud->points[i].r) << " "
                    << (CMF[2][1][out_cloud->points[i].g] - out_cloud->points[i].g) << " "
                    << (CMF[2][2][out_cloud->points[i].b] - out_cloud->points[i].b) << " ] " << endl;
                cout << "\n color [ " << (int)out_cloud->points[i].r << " " << (int)out_cloud->points[i].g << " " << (int)out_cloud->points[i].b << " ] ->";
            }

            if (blend_color[0] == 256) { // case: real_K == 0 參考點色差都太大
                out_cloud->points[i].r = (CMF[2][0][out_cloud->points[i].r]);
                out_cloud->points[i].g = (CMF[2][1][out_cloud->points[i].g]);
                out_cloud->points[i].b = (CMF[2][2][out_cloud->points[i].b]);
            }
            else {
                float r = 0.0, g = 0.0, b = 0.0;
                r = out_cloud->points[i].r + w1 * blend_color[0] + w2 * (CMF[2][0][out_cloud->points[i].r] - out_cloud->points[i].r);
                g = out_cloud->points[i].g + w1 * blend_color[1] + w2 * (CMF[2][1][out_cloud->points[i].g] - out_cloud->points[i].g);
                b = out_cloud->points[i].b + w1 * blend_color[2] + w2 * (CMF[2][2][out_cloud->points[i].b] - out_cloud->points[i].b);
                r = r >= 255.0 ? 255.0 : r <= 0.0 ? 0.0 : r;
                g = g >= 255.0 ? 255.0 : g <= 0.0 ? 0.0 : g;
                b = b >= 255.0 ? 255.0 : b <= 0.0 ? 0.0 : b;
                out_cloud->points[i].r = round(r);
                out_cloud->points[i].g = round(g);
                out_cloud->points[i].b = round(b);
                /*out_cloud->points[i].r +=  (CMF[2][0][out_cloud->points[i].r] - out_cloud->points[i].r); // 純用 MHM
                out_cloud->points[i].g +=  (CMF[2][1][out_cloud->points[i].g] - out_cloud->points[i].g);
                out_cloud->points[i].b +=  (CMF[2][2][out_cloud->points[i].b] - out_cloud->points[i].b);*/
                /*out_cloud->points[i].r += blend_color[0];    // 純用 JBI
                out_cloud->points[i].g += blend_color[1];
                out_cloud->points[i].b += blend_color[2];*/
                /*onlyHM->points[i].r = (CMF[2][0][onlyHM->points[i].r]);
                onlyHM->points[i].g = (CMF[2][1][onlyHM->points[i].g]);
                onlyHM->points[i].b = (CMF[2][2][onlyHM->points[i].b]);*/
            }
            if (count % 1000 == 0 && print) {
                cout << "[ " << (int)out_cloud->points[i].r << " " << (int)out_cloud->points[i].g << " " << (int)out_cloud->points[i].b <<  " ] " << endl;
            }
            // 紀錄 knn_0_c 藍色 = moderate
            color_blue(knn_0_c,i);
        }
    }
    writer.write<pcl::PointXYZRGB>(path + "/moderate.ply", *out_cloud, false, false);
    writer.write<pcl::PointXYZRGB>(path + "/knn_0_c.ply", *knn_0_c, false, false);
    writer.write<pcl::PointXYZRGB>(path + "/knn_0_m.ply", *knn_0_m, false, false);
    
    // [Step 3: distant]
    cout << "In [DISTANT]..." << endl;
    count = 0;

    CMF = cal_histogram(target, source, triclass, correspond, path, 1); // (1~3)c/m/d, (4)c+m, (5)all

    #pragma omp parallel for
    for (int i = 0; i < out_cloud->size(); i++) {
        if (triclass[i] == 3) {
            count++;
            if (count % 1000 == 0)
                cout << count << endl;
            out_cloud->points[i].r = CMF[2][0][out_cloud->points[i].r];
            out_cloud->points[i].g = CMF[2][1][out_cloud->points[i].g];
            out_cloud->points[i].b = CMF[2][2][out_cloud->points[i].b];
        }
    }
    writer.write<pcl::PointXYZRGB>(path + "/distant.ply", *out_cloud, false, false);

    return out_cloud;
}

pcl::PointCloud<pcl::PointXYZRGB>::Ptr biclass_color_correction(
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr target,
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr source,
    int* triclass,
    int* correspond,
    float* dif,
    double low,
    double high,
    const string& path) {    
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr out_cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
    copyPointCloud(*target, *out_cloud);

    pcl::PLYWriter writer;
    // 計算 MHM
    // [0][ch][*] = Fecker, [1][ch][*] = Ding. [2][ch][*] = He, [3][ch][*] = Fecker_del, * for color value.
    int*** CMF = cal_histogram(target, source, triclass, correspond, path, 1); // close(1), moderate(2), distant(3), c+m(4), all(5)
    
    int count = 0;
    int print = 0;
    float close_color_constrain = 20.0;
    float moderate_color_constrain = 10.0;
    // [Step 1: close] -> (KNN + JBI)
    cout << "\nIn [CLOSE]..." << endl;
    #pragma omp parallel for
    for (int i = 0; i < out_cloud->size(); i++) {
        if (triclass[i] == 1) {
            count++;
            if (count % 1000 == 0)
                cout << "\n==================== " << count << " ====================" << endl;
            int K = 10;
            pcl::KdTreeFLANN<pcl::PointXYZRGB> kdtree;  // 建立 k-d tree 
            kdtree.setInputCloud(source);   // 設定用來比較的點雲
            std::vector<int> pointIdxNKNSearch(K);
            std::vector<float> pointNKNSquaredDistance(K);
            pcl::PointXYZRGB searchPoint;
            // 搜索點
            searchPoint.x = out_cloud->points[i].x;
            searchPoint.y = out_cloud->points[i].y;
            searchPoint.z = out_cloud->points[i].z;
            searchPoint.r = (int)out_cloud->points[i].r;
            searchPoint.g = (int)out_cloud->points[i].g;
            searchPoint.b = (int)out_cloud->points[i].b;
            if (kdtree.nearestKSearch(searchPoint, K, pointIdxNKNSearch, pointNKNSquaredDistance) > 0) {
                int* r = new int[K] {}, * g = new int[K] {}, * b = new int[K] {};    // 鄰近點的RGB值
                long double* dist = new long double[K] {};    // K個鄰近點的距離
                int blend_color[3] = { 0 };	// 最後計算之顏色結果
                int init_color[3] = { searchPoint.r, searchPoint.g, searchPoint.b };	// target 點初始顏色值
                int real_K = 0;
                for (int j = 0; j < K; j++) {   // 讀入色彩數值以及距離差異
                    if (real_K < 5) {
                        int ref_color[3] = { (int)source->points[pointIdxNKNSearch[j]].r, (int)source->points[pointIdxNKNSearch[j]].g, (int)source->points[pointIdxNKNSearch[j]].b };
                        if (color_dif(init_color, ref_color) < close_color_constrain) {
                            r[real_K] = (int)source->points[pointIdxNKNSearch[j]].r;
                            g[real_K] = (int)source->points[pointIdxNKNSearch[j]].g;
                            b[real_K] = (int)source->points[pointIdxNKNSearch[j]].b;
                            dist[real_K] = sqrt(pointNKNSquaredDistance[j]);
                            real_K++;
                        }
                    }
                    else
                        j = K;
                }
                
                if (count % 1000 == 0 && print) {
                    cout << "point_pos = ( " << searchPoint.x << " " << searchPoint.y << " " << searchPoint.z << " )" << endl;
                    cout << "point_color = ( " << (int)searchPoint.r << " " << (int)searchPoint.g << " " << (int)searchPoint.b << " )" << endl;
                    for (int a = 0; a < K; a++) {
                        cout << "\tknn_point[ " << a << " ] pos= ( " << source->points[pointIdxNKNSearch[a]].x << " " << source->points[pointIdxNKNSearch[a]].y << " " << source->points[pointIdxNKNSearch[a]].z << " )" << endl;
                        cout << "\tknn_point[ " << a << " ] color= ( " << (int)source->points[pointIdxNKNSearch[a]].r << " " << (int)source->points[pointIdxNKNSearch[a]].g << " " << (int)source->points[pointIdxNKNSearch[a]].b << " )" << endl;
                    }
                }
                gaussian_color_change(real_K, init_color, blend_color, dist, r, g, b);
                if (blend_color[0] == 256) { // case: real_K == 0, 用最近 source 點取代
                    out_cloud->points[i].r = (int)source->points[pointIdxNKNSearch[0]].r;
                    out_cloud->points[i].g = (int)source->points[pointIdxNKNSearch[0]].g;
                    out_cloud->points[i].b = (int)source->points[pointIdxNKNSearch[0]].b;
                }
                else {
                    out_cloud->points[i].r += blend_color[0];
                    out_cloud->points[i].g += blend_color[1];
                    out_cloud->points[i].b += blend_color[2];
                }                
                if (count % 1000 == 0 && print) {
                    cout << "b [ " << (int)blend_color[0] << " " << (int)blend_color[1] << " " << (int)blend_color[2] << " ] " << endl;
                    cout << "\ncolor : [ " << init_color[0] << " " << init_color[1] << " " << init_color[2] << " ] -> [ "
                        << (int)out_cloud->points[i].r << " " << (int)out_cloud->points[i].g << " " << (int)out_cloud->points[i].b << " ] ";
                }
            }
        }
    }
    writer.write<pcl::PointXYZRGB>(path + "/close.ply", *out_cloud, false, false);

    pcl::PointCloud<pcl::PointXYZRGB>::Ptr onlyHM(new pcl::PointCloud<pcl::PointXYZRGB>);
    copyPointCloud(*out_cloud, *onlyHM);

    // [Step 2: moderate] -> (KNN + JBI) * w1 + MHM * (1 - w1) 
    cout << "In [MODERATE]..." << endl;
    count = 0;
    print = 0;

    // 建立 mask 供辨識 knn 中參考顏色差異過大點
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr knn_0;
    knn_0 = white_mask(target);

    #pragma omp parallel for
    for (int i = 0; i < out_cloud->size(); i++) {
        if (triclass[i] == 2) {
            count++;
            if (count % 1000 == 0)
                cout << "\n************************ " << count << " ************************" << endl;
            int K = 10;
            pcl::KdTreeFLANN<pcl::PointXYZRGB> kdtree;  // 建立 k-d tree 
            kdtree.setInputCloud(source);   // 設定用來比較的點雲
            std::vector<int> pointIdxNKNSearch(K);
            std::vector<float> pointNKNSquaredDistance(K);
            pcl::PointXYZRGB searchPoint;
            // 搜索點
            searchPoint.x = out_cloud->points[i].x;
            searchPoint.y = out_cloud->points[i].y;
            searchPoint.z = out_cloud->points[i].z;
            searchPoint.r = (int)out_cloud->points[i].r;
            searchPoint.g = (int)out_cloud->points[i].g;
            searchPoint.b = (int)out_cloud->points[i].b;
            int blend_color[3] = { 0 };	// 最後計算之顏色補償值結果
            if (kdtree.nearestKSearch(searchPoint, K, pointIdxNKNSearch, pointNKNSquaredDistance) > 0) {
                int* r = new int[K] {}, * g = new int[K] {}, * b = new int[K] {};    // 鄰近點的RGB值
                long double* dist = new long double[K] {};    // K個鄰近點的距離                
                int init_color[3] = { searchPoint.r, searchPoint.g, searchPoint.b };	// target 點初始顏色值
                int real_K = 0;
                for (int j = 0; j < K; j++) {   // 讀入色彩數值以及距離差異
                    if (real_K < 5) {
                        int ref_color[3] = { (int)source->points[pointIdxNKNSearch[j]].r, (int)source->points[pointIdxNKNSearch[j]].g, (int)source->points[pointIdxNKNSearch[j]].b };
                        if (color_dif(init_color, ref_color) < moderate_color_constrain) {
                            r[real_K] = (int)source->points[pointIdxNKNSearch[j]].r;
                            g[real_K] = (int)source->points[pointIdxNKNSearch[j]].g;
                            b[real_K] = (int)source->points[pointIdxNKNSearch[j]].b;
                            dist[real_K] = sqrt(pointNKNSquaredDistance[j]);
                            real_K++;
                        }
                    }
                    else
                        j = K;
                }
                if (real_K == 0) {
                    color_red(knn_0, i);
                }
                gaussian_color_change(real_K, init_color, blend_color, dist, r, g, b);
            }
            // 利用距離百分比決定 w1, w2
            double w2 = (dif[i] - low) / (high - low); // weight for MHM
            double w1 = 1 - w2; // weight for JBI

            // 利用指數函數決定 w1, w2
            //double d_var = d_variance_moderate(dif,triclass);
            //double w1 = exp((double)(-1) * pow(dif[i], 2) / d_var); // weight for JBI
            //double w2 = 1 - w1; // weight for MHM            

            if (count % 1000 == 0 && print) {
                cout << "w1 = " << w1 << "\tw2 = " << w2 << endl;
                cout << "b [ " << (int)blend_color[0] << " " << (int)blend_color[1] << " " << (int)blend_color[2] << " ] " << endl;
                cout << "c [ " << (CMF[2][0][out_cloud->points[i].r] - out_cloud->points[i].r) << " "
                    << (CMF[2][1][out_cloud->points[i].g] - out_cloud->points[i].g) << " "
                    << (CMF[2][2][out_cloud->points[i].b] - out_cloud->points[i].b) << " ] " << endl;
                cout << "\n color [ " << (int)out_cloud->points[i].r << " " << (int)out_cloud->points[i].g << " " << (int)out_cloud->points[i].b << " ] ->";
            }

            if (blend_color[0] == 256) { // case: real_K == 0, 用 CMF 數值回填
                out_cloud->points[i].r += (CMF[2][0][out_cloud->points[i].r] - out_cloud->points[i].r);
                out_cloud->points[i].g += (CMF[2][1][out_cloud->points[i].g] - out_cloud->points[i].g);
                out_cloud->points[i].b += (CMF[2][2][out_cloud->points[i].b] - out_cloud->points[i].b);
            }
            else {
                out_cloud->points[i].r += w1 * blend_color[0] + w2 * (CMF[2][0][out_cloud->points[i].r] - out_cloud->points[i].r);
                out_cloud->points[i].g += w1 * blend_color[1] + w2 * (CMF[2][1][out_cloud->points[i].g] - out_cloud->points[i].g);
                out_cloud->points[i].b += w1 * blend_color[2] + w2 * (CMF[2][2][out_cloud->points[i].b] - out_cloud->points[i].b);
                /*out_cloud->points[i].r +=  (CMF[2][0][out_cloud->points[i].r] - out_cloud->points[i].r); // 純用 MHM
                out_cloud->points[i].g +=  (CMF[2][1][out_cloud->points[i].g] - out_cloud->points[i].g);
                out_cloud->points[i].b +=  (CMF[2][2][out_cloud->points[i].b] - out_cloud->points[i].b);*/
                /*out_cloud->points[i].r += blend_color[0];    // 純用 JBI
                out_cloud->points[i].g += blend_color[1];
                out_cloud->points[i].b += blend_color[2];*/
                /*onlyHM->points[i].r = (CMF[2][0][onlyHM->points[i].r]);
                onlyHM->points[i].g = (CMF[2][1][onlyHM->points[i].g]);
                onlyHM->points[i].b = (CMF[2][2][onlyHM->points[i].b]);*/
            }
            if (count % 1000 == 0 && print) {
                cout << "[ " << (int)out_cloud->points[i].r << " " << (int)out_cloud->points[i].g << " " << (int)out_cloud->points[i].b << " ] " << endl;
            }
        }
    }
    writer.write<pcl::PointXYZRGB>(path + "/moderate.ply", *out_cloud, false, false);
    writer.write<pcl::PointXYZRGB>(path + "/knn_0.ply", *knn_0, false, false);

    // [Step 3: distant]
    //cout << "In [DISTANT]..." << endl;
    /*count = 0;
    #pragma omp parallel for
    for (int i = 0; i < result->size(); i++) {
        if (triclass[i] == 3) {
            count++;
            if (count % 1000 == 0)
                cout << count << endl;
            result->points[i].r = CMF5[2][0][result->points[i].r];
            result->points[i].g = CMF5[2][1][result->points[i].g];
            result->points[i].b = CMF5[2][2][result->points[i].b];
        }
    }
    writer.write<pcl::PointXYZRGB>(path + "/distant.ply", *result, false, false);*/
    
    return out_cloud;
}

int** cmf_cal(pcl::PointCloud<pcl::PointXYZRGB>::Ptr dark, pcl::PointCloud<pcl::PointXYZRGB>::Ptr proccessed, std::vector<int>& border_origin_index) {
    int** tar_his = new int* [3];
    int** src_his = new int* [3];
    int** cmf = new int* [3];
    for (int ch = 0; ch < 3; ch++) {
        tar_his[ch] = new int[256] {};
        src_his[ch] = new int[256] {};
        cmf[ch] = new int[256] {};
    }

    // 計算 src(處理後) 和 tar(處理前) 各自的直方圖
    for (int ch = 0; ch < 3; ch++) {
        for (int i = 0; i < border_origin_index.size(); i++) {            
            switch (ch) {
            case 0:
                tar_his[0][dark->points[border_origin_index[i]].r]++;
                src_his[0][proccessed->points[border_origin_index[i]].r]++;
                break;
            case 1:
                tar_his[1][dark->points[border_origin_index[i]].g]++;
                src_his[1][proccessed->points[border_origin_index[i]].g]++;
                break;
            case 2:
                tar_his[2][dark->points[border_origin_index[i]].b]++;
                src_his[2][proccessed->points[border_origin_index[i]].b]++;
                break;
            }            
        }
    }

    // 計算 src 和 tar 各自累積直方圖   
    for (int ch = 0; ch < 3; ch++) {
        for (int i = 1; i < 256; i++) {
            tar_his[ch][i] = tar_his[ch][i - 1] + tar_his[ch][i];
            src_his[ch][i] = tar_his[ch][i - 1] + src_his[ch][i];
        }
    }

    cmf = color_mapping_function(tar_his, src_his);

    return cmf;
}

void histogram_mapping(pcl::PointCloud<pcl::PointXYZRGB>::Ptr target, int** function, int* inter_intra) {
    
    for (int i = 0; i < target->size(); i++) {
        if (inter_intra[i] == 0) {
            target->points[i].r = function[0][target->points[i].r];
            target->points[i].g = function[1][target->points[i].g];
            target->points[i].b = function[2][target->points[i].b];
        }
    }    
}

pcl::PointCloud<pcl::PointXYZRGB>::Ptr split_target_by_num(
    int num, 
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr target, 
    int* which_class,
    std::vector<int>& point_origin_index) {
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr out_cloud(new pcl::PointCloud<pcl::PointXYZRGB>);  
    pcl::PointIndices::Ptr inliers(new pcl::PointIndices());
    
    for (int i = 0; i < target->size(); i++) {
        if (which_class[i] == num) {
            inliers->indices.push_back(i);
            point_origin_index.push_back(i);
        }
    }

    pcl::ExtractIndices<pcl::PointXYZRGB> extract;
    extract.setInputCloud(target);
    extract.setIndices(inliers);

    extract.setNegative(false); // 取 inliers 中的點
    //extract.setNegative(true);  // 取 liniers 以外的點
    extract.filter(*out_cloud);

    return out_cloud;
}

pcl::PointCloud<pcl::PointXYZRGB>::Ptr split_correspond(
    int inter,
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr source,
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr target,
    int* inter_intra,
    int* correspond,
    std::vector<int>& point_origin_index) {
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr out_cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
    pcl::PointIndices::Ptr inliers(new pcl::PointIndices());

    for (int i = 0; i < target->size(); i++) {
        if (inter_intra[i] == inter) {
            inliers->indices.push_back(correspond[i]);
            point_origin_index.push_back(correspond[i]);
        }
    }

    pcl::ExtractIndices<pcl::PointXYZRGB> extract;
    extract.setInputCloud(source);
    extract.setIndices(inliers);

    extract.setNegative(false); // 取 inliers 中的點
    //extract.setNegative(true);  // 取 liniers 以外的點
    extract.filter(*out_cloud);

    return out_cloud;
}

float square_dist(pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud, int p1, int p2) {
    return pow(cloud->points[p1].x - cloud->points[p2].x, 2) + pow(cloud->points[p1].y - cloud->points[p2].y, 2)
        + pow(cloud->points[p1].z - cloud->points[p2].z, 2);
}

pcl::PointCloud<pcl::PointXYZRGB>::Ptr knn_color_correction(
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr source,
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr target) {
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr out_cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
    copyPointCloud(*target, *out_cloud);

    // 建立 k-d tree 
    pcl::KdTreeFLANN<pcl::PointXYZRGB> kdtree;
    kdtree.setInputCloud(source); // 設定用來比較的點雲

    // 定義 k-d tree 參數
    int K = 5;
    int pointCount = target->size();
    std::vector<int> pointIdxNKNSearch(K);
    std::vector<float> pointNKNSquaredDistance(K);
    pcl::PointXYZRGB searchPoint;

    // K-nn 找最近點 & 用 K 個鄰居色彩以及距離權重來做調色
    // 記錄開始時間點
    auto start = chrono::high_resolution_clock::now();
    for (size_t i = 0; i < pointCount; i++) {
        // 填入搜尋點數值
        searchPoint.x = out_cloud->points[i].x;
        searchPoint.y = out_cloud->points[i].y;
        searchPoint.z = out_cloud->points[i].z;
        searchPoint.r = (int)out_cloud->points[i].r;
        searchPoint.g = (int)out_cloud->points[i].g;
        searchPoint.b = (int)out_cloud->points[i].b;

        if (kdtree.nearestKSearch(searchPoint, K, pointIdxNKNSearch, pointNKNSquaredDistance) > 0)
        {
            int* r = new int[K] {}, * g = new int[K] {}, * b = new int[K] {};    // 鄰近點的RGB值
            float* dist = new float[K] {};    // K個鄰近點的距離
            // 儲存K個鄰近點的色彩資訊&距離
            for (int j = 0; j < K; ++j) {
                r[j] = (int)source->points[pointIdxNKNSearch[j]].r;
                g[j] = (int)source->points[pointIdxNKNSearch[j]].g;
                b[j] = (int)source->points[pointIdxNKNSearch[j]].b;
                dist[j] = sqrt(pointNKNSquaredDistance[j]);                
            }

            // Color change - avg weighted color blending
            int fr = 0, fg = 0, fb = 0;

            for (int i = 0; i < K; i++) {
                fr += r[i];
                fg += g[i];
                fb += b[i];
            }

            out_cloud->points[i].r = round((float)fr / K);
            out_cloud->points[i].g = round((float)fg / K);
            out_cloud->points[i].b = round((float)fb / K);
        }
    }
    // 記錄結束時間點
    auto end = chrono::high_resolution_clock::now();
    // 計算時間間隔
    auto duration = chrono::duration_cast<std::chrono::milliseconds>(end - start);
    cout << "[knn]time : " << duration.count() << "ms" << endl;

    return out_cloud;
}

pcl::PointCloud<pcl::PointXYZRGB>::Ptr direct_color_correction(
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr source,
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr target,
    int* correspond) {
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr out_cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
    copyPointCloud(*target, *out_cloud);

    // 記錄開始時間點
    auto start = chrono::high_resolution_clock::now();
    for (int i = 0; i < target->size(); i++) {
        out_cloud->points[i].r = source->points[correspond[i]].r;
        out_cloud->points[i].g = source->points[correspond[i]].g;
        out_cloud->points[i].b = source->points[correspond[i]].b;
    }
    // 記錄結束時間點
    auto end = chrono::high_resolution_clock::now();
    // 計算時間間隔
    auto duration = chrono::duration_cast<std::chrono::milliseconds>(end - start);
    cout << "[direct]time : " << duration.count() << "ms" << endl;

    return out_cloud;
}

pcl::PointCloud<pcl::PointXYZRGB>::Ptr Fecker_color_correction(
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr source,
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr target, 
    int* triclass,
    int* correspond,    
    const string& path){
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr out_cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
    copyPointCloud(*target, *out_cloud);
    // 記錄開始時間點
    auto start = chrono::high_resolution_clock::now();
    int*** CMF = cal_histogram(target, source, triclass, correspond, path, 5);

    for (int i = 0; i < out_cloud->size(); i++) {
        out_cloud->points[i].r = CMF[0][0][out_cloud->points[i].r];
        out_cloud->points[i].g = CMF[0][1][out_cloud->points[i].g];
        out_cloud->points[i].b = CMF[0][2][out_cloud->points[i].b];
    }
    // 記錄結束時間點
    auto end = chrono::high_resolution_clock::now();
    // 計算時間間隔
    auto duration = chrono::duration_cast<std::chrono::milliseconds>(end - start);
    cout << "[Fecker]time : " << duration.count() << "ms" << endl;

    return out_cloud;
}

pcl::PointCloud<pcl::PointXYZRGB>::Ptr Ding_color_correction(
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr source,
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr target,
    int* triclass,
    int* correspond,
    const string& path) {
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr out_cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
    copyPointCloud(*target, *out_cloud);
    // 記錄開始時間點
    auto start = chrono::high_resolution_clock::now();
    int*** CMF = cal_histogram(target, source, triclass, correspond, path, 5);

    for (int i = 0; i < out_cloud->size(); i++) {
        out_cloud->points[i].r = CMF[1][0][out_cloud->points[i].r];
        out_cloud->points[i].g = CMF[1][1][out_cloud->points[i].g];
        out_cloud->points[i].b = CMF[1][2][out_cloud->points[i].b];
    }
    // 記錄結束時間點
    auto end = chrono::high_resolution_clock::now();
    // 計算時間間隔
    auto duration = chrono::duration_cast<std::chrono::milliseconds>(end - start);
    cout << "[Ding]time : " << duration.count() << "ms" << endl;

    return out_cloud;
}

void color_trans(pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud, float a, float b, int ch) {
    for (int i = 0; i < cloud->size(); i++) {
        float temp = 0.0;
        switch (ch) {
        case 0:
            temp = a * (cloud->points[i].r) + b;
            temp = temp >= 255.0 ? 255.0 : temp <= 0.0 ? 0.0 : temp;
            cloud->points[i].r = round(temp);
            break;
        case 1:
            temp = a * (cloud->points[i].g) + b;
            temp = temp >= 255.0 ? 255.0 : temp <= 0.0 ? 0.0 : temp;
            cloud->points[i].g = round(temp);
            break;
        case 2:
            temp = a * (cloud->points[i].b) + b;
            temp = temp >= 255.0 ? 255.0 : temp <= 0.0 ? 0.0 : temp;
            cloud->points[i].b = round(temp);
            break;
        }        
    }
}

void mean_preserve(pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud, float *my_mean, float *ref_mean) {
    for (int ch = 0; ch < 3; ch++) {
        for (int i = 0; i < cloud->size(); i++) {
            float temp = 0.0;
            float avg = (my_mean[ch] + ref_mean[ch]) / 2;
            float rate = 0.5;
            switch (ch) {
            case 0:
                temp = cloud->points[i].r + rate * (avg - my_mean[0]);
                temp = temp >= 255.0 ? 255.0 : temp <= 0.0 ? 0.0 : temp;
                cloud->points[i].r = round(temp);
                break;
            case 1:
                temp = cloud->points[i].g + rate * (avg - my_mean[1]);
                temp = temp >= 255.0 ? 255.0 : temp <= 0.0 ? 0.0 : temp;
                cloud->points[i].g = round(temp);
                break;
            case 2:
                temp = cloud->points[i].b + rate * (avg - my_mean[2]);
                temp = temp >= 255.0 ? 255.0 : temp <= 0.0 ? 0.0 : temp;
                cloud->points[i].b = round(temp);
                break;
            }
        }
    }
}

void Yu_color_correction(
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr source,
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr target,
    int* triclass,
    int* correspond,
    float *dist,
    const string& path) {
    // cal target's mean & STD
    // mean
    float tar_mean[3] = { }, tar_std[3] = { };
    float r = 0.0, g = 0.0, b = 0.0;
    int count = 0;
    float dist_th = 0.003;
    for (int i = 0; i< target->size(); i++) {
        if (sqrt(dist[i]) < 0.003) {
            r += target->points[i].r;
            g += target->points[i].g;
            b += target->points[i].b;
            count++;
        }
    }
    tar_mean[0] = r / count;
    tar_mean[1] = g / count;
    tar_mean[2] = b / count;

    // STD
    r = 0.0, g = 0.0, b = 0.0;
    for (int i = 0; i < target->size(); i++) {
        if (sqrt(dist[i]) < dist_th) {
            r += pow((target->points[i].r - tar_mean[0]), 2);
            g += pow((target->points[i].g - tar_mean[1]), 2);
            b += pow((target->points[i].b - tar_mean[2]), 2);
        }
    }
    tar_std[0] = sqrt(r / count);
    tar_std[1] = sqrt(g / count);
    tar_std[2] = sqrt(b / count);
    
    // cal source's mean & STD
    // mean
    float src_mean[3] = { }, src_std[3] = { };
    r = 0.0, g = 0.0, b = 0.0;
    for (int i = 0; i < target->size(); i++) {
        if (sqrt(dist[i]) < 0.003) {
            r += source->points[correspond[i]].r;
            g += source->points[correspond[i]].g;
            b += source->points[correspond[i]].b;
        }
    }
    src_mean[0] = r / count;
    src_mean[1] = g / count;
    src_mean[2] = b / count;

    // STD
    r = 0.0, g = 0.0, b = 0.0;
    for (int i = 0; i < target->size(); i++) {
        if (sqrt(dist[i]) < 0.003) {
            r += pow((source->points[correspond[i]].r - src_mean[0]), 2);
            g += pow((source->points[correspond[i]].g - src_mean[1]), 2);
            b += pow((source->points[correspond[i]].b - src_mean[2]), 2);
        }
    }
    src_std[0] = sqrt(r / count);
    src_std[1] = sqrt(g / count);
    src_std[2] = sqrt(b / count);

    // fill matrix b
    MatrixXf A(6, 4);
    MatrixXf x(4, 1);
    MatrixXf L(6, 1);

    float tar_after_mean[3] = { }, src_after_mean[3] = { };
    mean_cal(target, tar_after_mean);
    mean_cal(source, src_after_mean);
    /*cout << "[Yu] Before" << endl;
    for (int i = 0; i < 3; i++) {
        cout << "[ " << i << " ] = tar[ " << tar_after_mean[i] << " ] , src[ " << src_after_mean[i] << " ]" << endl;
    }*/

    for (int ch = 0; ch < 3; ch++) {
        L << 0, 0, tar_mean[ch], tar_std[ch], src_mean[ch], src_std[ch];
        A << tar_mean[ch], 1, (-1)* src_mean[ch], (-1),
             tar_std[ch], 0, (-1)* src_std[ch], 0,
             tar_mean[ch], 1, 0, 0,
             tar_std[ch], 0, 0, 0,
             0, 0, src_mean[ch], 1,
             0, 0, src_std[ch], 0;
        x = (A.transpose() * A).fullPivLu().solve(A.transpose() * L);
                
        color_trans(target, x(0), x(1), ch);
        color_trans(source, x(2), x(3), ch);        

        mean_cal(target, tar_after_mean);
        mean_cal(source, src_after_mean);
        /*cout << "[Yu] After" << endl;        
            cout << "[ " << ch << " ] = tar[ " << tar_after_mean[ch] << " ] , src[ " << src_after_mean[ch] << " ]" << endl;*/
    }

    mean_preserve(target, tar_after_mean, src_after_mean);
    mean_preserve(source, src_after_mean, tar_after_mean);

    mean_cal(target, tar_after_mean);
    mean_cal(source, src_after_mean);
    /*cout << "[Yu] Result" << endl;
    for (int i = 0; i < 3; i++) {
        cout << "[ " << i << " ] = tar[ " << tar_after_mean[i] << " ] , src[ " << src_after_mean[i] << " ]" << endl;
    }*/
}

int main() {  
    for(int picNum = 0; picNum < 11; picNum++) {
        // Data name : Mario(0~11), Squirrel(0~11), Duck(0~13), PeterRabbit(0~15), Doll(5~18),  Frog(0~13)  
        string name = "Mario";
        string source_num;
        string target_num;
        string folderPath;
        if (picNum < 9) {
            source_num = "0" + std::to_string(picNum);
            target_num = "0" + std::to_string(picNum + 1);
        }
        else if (picNum == 9) {
            source_num = "0" + std::to_string(picNum);
            target_num = std::to_string(picNum + 1);
        }
        else {
            source_num = std::to_string(picNum);
            target_num = std::to_string(picNum + 1);
        }            

        folderPath = "Data/" + name + "/test/s" + source_num + "t" + target_num;
        if (!fs::exists(folderPath)) {
            if (fs::create_directory(folderPath)) {
                cout << "Folder create success." << endl;
            }
            else {
                cout << "Can't create folder." << endl;
            }
        }
        else {
            cout << "Folder already exist." << endl;
        }                
        
        int printTofile = 1;

        // 打開預計輸出之文本檔
        std::ofstream outputFile(folderPath + "/output.txt");
        // 保存 cout 的原始輸出流以便結束後回復
        std::streambuf* coutBuffer = std::cout.rdbuf();
        // 用法
        // 將 cout 的輸出流重定向到文本檔
        std::cout.rdbuf(outputFile.rdbuf());          
        // 恢復 cout 的原始輸出流
        std::cout.rdbuf(coutBuffer);
                
        // 點雲資料以及 transformation matrix 路徑
        string src_filename = "Data/" + name + "/point/" + source_num + ".ply";
        string tar_filename = "Data/" + name + "/point/" + target_num + ".ply";
        string matrix_location = "Data/" + name + "/point/transformation/matrix_" + source_num + "_" + target_num + ".txt";
        
        // 產生點雲準備讀取資料
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_source(new pcl::PointCloud<pcl::PointXYZRGB>);
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_target(new pcl::PointCloud<pcl::PointXYZRGB>);
        
        
        // 將 cout 的輸出流定向到文本檔
        if(printTofile)
            std::cout.rdbuf(outputFile.rdbuf());        
        
        // 讀取點雲資料
        pcl::io::loadPLYFile(src_filename, *cloud_source);
        cout << "Cloud_source[ " << source_num << " ] [" << cloud_source->size() << "] data points to input " << endl;
        pcl::io::loadPLYFile(tar_filename, *cloud_target);
        cout << "Cloud_target[ " << target_num << " ] [" << cloud_target->size() << "] data points to input " << endl;        
        
        // 恢復 cout 的原始輸出流
        std::cout.rdbuf(coutBuffer);        

        // 開啟紀錄 transformation matrix 文檔
        std::ifstream file(matrix_location);
        if (!file.is_open()) {
            std::cout << "Failed to open file." << std::endl;
            return 1;
        }
                
        // 將檔案內容讀取到  transformation matrix
        Eigen::Matrix4f matrix;
        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 4; j++) {
                if (!(file >> matrix(i, j))) {
                    std::cout << "Error reading matrix element at (" << i << ", " << j << ")" << std::endl;
                    file.close();
                    return 1;
                }
            }
        }

        // 關閉檔案
        file.close();

        // 輸出讀取到之內容
        std::cout << "Matrix read from file:" << std::endl;
        std::cout << matrix << std::endl;

        // 執行 transformation 
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr aligned_source(new pcl::PointCloud<pcl::PointXYZRGB>); 
        pcl::transformPointCloud(*cloud_source, *aligned_source, matrix);
        pcl::PLYWriter writer;
        writer.write<pcl::PointXYZRGB>(folderPath + "/aligned_source.ply", *aligned_source, false, false);

        // 將 cloud_target 做 gamma color correction 
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr dark_target(new pcl::PointCloud<pcl::PointXYZRGB>);
        Gamma_color_change(cloud_target, dark_target, 2.5, folderPath);    // gamma < 1.0 調亮; gamma > 1.0 調暗
        writer.write<pcl::PointXYZRGB>(folderPath + "/dark_target.ply", *dark_target, false, false);

        // 建立 correspondence 和 dist
        int* correspond = new int[dark_target->size()] {};   // target point所對應 aligned source point編號
        float* dist = new float[dark_target->size()] {};      // target point所對應 aligned source point距離
        establish_correspondence(dark_target, aligned_source, correspond, dist);        

        // 將 cout 的輸出流定向到文本檔
        if (printTofile)
            std::cout.rdbuf(outputFile.rdbuf());  


        // 直接給數值判斷重疊率
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_overlap(new pcl::PointCloud<pcl::PointXYZRGB>);
        copyPointCloud(*dark_target, *cloud_overlap);
        float overlap_dist = 0.003;
        float overlap_th = 0.45;
        int count = 0;
        for (int i = 0; i < cloud_overlap->size(); i++) {
            if (sqrt(dist[i]) < overlap_dist) {
                cloud_overlap->points[i].r = 255;
                cloud_overlap->points[i].g = 0;
                cloud_overlap->points[i].b = 0;
                count++;
            }
            else {
                cloud_overlap->points[i].r = 0;
                cloud_overlap->points[i].g = 0;
                cloud_overlap->points[i].b = 255;
            }
        }
        float overlap_rt = (float)count / cloud_overlap->size();
        writer.write<pcl::PointXYZRGB>(folderPath + "/thresh_biClass.ply", *cloud_overlap, false, false);

        int* classify;
        double th_low, th_high;
        if (overlap_rt > overlap_th) {
            // th = hresh_triclass_ 法 (by Chung)
            pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_th_triClass_(new pcl::PointCloud<pcl::PointXYZRGB>);
            copyPointCloud(*dark_target, *cloud_th_triClass_);            
            classify = thresh_triclass_(cloud_th_triClass_->size(), dist, folderPath, &th_low, &th_high);
            for (int i = 0; i < cloud_th_triClass_->size(); i++) {
                switch (classify[i]) {
                case 1:
                    cloud_th_triClass_->points[i].r = 173;
                    cloud_th_triClass_->points[i].g = 0;
                    cloud_th_triClass_->points[i].b = 0;
                    break;
                case 2:
                    cloud_th_triClass_->points[i].r = 255;
                    cloud_th_triClass_->points[i].g = 0;
                    cloud_th_triClass_->points[i].b = 0;
                    break;
                case 3:
                    cloud_th_triClass_->points[i].r = 255;
                    cloud_th_triClass_->points[i].g = 179;
                    cloud_th_triClass_->points[i].b = 179;
                    break;
                }
            }
            writer.write<pcl::PointXYZRGB>(folderPath + "/th_triclass_.ply", *cloud_th_triClass_, false, false);
            cloud_th_triClass_->clear();
        }
        else {
            // th = thresh_biclass
            pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_th_biClass(new pcl::PointCloud<pcl::PointXYZRGB>);
            copyPointCloud(*dark_target, *cloud_th_biClass);
            classify = thresh_biclass(dark_target->size(), dist, folderPath, &th_low, &th_high);
            for (int i = 0; i < cloud_th_biClass->size(); i++) {
                switch (classify[i]) {
                case 1:
                    cloud_th_biClass->points[i].r = 255;
                    cloud_th_biClass->points[i].g = 0;
                    cloud_th_biClass->points[i].b = 0;
                    break;
                case 2:
                    cloud_th_biClass->points[i].r = 0;
                    cloud_th_biClass->points[i].g = 0;
                    cloud_th_biClass->points[i].b = 255;
                    break;
                }
            }
            writer.write<pcl::PointXYZRGB>(folderPath + "/th_biClass.ply", *cloud_th_biClass, false, false);
            cloud_th_biClass->clear();
        }
        
        // 恢復 cout 的原始輸出流
        std::cout.rdbuf(coutBuffer);        

        // knn 調色
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr knn_result = knn_color_correction(aligned_source, dark_target);
        writer.write<pcl::PointXYZRGB>(folderPath + "/knn_color_correction.ply", *knn_result, false, false);

        // nn 調色
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr nn_result = direct_color_correction(aligned_source, dark_target, correspond);
        writer.write<pcl::PointXYZRGB>(folderPath + "/nn_color_correction.ply", *nn_result, false, false);

        // Fecker 調色
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr Fecker_result = Fecker_color_correction(aligned_source, dark_target, classify, correspond, folderPath);
        writer.write<pcl::PointXYZRGB>(folderPath + "/Fecker_color_correction.ply", *Fecker_result, false, false);

        // Ding 調色
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr Ding_result = Ding_color_correction(aligned_source, dark_target, classify, correspond, folderPath);
        writer.write<pcl::PointXYZRGB>(folderPath + "/Ding_color_correction.ply", *Ding_result, false, false);

        // AGL 調色
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr Yu_target(new pcl::PointCloud<pcl::PointXYZRGB>);
        copyPointCloud(*dark_target, *Yu_target);
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr Yu_source(new pcl::PointCloud<pcl::PointXYZRGB>);
        copyPointCloud(*aligned_source, *Yu_source);

        Yu_color_correction(Yu_source, Yu_target, classify, correspond, dist, folderPath);
        writer.write<pcl::PointXYZRGB>(folderPath + "/Yu_source.ply", *Yu_source, false, false);
        writer.write<pcl::PointXYZRGB>(folderPath + "/Yu_target.ply", *Yu_target, false, false);
        
        // 將 cout 的輸出流重定向到文本檔
        if (printTofile)
            std::cout.rdbuf(outputFile.rdbuf());        

        // 恢復 cout 的原始輸出流
        std::cout.rdbuf(coutBuffer);        

        pcl::PointCloud<pcl::PointXYZRGB>::Ptr final_result;
        if (overlap_rt > overlap_th) {
            // Ours tri class color correction方法
            final_result = triclass_color_correction_(dark_target, aligned_source, classify, correspond, dist, th_low, th_high, folderPath);
            writer.write<pcl::PointXYZRGB>(folderPath + "/triClass_color_correction.ply", *final_result, false, false);

            // 由於 Ours 計算太費時，只想看其他方法之結果時用複製原target替代 
            /*pcl::PointCloud<pcl::PointXYZRGB>::Ptr triClass_result(new pcl::PointCloud<pcl::PointXYZRGB>);
            copyPointCloud(*dark_target, *triClass_result);*/
        }
        else {
            // Ours bi class color correction 方法
            final_result = biclass_color_correction(dark_target, aligned_source, classify, correspond, dist, th_low, th_high, folderPath);
            writer.write<pcl::PointXYZRGB>(folderPath + "/biClass_color_correction.ply", *final_result, false, false);

            // 由於 Ours 計算太費時，只想看其他方法之結果時用複製原target替代 
            /*pcl::PointCloud<pcl::PointXYZRGB>::Ptr biClass_result(new pcl::PointCloud<pcl::PointXYZRGB>);
              copyPointCloud(*dark_target, *biClass_result);*/
        }
        
        // 將 cout 的輸出流重定向到文本檔
        if (printTofile)
            std::cout.rdbuf(outputFile.rdbuf());        

        double* MV;
        MV = Mean_Varience_color(aligned_source, cloud_target);
        cout << "[origin]Mean dif= " << *(MV + 3 * 2 + 0) << endl;
        cout << "[origin]Variance dif = " << *(MV + 3 * 2 + 1) << endl;        

        MV = Mean_Varience_color(aligned_source, dark_target);
        cout << "[dark]Mean dif= " << *(MV + 3 * 2 + 0) << endl;
        cout << "[dark]Variance dif = " << *(MV + 3 * 2 + 1) << endl;        

        MV = Mean_Varience_color(aligned_source, knn_result);
        cout << "[knn]Mean dif= " << *(MV + 3 * 2 + 0) << endl;
        cout << "[knn]Variance dif = " << *(MV + 3 * 2 + 1) << endl; 

        MV = Mean_Varience_color(aligned_source, nn_result);
        cout << "[nn]Mean dif = " << *(MV + 3 * 2 + 0) << endl;
        cout << "[nn]Variance dif = " << *(MV + 3 * 2 + 1) << endl;

        MV = Mean_Varience_color(aligned_source, Fecker_result);
        cout << "[Fecker]Mean dif = " << *(MV + 3 * 2 + 0) << endl;
        cout << "[Fecker]Variance dif = " << *(MV + 3 * 2 + 1) << endl;

        MV = Mean_Varience_color(aligned_source, Ding_result);
        cout << "[Ding]Mean dif = " << *(MV + 3 * 2 + 0) << endl;
        cout << "[Ding]Variance dif = " << *(MV + 3 * 2 + 1) << endl;

        MV = Mean_Varience_color(Yu_source, Yu_target);
        cout << "[Yu]Mean dif = " << *(MV + 3 * 2 + 0) << endl;
        cout << "[Yu]Variance dif = " << *(MV + 3 * 2 + 1) << endl;

        MV = Mean_Varience_color(aligned_source, final_result);
        cout << "[triClass]Mean dif = " << *(MV + 3 * 2 + 0) << endl;
        cout << "[triClass]Variance dif = " << *(MV + 3 * 2 + 1) << endl;             
        
        // 恢復 cout 的原始輸出流
        std::cout.rdbuf(coutBuffer);        

        cout << "---------------------------" << endl;
        cout << "Finish : " << source_num  << " & " << target_num << endl;
        cout << "---------------------------\n" << endl;
    }          
    return 0;
}
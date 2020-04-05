#include <iostream>

#include <math.h>

#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

inline float histogram(int bin, int value) {
    return bin == value? 1 : 0;
}

template<float (*windowFunction)(int, int), int size>
inline float parzenEstimator(unsigned char* If, unsigned char* Im, int i_bin, int k_bin) {
    float sum = 0;

    for(int i = 0; i < size; ++i) {
        sum += windowFunction(i_bin, If[i]) * windowFunction(k_bin, Im[i]);
    }

    return sum;
}

template<float (*windowFunction)(int, int), int size>
float mutualInformation(unsigned char* If, unsigned char* Im) {
    float sum = 0;

    float normalizationFactor = 0;
    float estimators[256][256];
    float partial_i_estimators[256] = {0};
    float partial_k_estimators[256] = {0};

    for(int i = 0; i < 256; ++i) {
        for(int k = 0; k < 256; ++k) {
            estimators[i][k] = parzenEstimator<windowFunction, size>(If, Im, i, k);
            normalizationFactor += estimators[i][k];
        }
    }

    for(int i = 0; i < 256; ++i) {
        for (int k = 0; k < 256; ++k) {
            estimators[i][k] /= normalizationFactor;
            partial_i_estimators[i] += estimators[i][k];
            partial_k_estimators[k] += estimators[i][k];
        }
    }

    for(int i = 0; i < 256; ++i) {
        float partProb_i = partial_i_estimators[i];
        for(int k = 0; k < 256; ++k) {
            float partProb_k = partial_k_estimators[k];
            if(partProb_k != 0) {
                float prob_ik = estimators[i][k];
                    if(prob_ik != 0) {
                        sum += prob_ik * log2(prob_ik / (partProb_i * partProb_k));
                    }
            }
        }
    }

    return sum;
}

int main(int argc, char **argv) {
    cv::Mat img = imread("rock_landscape.jpg", cv::IMREAD_GRAYSCALE);

    std::cout << mutualInformation<histogram, 1000>(img.data, img.data) << std::endl;

    return 0;
}



#include <iostream>

#include <math.h>

#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

inline float windowFunction(int bin, int value) {
    return bin == value? 1 : 0;
}

inline float parzenEstimator(unsigned char* If, unsigned char* Im, int i_bin, int k_bin, int size) {
    float sum = 0;

    for(int i = 0; i < size; ++i) {
        sum += windowFunction(i_bin, If[i]) * windowFunction(k_bin, Im[i]);
    }

    return sum;
}

float mutualInformation(unsigned char* If, unsigned char* Im, int size) {
    float sum = 0;

    float normalizationFactor = 0;
    float estimators[256][256];
    float partial_i_estimators[256] = {0};
    float partial_k_estimators[256] = {0};

    #pragma omp parallel for
    for(int i = 0; i < 256; ++i) {
        for(int k = 0; k < 256; ++k) {
            estimators[i][k] = parzenEstimator(If, Im, i, k, size);
            normalizationFactor += estimators[i][k];
        }
    }

    #pragma omp parallel for
    for(int i = 0; i < 256; ++i) {
        for (int k = 0; k < 256; ++k) {
            estimators[i][k] /= normalizationFactor;
            partial_i_estimators[i] += estimators[i][k];
            partial_k_estimators[k] += estimators[i][k];
        }
    }

    #pragma omp parallel for
    for(int i = 0; i < 256; ++i) {
        float partProb_i = partial_i_estimators[i];
        if(partProb_i != 0) {
            for(int k = 0; k < 256; ++k) {
                float partProb_k = partial_k_estimators[k];
                if(partProb_k != 0) {
                    float prob = estimators[i][k];
                        if(prob != 0) {
                            sum += prob * log2(prob / (partProb_i * partProb_k));
                        }
                }
            }
        }
    }

    return -sum;
}

float simpleMutualInformation(unsigned char* If, unsigned char* Im, int size) {
    float estimators[256][256] = {0},
        partial_i_estimators[256] = {0},
        partial_k_estimators[256] = {0};

    #pragma omp parallel for
    for(int i = 0; i < size; ++i) {
        estimators[If[i]][Im[i]]++;
        partial_i_estimators[If[i]]++;
        partial_k_estimators[Im[i]]++;
    }

    float sum = 0;

    #pragma omp parallel for
    for(int i = 0; i < 256; ++i) {
        float partProb_i = partial_i_estimators[i];
        if (partProb_i != 0) {
            for (int k = 0; k < 256; ++k) {
                float partProb_k = partial_k_estimators[k];
                if (partProb_k != 0) {
                    float prob = estimators[i][k];
                    if (prob != 0) {
                        sum += prob * log2(prob / (partProb_i * partProb_k));
                    }
                }
            }
        }
    }

    return -sum;
}


int main(int argc, char **argv) {
    cv::Mat img = imread("rock_landscape.jpg", cv::IMREAD_GRAYSCALE);

    std::cout << mutualInformation(img.data, img.data, 100000) << std::endl;

    return 0;
}



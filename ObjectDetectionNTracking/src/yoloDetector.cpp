#include <iostream>
#include <vector>
#include <string>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <onnxruntime_cxx_api.h>
namespace py = pybind11;

const std::vector<int> TARGET_CLASSES = {0, 2, 3, 5, 7};

struct Detection {
    int classId;
    float confidence;
    std::vector<int> box; // [x, y, width, height]
};

class YOLOv8Detector {
private:
    Ort::Env env;
    Ort::SessionOptions session_options;
    std::unique_ptr<Ort::Session> session;
    OrtCUDAProviderOptions cuda_options;
    const int targetWidth = 640;
    const int targetHeight = 640;
    const float confThreshold = 0.45f;
    const float nmsThreshold = 0.5f;
    std::vector<const char*> inputNames = {"images"};
    std::vector<const char*> outputNames = {"output0"};


    // Convert NumPy ndarray (RGB/BGR image from Python) to cv::Mat
    cv::Mat numpyToCvMat(py::array_t<uint8_t>& input) {
        py::buffer_info buf = input.request();
        if (buf.ndim != 3) {
            throw std::runtime_error("Expected 3D image array (H, W, C)");
        }
        return cv::Mat(buf.shape[0], buf.shape[1], CV_8UC3, buf.ptr);
    }

public:
    YOLOv8Detector(bool useCUDA = false)
    {
        env = Ort::Env(ORT_LOGGING_LEVEL_WARNING, "YOLOv8Pipeline");

        session_options.SetIntraOpNumThreads(4);
        session_options.SetExecutionMode(ExecutionMode::ORT_PARALLEL);

        if (useCUDA) {
            session_options.AppendExecutionProvider_CUDA(cuda_options);
        }
    }

    bool loadModel(const std::string& modelPath) {
        try {
            //Load model
            std::string tempPath = modelPath + "yolov8n.onnx";
            session = std::make_unique<Ort::Session>(env, tempPath.c_str(), session_options);
        } catch (const Ort::Exception& e) {
            std::cerr << "Failed to load model: " << e.what() << std::endl;
            return false;
        }

        return true;
    }

    std::vector<Detection> detect(py::array_t<uint8_t> inputImage, 
                              float confThreshold = 0.45f, 
                              float nmsThreshold = 0.5f) 
    {
    
        cv::Mat input = numpyToCvMat(inputImage);

        // 1. Calculate aspect-ratio preserving scale
        int w = input.cols;
        int h = input.rows;
        float scale = std::min((float)targetWidth / w, (float)targetHeight / h);
        
        int newW = std::round(w * scale);
        int newH = std::round(h * scale);

        // 2. Resize maintaining aspect ratio
        cv::Mat resized;
        cv::resize(input, resized, cv::Size(newW, newH));

        // 3. Create canvas with letterbox padding (gray or black background)
        int padW = (targetWidth - newW) / 2;
        int padH = (targetHeight - newH) / 2;
        
        cv::Mat padded(targetHeight, targetWidth, CV_8UC3, cv::Scalar(114, 114, 114));
        resized.copyTo(padded(cv::Rect(padW, padH, newW, newH)));

        // 4. Create blob from padded image (DO NOT pass cv::Size(640, 640) here to avoid re-scaling)
        cv::Mat blob;
        cv::dnn::blobFromImage(padded, blob, 1.0 / 255.0, cv::Size(), cv::Scalar(), true, false);

        std::vector<int64_t> inputShape = {1, 3, targetHeight, targetWidth};
        Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

        Ort::Value inputTensor = Ort::Value::CreateTensor<float>(
            memoryInfo, 
            reinterpret_cast<float*>(blob.data), 
            blob.total(),
            inputShape.data(), 
            inputShape.size()
        );

        // Run Inference
        auto outputTensors = session->Run(
            Ort::RunOptions{nullptr}, 
            inputNames.data(), &inputTensor, 1, 
            outputNames.data(), 1);

        float* rawOutput = outputTensors[0].GetTensorMutableData<float>();
        auto outputShape = outputTensors[0].GetTensorTypeAndShapeInfo().GetShape(); 

        int numAttributes = outputShape[1]; // 84
        int numAnchors = outputShape[2];    // 8400

        std::vector<int> classIds;
        std::vector<float> confidences;
        std::vector<cv::Rect> boxes;

        // Postprocessing
        for (int i = 0; i < numAnchors; ++i) {
            float maxScore = -1.0f;
            int classId = -1;
            for (int c = 0; c < numAttributes - 4; ++c) {
                float score = rawOutput[(4 + c) * numAnchors + i];
                if (score > maxScore) {
                    maxScore = score;
                    classId = c;
                }
            }

            if (maxScore >= confThreshold) {
                float cx = rawOutput[0 * numAnchors + i];
                float cy = rawOutput[1 * numAnchors + i];
                float w_box = rawOutput[2 * numAnchors + i];
                float h_box = rawOutput[3 * numAnchors + i];

                // Multiply by input size if model outputs normalized coordinates [0, 1]
                if (cx <= 1.0f && cy <= 1.0f && w_box <= 1.0f && h_box <= 1.0f) {
                    cx *= targetWidth;
                    cy *= targetHeight;
                    w_box *= targetWidth;
                    h_box *= targetHeight;
                }

                // Scale coordinates back to original unpadded image
                int left = static_cast<int>((cx - padW - 0.5f * w_box) / scale);
                int top  = static_cast<int>((cy - padH - 0.5f * h_box) / scale);
                int width = static_cast<int>(w_box / scale);
                int height = static_cast<int>(h_box / scale);

                // Clamp coordinates to image boundary
                left = std::max(0, std::min(left, w - 1));
                top  = std::max(0, std::min(top, h - 1));
                width = std::max(1, std::min(width, w - left));
                height = std::max(1, std::min(height, h - top));

                boxes.push_back(cv::Rect(left, top, width, height));
                confidences.push_back(maxScore);
                classIds.push_back(classId);
            }
        }

        std::vector<int> indices;
        cv::dnn::NMSBoxes(boxes, confidences, confThreshold, nmsThreshold, indices);

        std::vector<Detection> results;
        for (int idx : indices) {
            cv::Rect b = boxes[idx];
            results.push_back({classIds[idx], confidences[idx], {b.x, b.y, b.width, b.height}});
        }

        return results;
    }
};

// --- Pybind11 Module Binding ---
PYBIND11_MODULE(yoloDetector, m) {
    m.doc() = "C++ OpenCV YOLOv8 Detector Pybind11 Module";

    py::class_<Detection>(m, "Detection")
        .def_readonly("class_id", &Detection::classId)
        .def_readonly("confidence", &Detection::confidence)
        .def_readonly("box", &Detection::box);

    py::class_<YOLOv8Detector>(m, "YOLOv8Detector")
        .def(py::init<bool>(), py::arg("use_cuda") = false)
        .def("detect", &YOLOv8Detector::detect, 
             py::arg("image"), 
             py::arg("conf_threshold") = 0.45f, 
             py::arg("nms_threshold") = 0.5f)
        .def("load_model", &YOLOv8Detector::loadModel,
             py::arg("model_path"));
}
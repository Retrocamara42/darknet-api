#include <std_micro_service.hpp>
#include "api-controller.h"
#include "yolo_detector_class.hpp"
#include <opencv2/opencv.hpp>
#include <opencv2/imgcodecs.hpp>

using namespace web;
using namespace http;

std::vector<det_obj> detect(std::string names_file, std::string cfg_file, std::string weights_file, cv::Mat mat_img, float thresh);

void MicroserviceController::initRestOpHandlers() {
   _listener.support(methods::GET, std::bind(&MicroserviceController::handleGet, this, std::placeholders::_1));
    _listener.support(methods::POST, std::bind(&MicroserviceController::handlePost, this, std::placeholders::_1));
}

void MicroserviceController::handleGet(http_request message) {
    auto path = requestPath(message);
    if (!path.empty()) {
        if (path[0] == "test") {
            auto response = json::value::object();
            response["version"] = json::value::string("0.1.1");
            response["status"] = json::value::string("ready!");
            message.reply(status_codes::OK, response);
        }
    }
    else {
        message.reply(status_codes::NotFound);
    }
}

void MicroserviceController::handlePost(http_request message) {
    auto path = requestPath(message);
    if (!path.empty()) {
        if (path[0] == "tinyyolo") {
            message.extract_vector().then([=](std::vector<unsigned char> request) {
                  try {
                     //unsigned char * buf = reinterpret_cast<unsigned char*>(request.data());
                     cv::Mat img = cv::imdecode(request, cv::IMREAD_COLOR);
                     std::vector<det_obj> obj_det = detect("resources/epp.names", "resources/epp-yolo3.cfg", "epp-yolo3_100000.weights", img, 0.2);
                     auto response = json::value::object();
                     int j = 1;
                     for (auto &i : obj_det) {
                       response["version"] = json::value::string("1.0");
                       response["object"+std::to_string(j)+".name"] = json::value::string(i.name);
                       response["object"+std::to_string(j)+".x"] = json::value::number(i.x);
                       response["object"+std::to_string(j)+".y"] = json::value::number(i.y);
                       response["object"+std::to_string(j)+".h"] = json::value::number(i.h);
                       response["object"+std::to_string(j)+".w"] = json::value::number(i.w);
                       response["object"+std::to_string(j)+".prob"] = json::value::number(i.prob);
                       j++;
                    }
                     message.reply(status_codes::OK, response);
                  }
                  catch(json::json_exception & e) {
                     message.reply(status_codes::BadRequest);
                  }
            });

        }
        else {
            message.reply(status_codes::NotFound);
        }
    }
    else {
        message.reply(status_codes::NotFound);
    }
}

void MicroserviceController::handlePatch(http_request message) {
    message.reply(status_codes::NotImplemented, responseNotImpl(methods::PATCH));
}

void MicroserviceController::handlePut(http_request message) {
    message.reply(status_codes::NotImplemented, responseNotImpl(methods::PUT));
}

void MicroserviceController::handleDelete(http_request message) {
    message.reply(status_codes::NotImplemented, responseNotImpl(methods::DEL));
}

void MicroserviceController::handleHead(http_request message) {
    message.reply(status_codes::NotImplemented, responseNotImpl(methods::HEAD));
}

void MicroserviceController::handleOptions(http_request message) {
    message.reply(status_codes::NotImplemented, responseNotImpl(methods::OPTIONS));
}

void MicroserviceController::handleTrace(http_request message) {
    message.reply(status_codes::NotImplemented, responseNotImpl(methods::TRCE));
}

void MicroserviceController::handleConnect(http_request message) {
    message.reply(status_codes::NotImplemented, responseNotImpl(methods::CONNECT));
}

void MicroserviceController::handleMerge(http_request message) {
    message.reply(status_codes::NotImplemented, responseNotImpl(methods::MERGE));
}

json::value MicroserviceController::responseNotImpl(const http::method & method) {
    auto response = json::value::object();
    response["serviceName"] = json::value::string("C++ Mircroservice Sample");
    response["http_method"] = json::value::string(method);
    return response ;
}

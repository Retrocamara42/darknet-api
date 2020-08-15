#include <iostream>

#include <usr_interrupt_handler.hpp>
//#include <runtime_utils.hpp>
#include "api-controller.h"

using namespace web;

int main(int argc, const char * argv[]) {
    //InterruptHandler::hookSIGINT();

    MicroserviceController server;
    server.setEndpoint("http://host_auto_ip4:5001/v1/api/");

    try {
        server.accept().wait();
        std::cout << "Computer vision service listening at: " << server.endpoint() << '\n';
        InterruptHandler::waitForUserInterrupt();
        server.shutdown().wait();
    }
    catch(std::exception & e) {
        std::cerr << "Error ocurred: " << e.what() << '\n';
    }
    catch(...) {
        std::cerr << "Unknown failure occurred." << std::endl;
    }

    return 0;
}

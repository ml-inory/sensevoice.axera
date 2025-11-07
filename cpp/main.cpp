#include <stdio.h>
#include <librosa/librosa.h>
#include <vector>
#include <limits>
#include <algorithm>
#include <fstream>
#include <ax_sys_api.h>
#include <unordered_map>
#include <ctime>
#include <sys/time.h>

#include <ax_sys_api.h>

#include "cmdline.hpp"
#include "AudioFile.h"
#include "SenseVoice.hpp"


static double get_current_time()
{
    struct timeval tv;
    gettimeofday(&tv, NULL);

    return tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
}

int main(int argc, char** argv) {
    cmdline::parser cmd;
    cmd.add<std::string>("wav", 'w', "wav file", false, "../wav/BAC009S0764W0121.wav");
    cmd.add<std::string>("model", 'm', "axmodel path", false, "../models/SenseVoice/sensevoice_ax650/sensevoice.axmodel");
    cmd.add<std::string>("token", 't', "tokens.txt", false, "./tokens.txt");
    cmd.parse_check(argc, argv);

    // 0. get app args, can be removed from user's app
    auto wav_file = cmd.get<std::string>("wav");
    auto model_path = cmd.get<std::string>("model");
    auto token_txt = cmd.get<std::string>("token");

    int ret = AX_SYS_Init();
    if (0 != ret) {
        fprintf(stderr, "AX_SYS_Init failed! ret = 0x%x\n", ret);
        return -1;
    }

    AX_ENGINE_NPU_ATTR_T npu_attr;
    memset(&npu_attr, 0, sizeof(npu_attr));
    npu_attr.eHardMode = static_cast<AX_ENGINE_NPU_MODE_T>(0);
    ret = AX_ENGINE_Init(&npu_attr);
    if (0 != ret) {
        fprintf(stderr, "Init ax-engine failed{0x%8x}.\n", ret);
        return -1;
    }

    SenseVoice sense_voice(model_path, token_txt);

    return 0;
}
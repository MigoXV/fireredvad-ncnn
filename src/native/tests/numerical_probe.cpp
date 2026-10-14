// Internal numeric verification executable; not installed and no public ABI.
#include "../src/engine.cpp"
#include <iostream>
int main(int argc, char **argv) {
    try {
        if (argc != 3)
            return 2;
        std::ifstream input(argv[2]);
        nlohmann::json requests;
        input >> requests;
        frv_engine_config c;
        frv_engine_config_init(&c);
        std::ifstream meta(std::string(argv[1]) + "/metadata.json");
        nlohmann::json m;
        meta >> m;
        c.chunk_frames = m.at("chunk_frames").get<uint32_t>();
        frv::Engine engine(argv[1], c);
        frv_post_config pc;
        frv_post_config_init(&pc);
        frv::Session s(pc, false);
        for (const auto &request : requests) {
            std::vector<float> features;
            if (request.is_object()) {
                frv::Task task;
                task.session = std::make_shared<frv::Session>(pc, false);
                task.audio = request.at("audio").get<std::vector<float>>();
                task.rate = request.value("rate", 16000u);
                engine.prepare(task);
                features = std::move(task.features);
            } else
                features = request.get<std::vector<float>>();
            if (features.size() % 80)
                return 2;
            std::vector<float> probs;
            for (size_t begin = 0; begin < features.size() / 80; begin += c.chunk_frames) {
                auto p =
                    engine.model(*engine.nets[0], features, begin,
                                 std::min(size_t(c.chunk_frames), features.size() / 80 - begin), s);
                probs.insert(probs.end(), p.begin(), p.end());
            }
            std::cout << nlohmann::json{{"features", features},
                                        {"probs", probs},
                                        {"caches", s.caches}}
                             .dump()
                      << '\n';
        }
    } catch (const std::exception &e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}

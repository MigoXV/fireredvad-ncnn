#include "postprocessor.h"
#include <stdexcept>
void expect(bool b) {
    if (!b)
        throw std::runtime_error("postprocessor invariant failed");
}
int main() {
    frv_post_config c{sizeof(c), FRV_ABI_VERSION, 1, 3, 3, 20, 4, 30000, 0.5};
    frv::Postprocessor p(c);
    for (int i = 0; i < 12; ++i) {
        auto r = p.process(0);
        expect(r.start < 0 && r.end < 0);
    }
    expect(p.confirmed() == 9);
    p.process(1);
    p.process(1);
    auto r = p.process(1);
    expect(r.start == 10);
    p.process(0);
    p.process(0);
    p.process(0);
    r = p.process(0);
    expect(r.end == 19);
    expect(p.confirmed() == 19);
    frv::Publisher publisher;
    publisher.endpoint_only = true;
    std::vector<frv_event> events;
    publisher.publish({0.1, -1, 0.15}, events);
    expect(events.back().processed_time == 0.1);
    publisher.publish({-1, 0.2, 0.2}, events);
    expect(events[events.size() - 2].start == 0.1 && events.back().processed_time == 0.2);
    frv::Postprocessor trunc(c);
    for (int i = 0; i < 20; ++i)
        r = trunc.process(1);
    expect(r.end == 20);
    r = trunc.process(1);
    expect(r.start == 21);
}

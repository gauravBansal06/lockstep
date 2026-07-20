// Concurrent stateful service — C++ (std::thread + std::mutex + POSIX sockets). See ../SPEC.md.
#include <cstdio>
#include <cstring>
#include <cstdint>
#include <vector>
#include <thread>
#include <mutex>
#include <unistd.h>
#include <arpa/inet.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <fcntl.h>
static const int N = 16, START = 150, CLIENTS = 6, PERCLIENT = 500, SNAPIVL = 137, PORT = 9000;
static int64_t bal[N];
static int applied = 0, failed = 0;
static std::mutex mtx;
static int diskfd = -1;

static void snapshot() {
    if (diskfd < 0) return;
    unsigned char buf[512] = {0};
    *(uint32_t *)(buf) = 0x53464244u;
    *(uint32_t *)(buf + 4) = (uint32_t)applied;
    for (int i = 0; i < N; i++) *(int64_t *)(buf + 8 + i * 8) = bal[i];
    pwrite(diskfd, buf, 512, 0);
    fsync(diskfd);
}
static bool apply(int frm, int to, int64_t amt) {
    std::lock_guard<std::mutex> g(mtx);
    if (bal[frm] >= amt) {
        bal[frm] -= amt; bal[to] += amt; applied++;
        if (applied % SNAPIVL == 0) snapshot();
        return true;
    }
    failed++;
    return false;
}
static int readline(int fd, char *buf, int max) {
    int n = 0; char c;
    while (n < max - 1) {
        int k = read(fd, &c, 1);
        if (k <= 0) return n > 0 ? n : -1;
        if (c == '\n') break;
        buf[n++] = c;
    }
    buf[n] = 0;
    return n;
}
static void handle(int fd) {
    char line[128];
    while (readline(fd, line, sizeof line) >= 0) {
        int frm, to; long amt;
        if (sscanf(line, "%d %d %ld", &frm, &to, &amt) != 3) break;
        if (write(fd, apply(frm, to, amt) ? "1\n" : "0\n", 2) != 2) break;
    }
    close(fd);
}
static void client(long idx) {
    int64_t state = (idx + 1) * 1000 + 1;
    int fd = socket(AF_INET, SOCK_STREAM, 0);
    sockaddr_in a{}; a.sin_family = AF_INET; a.sin_port = htons(PORT); a.sin_addr.s_addr = htonl(0x7f000001);
    if (connect(fd, (sockaddr *)&a, sizeof a) < 0) { close(fd); return; }
    char line[128];
    for (int i = 0; i < PERCLIENT; i++) {
        state = (state * 16807) % 2147483647; int64_t frm = state % N;
        state = (state * 16807) % 2147483647; int64_t to = state % N;
        state = (state * 16807) % 2147483647; int64_t amt = state % 90 + 10;
        int len = snprintf(line, sizeof line, "%ld %ld %ld\n", (long)frm, (long)to, (long)amt);
        if (write(fd, line, len) != len) break;
        if (readline(fd, line, sizeof line) < 0) break;
    }
    close(fd);
}
int main() {
    printf("CSVC-START cpp\n"); fflush(stdout);
    diskfd = open("/dev/vda", O_RDWR);
    for (int i = 0; i < N; i++) bal[i] = START;
    int ls = socket(AF_INET, SOCK_STREAM, 0);
    int one = 1; setsockopt(ls, SOL_SOCKET, SO_REUSEADDR, &one, sizeof one);
    sockaddr_in a{}; a.sin_family = AF_INET; a.sin_port = htons(PORT); a.sin_addr.s_addr = htonl(0x7f000001);
    if (bind(ls, (sockaddr *)&a, sizeof a) < 0) { printf("CSVC-LISTEN-ERR\n"); return 1; }
    listen(ls, 16);
    std::thread([ls] {
        for (;;) {
            int fd = accept(ls, nullptr, nullptr);
            if (fd < 0) break;
            std::thread(handle, fd).detach();
        }
    }).detach();
    std::vector<std::thread> cl;
    for (long c = 0; c < CLIENTS; c++) cl.emplace_back(client, c);
    for (auto &t : cl) t.join();
    snapshot();
    int64_t total = 0;
    for (int i = 0; i < N; i++) total += bal[i];
    unsigned char mb[N * 8];
    for (int i = 0; i < N; i++) *(int64_t *)(mb + i * 8) = bal[i];
    uint64_t h = 0xcbf29ce484222325ull;
    for (int i = 0; i < N * 8; i++) h = (h ^ mb[i]) * 0x100000001b3ull;
    int64_t disktotal = -1;
    if (diskfd >= 0) {
        unsigned char rb[512];
        if (pread(diskfd, rb, 512, 0) == 512 && *(uint32_t *)rb == 0x53464244u) {
            disktotal = 0;
            for (int i = 0; i < N; i++) disktotal += *(int64_t *)(rb + 8 + i * 8);
        }
    }
    printf("CSVC-APPLIED %d CSVC-FAILED %d\n", applied, failed);
    printf("INV-CONSERVED %s (total=%ld want=2400)\n", total == N * START ? "true" : "false", (long)total);
    printf("INV-DISK-MATCH %s (diskTotal=%ld)\n", disktotal == total ? "true" : "false", (long)disktotal);
    printf("CSVC-HASH %016lx\n", (unsigned long)h);
    printf("CSVC-EXIT 0\n");
    return 0;
}

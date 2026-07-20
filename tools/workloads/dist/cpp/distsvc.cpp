// Distributed bank-service — C++ (std::thread + std::mutex + POSIX sockets), the
// concurrent svc.cpp SPLIT into two roles over a real network. See ../c/distsvc.c.
//   distsvc server            → node B: bind 0.0.0.0:9000, apply transfers, invariant
//   distsvc client <peer_ip>  → node A: 6 workers × 500 requests to the server
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
static const int N = 16, START = 150, CLIENTS = 6, PERCLIENT = 500, PORT = 9000;
static int64_t bal[N];
static int applied = 0, failed = 0, done_conns = 0;
static std::mutex mtx;
static char peer[64];

static bool apply(int frm, int to, int64_t amt) {
    std::lock_guard<std::mutex> g(mtx);
    if (bal[frm] >= amt) { bal[frm] -= amt; bal[to] += amt; applied++; return true; }
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
    { std::lock_guard<std::mutex> g(mtx); done_conns++; }
}
static void run_server() {
    printf("CSVC-START cpp-server\n"); fflush(stdout);
    for (int i = 0; i < N; i++) bal[i] = START;
    int ls = socket(AF_INET, SOCK_STREAM, 0);
    int one = 1; setsockopt(ls, SOL_SOCKET, SO_REUSEADDR, &one, sizeof one);
    sockaddr_in a{}; a.sin_family = AF_INET; a.sin_port = htons(PORT); a.sin_addr.s_addr = htonl(INADDR_ANY);
    if (bind(ls, (sockaddr *)&a, sizeof a) < 0) { printf("CSVC-LISTEN-ERR\n"); return; }
    listen(ls, 16);
    printf("CSVC-LISTEN 9000\n"); fflush(stdout);
    std::thread([ls] {
        for (;;) {
            int fd = accept(ls, nullptr, nullptr);
            if (fd < 0) break;
            std::thread(handle, fd).detach();
        }
    }).detach();
    for (int i = 0; i < 4000; i++) {   // ~40s virtual-time cap so a partition can't hang
        mtx.lock(); int d = done_conns; mtx.unlock();
        if (d >= CLIENTS) break;
        usleep(10000);
    }
    int64_t total = 0;
    for (int i = 0; i < N; i++) total += bal[i];
    unsigned char mb[N * 8];
    for (int i = 0; i < N; i++) *(int64_t *)(mb + i * 8) = bal[i];
    uint64_t h = 0xcbf29ce484222325ull;
    for (int i = 0; i < N * 8; i++) h = (h ^ mb[i]) * 0x100000001b3ull;
    printf("CSVC-APPLIED %d CSVC-FAILED %d\n", applied, failed);
    printf("INV-CONSERVED %s (total=%ld want=2400)\n", total == N * START ? "true" : "false", (long)total);
    printf("CSVC-HASH %016lx\n", (unsigned long)h);
    printf("CSVC-EXIT 0\n"); fflush(stdout);
}
static void client(long idx) {
    int64_t state = (idx + 1) * 1000 + 1;
    sockaddr_in a{}; a.sin_family = AF_INET; a.sin_port = htons(PORT); inet_pton(AF_INET, peer, &a.sin_addr);
    int fd = -1, ok = 0;
    for (int r = 0; r < 80; r++) {   // the server may still be booting
        fd = socket(AF_INET, SOCK_STREAM, 0);
        if (connect(fd, (sockaddr *)&a, sizeof a) == 0) { ok = 1; break; }
        close(fd); usleep(100000);
    }
    if (!ok) return;
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
static void run_client() {
    printf("CSVC-START cpp-client\n"); fflush(stdout);
    std::vector<std::thread> cl;
    for (long c = 0; c < CLIENTS; c++) cl.emplace_back(client, c);
    for (auto &t : cl) t.join();
    printf("CSVC-CLIENT-DONE\n"); fflush(stdout);
}
int main(int argc, char **argv) {
    if (argc >= 2 && strcmp(argv[1], "server") == 0) run_server();
    else if (argc >= 3 && strcmp(argv[1], "client") == 0) { strncpy(peer, argv[2], sizeof peer - 1); run_client(); }
    else { printf("usage: distsvc server | distsvc client <peer_ip>\n"); return 1; }
    return 0;
}

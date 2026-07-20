/* Concurrent stateful service — C (POSIX sockets + pthreads + mutex). See ../SPEC.md. */
#define _GNU_SOURCE
#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include <pthread.h>
#include <arpa/inet.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <fcntl.h>
#define N 16
#define START 150
#define CLIENTS 6
#define PERCLIENT 500
#define SNAPIVL 137
#define PORT 9000
static long bal[N];
static int applied = 0, failed = 0;
static pthread_mutex_t lk = PTHREAD_MUTEX_INITIALIZER;
static int diskfd = -1;

static void snapshot(void) {
    if (diskfd < 0) return;
    unsigned char buf[512] = {0};
    *(unsigned int *)(buf) = 0x53464244;
    *(unsigned int *)(buf + 4) = applied;
    for (int i = 0; i < N; i++) *(long *)(buf + 8 + i * 8) = bal[i];
    pwrite(diskfd, buf, 512, 0);
    fsync(diskfd);
}
static int apply(int frm, int to, long amt) {
    pthread_mutex_lock(&lk);
    int r;
    if (bal[frm] >= amt) {
        bal[frm] -= amt; bal[to] += amt; applied++;
        if (applied % SNAPIVL == 0) snapshot();
        r = 1;
    } else { failed++; r = 0; }
    pthread_mutex_unlock(&lk);
    return r;
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
static void *handle(void *arg) {
    int fd = (int)(long)arg; char line[128];
    while (readline(fd, line, sizeof line) >= 0) {
        int frm, to; long amt;
        if (sscanf(line, "%d %d %ld", &frm, &to, &amt) != 3) break;
        if (write(fd, apply(frm, to, amt) ? "1\n" : "0\n", 2) != 2) break;
    }
    close(fd);
    return NULL;
}
static void *server(void *arg) {
    int ls = (int)(long)arg;
    for (;;) {
        int fd = accept(ls, NULL, NULL);
        if (fd < 0) break;
        pthread_t t;
        pthread_create(&t, NULL, handle, (void *)(long)fd);
        pthread_detach(t);
    }
    return NULL;
}
static void *client(void *arg) {
    long idx = (long)arg, state = (idx + 1) * 1000 + 1;
    int fd = socket(AF_INET, SOCK_STREAM, 0);
    struct sockaddr_in a = {0};
    a.sin_family = AF_INET; a.sin_port = htons(PORT); a.sin_addr.s_addr = htonl(0x7f000001);
    if (connect(fd, (void *)&a, sizeof a) < 0) { close(fd); return NULL; }
    char line[128];
    for (int i = 0; i < PERCLIENT; i++) {
        state = (state * 16807) % 2147483647; long frm = state % N;
        state = (state * 16807) % 2147483647; long to = state % N;
        state = (state * 16807) % 2147483647; long amt = state % 90 + 10;
        int len = snprintf(line, sizeof line, "%ld %ld %ld\n", frm, to, amt);
        if (write(fd, line, len) != len) break;
        if (readline(fd, line, sizeof line) < 0) break;
    }
    close(fd);
    return NULL;
}
int main(void) {
    printf("CSVC-START c\n"); fflush(stdout);
    diskfd = open("/dev/vda", O_RDWR);
    for (int i = 0; i < N; i++) bal[i] = START;
    int ls = socket(AF_INET, SOCK_STREAM, 0);
    int one = 1; setsockopt(ls, SOL_SOCKET, SO_REUSEADDR, &one, sizeof one);
    struct sockaddr_in a = {0};
    a.sin_family = AF_INET; a.sin_port = htons(PORT); a.sin_addr.s_addr = htonl(0x7f000001);
    if (bind(ls, (void *)&a, sizeof a) < 0) { printf("CSVC-LISTEN-ERR\n"); return 1; }
    listen(ls, 16);
    pthread_t srv;
    pthread_create(&srv, NULL, server, (void *)(long)ls);
    pthread_detach(srv);
    pthread_t cl[CLIENTS];
    for (long c = 0; c < CLIENTS; c++) pthread_create(&cl[c], NULL, client, (void *)c);
    for (int c = 0; c < CLIENTS; c++) pthread_join(cl[c], NULL);
    snapshot();
    long total = 0;
    for (int i = 0; i < N; i++) total += bal[i];
    unsigned char mb[N * 8];
    for (int i = 0; i < N; i++) *(long *)(mb + i * 8) = bal[i];
    unsigned long h = 0xcbf29ce484222325UL;
    for (int i = 0; i < N * 8; i++) h = (h ^ mb[i]) * 0x100000001b3UL;
    long disktotal = -1;
    if (diskfd >= 0) {
        unsigned char rb[512];
        if (pread(diskfd, rb, 512, 0) == 512 && *(unsigned int *)rb == 0x53464244) {
            disktotal = 0;
            for (int i = 0; i < N; i++) disktotal += *(long *)(rb + 8 + i * 8);
        }
    }
    printf("CSVC-APPLIED %d CSVC-FAILED %d\n", applied, failed);
    printf("INV-CONSERVED %s (total=%ld want=2400)\n", total == N * START ? "true" : "false", total);
    printf("INV-DISK-MATCH %s (diskTotal=%ld)\n", disktotal == total ? "true" : "false", disktotal);
    printf("CSVC-HASH %016lx\n", h);
    printf("CSVC-EXIT 0\n");
    return 0;
}

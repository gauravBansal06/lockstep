/* Distributed bank-service — the concurrent statesvc SPLIT ACROSS TWO NODES, over
 * a real emulated network (virtio-net + the two-guest switch). One binary:
 *   distsvc server            → node B: bind 0.0.0.0:9000, apply transfers, invariant
 *   distsvc client <peer_ip>  → node A: 6 workers × 500 requests to the server
 * Same protocol/PRNG/hash as ../concurrent (SPEC.md). The server is the canonical
 * state; the conservation invariant must hold across every network schedule/fault.
 * Deterministic: the switch's frame delivery + each node's single-vCPU scheduling
 * are deterministic, so a seed replays byte-identically. Under drops TCP retransmits
 * (invariant holds); under partition fewer transfers apply but money is still
 * conserved — the property a DST harness checks. */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <pthread.h>
#include <arpa/inet.h>
#include <sys/socket.h>
#include <netinet/in.h>
#define N 16
#define START 150
#define CLIENTS 6
#define PERCLIENT 500
#define PORT 9000

static long bal[N];
static int applied = 0, failed = 0, done_conns = 0;
static pthread_mutex_t lk = PTHREAD_MUTEX_INITIALIZER;
static char peer[64];

static int apply(int frm, int to, long amt) {
    pthread_mutex_lock(&lk);
    int r;
    if (bal[frm] >= amt) { bal[frm] -= amt; bal[to] += amt; applied++; r = 1; }
    else { failed++; r = 0; }
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
    pthread_mutex_lock(&lk); done_conns++; pthread_mutex_unlock(&lk);
    return NULL;
}
static void *accept_loop(void *arg) {
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
static void run_server(void) {
    printf("CSVC-START c-server\n"); fflush(stdout);
    for (int i = 0; i < N; i++) bal[i] = START;
    int ls = socket(AF_INET, SOCK_STREAM, 0);
    int one = 1; setsockopt(ls, SOL_SOCKET, SO_REUSEADDR, &one, sizeof one);
    struct sockaddr_in a = {0};
    a.sin_family = AF_INET; a.sin_port = htons(PORT); a.sin_addr.s_addr = htonl(INADDR_ANY);
    if (bind(ls, (void *)&a, sizeof a) < 0) { printf("CSVC-LISTEN-ERR\n"); return; }
    listen(ls, 16);
    printf("CSVC-LISTEN 9000\n"); fflush(stdout);
    pthread_t srv;
    pthread_create(&srv, NULL, accept_loop, (void *)(long)ls);
    pthread_detach(srv);
    /* Wait until all client connections have finished, or a ~40s virtual-time cap
     * (so a partition doesn't hang the run — we then report the partial, still-
     * conserved state). */
    for (int i = 0; i < 4000; i++) {
        pthread_mutex_lock(&lk); int d = done_conns; pthread_mutex_unlock(&lk);
        if (d >= CLIENTS) break;
        usleep(10000);
    }
    long total = 0;
    for (int i = 0; i < N; i++) total += bal[i];
    unsigned char mb[N * 8];
    for (int i = 0; i < N; i++) *(long *)(mb + i * 8) = bal[i];
    unsigned long h = 0xcbf29ce484222325UL;
    for (int i = 0; i < N * 8; i++) h = (h ^ mb[i]) * 0x100000001b3UL;
    printf("CSVC-APPLIED %d CSVC-FAILED %d\n", applied, failed);
    printf("INV-CONSERVED %s (total=%ld want=2400)\n", total == N * START ? "true" : "false", total);
    printf("CSVC-HASH %016lx\n", h);
    printf("CSVC-EXIT 0\n"); fflush(stdout);
}
static void *client(void *arg) {
    long idx = (long)arg, state = (idx + 1) * 1000 + 1;
    struct sockaddr_in a = {0};
    a.sin_family = AF_INET; a.sin_port = htons(PORT); inet_pton(AF_INET, peer, &a.sin_addr);
    int fd = -1, ok = 0;
    for (int r = 0; r < 80; r++) { /* the server may still be booting */
        fd = socket(AF_INET, SOCK_STREAM, 0);
        if (connect(fd, (void *)&a, sizeof a) == 0) { ok = 1; break; }
        close(fd);
        usleep(100000);
    }
    if (!ok) return NULL;
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
static void run_client(void) {
    printf("CSVC-START c-client\n"); fflush(stdout);
    pthread_t cl[CLIENTS];
    for (long c = 0; c < CLIENTS; c++) pthread_create(&cl[c], NULL, client, (void *)c);
    for (int c = 0; c < CLIENTS; c++) pthread_join(cl[c], NULL);
    printf("CSVC-CLIENT-DONE\n"); fflush(stdout);
}
int main(int argc, char **argv) {
    if (argc >= 2 && strcmp(argv[1], "server") == 0) run_server();
    else if (argc >= 3 && strcmp(argv[1], "client") == 0) {
        strncpy(peer, argv[2], sizeof peer - 1);
        run_client();
    } else { printf("usage: distsvc server | distsvc client <peer_ip>\n"); return 1; }
    return 0;
}

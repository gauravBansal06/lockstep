/* Racy multi-threaded program with frequent scheduling points (sched_yield),
 * modelling an async microservice whose threads interleave at I/O boundaries.
 * The shared counter is unsynchronised -> lost updates depend on interleaving. */
#include <pthread.h>
#include <stdio.h>
#include <stdint.h>
#include <sched.h>
#define THREADS 8
#define ITERS   200000
static volatile long counter = 0;       /* shared, no lock: DATA RACE */
static long progress[THREADS];
static void* work(void* arg) {
    long id = (long)arg, s = 0;
    for (int i = 0; i < ITERS; i++) {
        long v = counter;                /* racy load  */
        v = v + 1;
        counter = v;                     /* racy store */
        s = s * 1000003 + counter;
        if ((i & 0x3f) == 0) sched_yield();   /* yield like an I/O point */
    }
    progress[id] = s;
    return NULL;
}
int main(void) {
    printf("RACE-START\n");
    pthread_t t[THREADS];
    for (long i = 0; i < THREADS; i++) pthread_create(&t[i], NULL, work, (void*)i);
    for (int i = 0; i < THREADS; i++) pthread_join(t[i], NULL);
    unsigned long h = (unsigned long)counter;
    for (int i = 0; i < THREADS; i++) h = h * 1000003UL + (unsigned long)progress[i];
    printf("RACE-COUNTER %ld of %d\n", counter, THREADS * ITERS);
    printf("RACE-HASH %016lx\n", h);
    printf("RACE-EXIT 0\n");
    return 0;
}

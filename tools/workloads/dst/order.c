/* Scheduling-ORDER-sensitive concurrency: N threads do work then append their
 * id to a shared log under a mutex. No data race (lock-protected), but the LOG
 * ORDER is the thread interleaving => a pure function of the scheduler. On real
 * HW it varies run-to-run; on a DST substrate it must be deterministic per seed
 * and EXPLORE different interleavings across seeds. */
#include <pthread.h>
#include <stdio.h>
#define THREADS 8
#define ROUNDS  3000
static pthread_mutex_t lk = PTHREAD_MUTEX_INITIALIZER;
static unsigned char seq[THREADS*ROUNDS];
static int pos = 0;
static volatile long spin = 0;
static void* work(void* arg) {
    long id = (long)arg;
    for (int r = 0; r < ROUNDS; r++) {
        for (volatile int k = 0; k < 300; k++) spin += k;  /* work between sync points */
        pthread_mutex_lock(&lk);
        seq[pos++] = (unsigned char)id;
        pthread_mutex_unlock(&lk);
    }
    return NULL;
}
int main(void) {
    printf("ORDER-START\n");
    pthread_t t[THREADS];
    for (long i = 0; i < THREADS; i++) pthread_create(&t[i], NULL, work, (void*)i);
    for (int i = 0; i < THREADS; i++) pthread_join(t[i], NULL);
    unsigned long h = 14695981039346656037UL;
    for (int i = 0; i < pos; i++) { h ^= seq[i]; h *= 1099511628211UL; }
    int firsthalf[THREADS] = {0};
    for (int i = 0; i < pos/2; i++) firsthalf[seq[i]]++;
    printf("ORDER-POS %d\n", pos);
    printf("ORDER-HASH %016lx\n", h);
    printf("ORDER-FIRSTHALF");
    for (int i = 0; i < THREADS; i++) printf(" %d", firsthalf[i]);
    printf("\n");
    printf("ORDER-EXIT 0\n");
    return 0;
}

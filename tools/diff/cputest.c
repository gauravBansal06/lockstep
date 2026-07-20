/* Differential conformance harness: runs each test vector on the REAL CPU and
 * prints the result registers, matching the emulator's `--difftest` output.
 *
 * Vector line: `rax rbx rcx rdx flagmask : <hex instruction bytes>`
 *   flagmask selects which RFLAGS bits are *defined* for this instruction.
 * Build:  gcc -O0 -no-pie cputest.c trampoline.S -o cputest
 * Run:    ./cputest vectors.txt
 */
#include <ctype.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>

extern void run_test(void *code, const uint64_t *in, uint64_t *out);
extern void run_test256(void *code, const uint64_t *in, uint64_t *out);

static int hex2(const char *s) {
    char b[3] = {s[0], s[1], 0};
    return (int)strtol(b, NULL, 16);
}

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s vectors.txt [256]\n", argv[0]);
        return 2;
    }
    int mode256 = (argc >= 3 && strcmp(argv[2], "256") == 0);
    int mode_x87 = (argc >= 3 && strcmp(argv[2], "x87") == 0);
    FILE *f = fopen(argv[1], "r");
    if (!f) {
        perror("open vectors");
        return 2;
    }
    void *page = mmap(NULL, 4096, PROT_READ | PROT_WRITE | PROT_EXEC,
                      MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (page == MAP_FAILED) {
        perror("mmap");
        return 2;
    }

    char line[2048];
    while (fgets(line, sizeof line, f)) {
        if (line[0] == '#' || line[0] == '\n')
            continue;
        char *colon = strchr(line, ':');
        if (!colon)
            continue;
        *colon = 0;

        uint64_t in[8] = {0, 0, 0, 0, 0, 0, 0, 0};
        uint64_t mask = 0;
        if (mode256) {
            sscanf(line, "%lx %lx %lx %lx %lx %lx %lx %lx", &in[0], &in[1], &in[2],
                   &in[3], &in[4], &in[5], &in[6], &in[7]);
        } else {
            sscanf(line, "%lx %lx %lx %lx %lx", &in[0], &in[1], &in[2], &in[3], &mask);
        }

        uint8_t code[256];
        int n = 0;
        for (char *p = colon + 1; *p && n < 254;) {
            while (*p == ' ' || *p == '\t' || *p == '\n')
                p++;
            if (!isxdigit((unsigned char)p[0]) || !isxdigit((unsigned char)p[1]))
                break;
            code[n++] = (uint8_t)hex2(p);
            p += 2;
        }
        code[n++] = 0xC3; /* RET back to the trampoline */
        memcpy(page, code, n);

        if (mode_x87) {
            /* x87 80-bit: two m80 operands in the input buffer, blob (fld/fld/op/fstp)
             * reads [rdi],[rdi+10] and writes the m80 result to [rsi]. */
            uint64_t am = 0, bm = 0;
            unsigned ase = 0, bse = 0;
            sscanf(line, "%lx %x %lx %x", &am, &ase, &bm, &bse);
            uint8_t inbuf[32] = {0}, outbuf[16] = {0};
            memcpy(inbuf, &am, 8);
            inbuf[8] = ase & 0xff;
            inbuf[9] = (ase >> 8) & 0xff;
            memcpy(inbuf + 10, &bm, 8);
            inbuf[18] = bse & 0xff;
            inbuf[19] = (bse >> 8) & 0xff;
            run_test256(page, (uint64_t *)inbuf, (uint64_t *)outbuf);
            uint64_t rm;
            memcpy(&rm, outbuf, 8);
            unsigned rse = outbuf[8] | (outbuf[9] << 8);
            printf("%016lx %04x\n", rm, rse);
        } else if (mode256) {
            /* 8 input qwords (ymm0||ymm1) -> blob writes ymm0 -> 4 output qwords. */
            uint64_t out[4] __attribute__((aligned(32))) = {0, 0, 0, 0};
            run_test256(page, in, out);
            printf("%016lx %016lx %016lx %016lx\n", out[0], out[1], out[2], out[3]);
        } else {
            uint64_t out[5] = {0, 0, 0, 0, 0};
            run_test(page, in, out);
            printf("%016lx %016lx %016lx %016lx %06lx\n", out[0], out[1], out[2],
                   out[3], out[4] & mask);
        }
    }
    fclose(f);
    return 0;
}

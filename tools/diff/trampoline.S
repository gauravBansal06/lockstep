// run_test(void *code /*rdi*/, const uint64_t in[4] /*rsi*/, uint64_t out[4] /*rdx*/)
//
// Loads RAX/RBX/RCX/RDX from in[], clears RFLAGS to a known state (0x2, so
// CF-dependent ops like ADC are deterministic), CALLs the code blob (which ends
// in RET), then captures RAX/RBX/RCX/RDX into out[]. Code/out pointers live on
// the stack across the call so the test may clobber any register.
.intel_syntax noprefix
.text
.globl run_test
run_test:
    push rbx
    push rbp
    push r12
    push r13
    push r14
    push r15
    push rdx                  // out  -> [rsp+8] after next push
    push rdi                  // code -> [rsp]

    mov rax, [rsi+0]
    mov rbx, [rsi+8]
    mov rcx, [rsi+16]
    mov rdx, [rsi+24]
    push 2
    popfq                     // known incoming flags (matches emulator's 0x2)

    call qword ptr [rsp]      // run the instruction(s) under test

    pushfq                    // capture flags FIRST (later movs/pops don't set flags)
    mov rsi, [rsp+16]         // out ptr ([rsp]=flags, [rsp+8]=code, [rsp+16]=out)
    mov [rsi+0], rax
    mov [rsi+8], rbx
    mov [rsi+16], rcx
    mov [rsi+24], rdx
    pop rax                   // saved flags
    mov [rsi+32], rax

    add rsp, 16               // pop code + out
    pop r15
    pop r14
    pop r13
    pop r12
    pop rbp
    pop rbx
    ret

// run_test256(void *code /*rdi*/, const uint64_t in[8] /*rsi*/, uint64_t out[4] /*rdx*/)
//
// 256-bit (YMM) path. The blob loads ymm0 from [rdi] and ymm1 from [rdi+0x20],
// runs the op, and stores ymm0 to [rsi]; so we pass the input buffer in RDI and
// the output buffer in RSI, then CALL the blob (which ends in RET). The result
// lands in out[] via the blob's store — no GPR capture needed.
.globl run_test256
run_test256:
    push rbx
    push rbp
    push r12
    push r13
    push r14
    push r15
    mov rax, rdi              // code ptr
    mov rdi, rsi              // rdi = input buffer  (blob reads [rdi], [rdi+0x20])
    mov rsi, rdx              // rsi = output buffer (blob writes [rsi])
    push 2
    popfq
    call rax                  // run the instruction(s) under test
    vzeroupper                // clean AVX→SSE transition before returning to C
    pop r15
    pop r14
    pop r13
    pop r12
    pop rbp
    pop rbx
    ret
.section .note.GNU-stack,"",@progbits

.text
.globl _start

_start:
    movq $1, %rax
    movq $1, %rdi
    leaq message(%rip), %rsi
    movq $14, %rdx
    syscall

    movq $60, %rax
    xorq %rdi, %rdi
    syscall

.section .rodata
message:
    .ascii "hello, world\n"

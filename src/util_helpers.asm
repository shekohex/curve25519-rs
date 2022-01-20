; assembly routines for MSVC
; extracted from https://github.com/DaGenix/rust-crypto
ifndef X64
.686p
.XMM
.model flat, C
endif

.code

; fixed_time_eq_asm for X86-64
ifdef X64
fixed_time_eq_asm PROC public lhs:QWORD, rhs:QWORD, count:QWORD
  ; lhs is in RCX
  ; rhs is in RDX
  ; count is in R8

  ; set the return value initially to 0
  xor eax, eax

  test r8, r8
  jz DONE

  BEGIN:

  mov r10b, [rcx]
  xor r10b, [rdx]
  or al, r10b

  inc rcx
  inc rdx
  dec r8
  jnz BEGIN

  DONE:

  ret
fixed_time_eq_asm ENDP
endif

; fixed_time_eq_asm for X86 (32-bit)
ifndef X64
fixed_time_eq_asm PROC public lhs:DWORD, rhs:DWORD, count:DWORD
  push ebx
  push esi

  mov ecx, lhs
  mov edx, rhs
  mov esi, count

  xor eax, eax

  test esi, esi
  jz DONE

  BEGIN:

  mov bl, [ecx]
  xor bl, [edx]
  or al, bl

  inc ecx
  inc edx
  dec esi
  jnz BEGIN

  DONE:

  pop esi
  pop ebx

  ret
fixed_time_eq_asm ENDP
endif

end


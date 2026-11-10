# Per-compiler warning and floating-point flags (spec section 4).
#
# Warnings apply only to this repo's own targets. Floating-point flags are part of the
# qkern::core usage requirements, because the kernels are header-only and their results
# depend on how the consumer's translation unit is compiled (DQ-08: no FMA contraction,
# never -ffast-math).

option(QKERN_WARNINGS_AS_ERRORS "Treat compiler warnings as errors" ON)

function(qkern_set_warnings target)
  if(MSVC)
    target_compile_options(${target} PRIVATE /W4 /permissive- /utf-8)
    if(QKERN_WARNINGS_AS_ERRORS)
      target_compile_options(${target} PRIVATE /WX)
    endif()
  else()
    target_compile_options(${target} PRIVATE -Wall -Wextra -Wpedantic -Wconversion -Wshadow)
    if(QKERN_WARNINGS_AS_ERRORS)
      target_compile_options(${target} PRIVATE -Werror)
    endif()
  endif()
endfunction()

function(qkern_set_fp_flags target)
  # GCC and Clang: forbid contraction of a*b+c into FMA. MSVC: /fp:precise (no contraction
  # unless /fp:contract is given).
  target_compile_options(
    ${target}
    INTERFACE $<$<COMPILE_LANG_AND_ID:CXX,GNU,Clang,AppleClang>:-ffp-contract=off>
              $<$<COMPILE_LANG_AND_ID:CXX,MSVC>:/fp:precise>)
endfunction()

/*
 * sha256.cu Implementation of SHA256 Hashing
 * https://github.com/mochimodev/cuda-hashing-algos/blob/master/sha256.cu
 *
 * Date: 12 June 2019
 * Revision: 1
 * *
 * Based on the public domain Reference Implementation in C, by
 * Brad Conte, original code here:
 *
 * https://github.com/B-Con/crypto-algorithms
 *
 * This file is released into the Public Domain.
 *
 *
 * Modified for CS4160 TU Delft by Hendrik Lambert, 2026
 *
 */

#ifndef KERNEL_CUH
#define KERNEL_CUH

#include <stdio.h>

typedef unsigned char BYTE;
typedef unsigned int WORD;
typedef unsigned long long LONG;

#define SHA256_DIGEST_SIZE 32

void mcm_cuda_sha256_hash_batch(BYTE* in, WORD inlen, BYTE* out, WORD n_batch);

/*
 * GPU PoW search. Loops until a winning nonce is found and writes it (with its
 * hash) to out_nonce / out_hash.
 */
void cuda_sha256_pow(const BYTE* prefix, WORD prefix_len,
                     WORD difficulty_bits, LONG nonce_start,
                     LONG* out_nonce, BYTE* out_hash);

#endif

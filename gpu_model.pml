/*
 * Promela model for GPU pipeline (sum_even kernel, PTX level)
 * based on VeHa problem statement
 * 
 * Configuration:
 *   SM_COUNT      - number of Streaming Multiprocessors
 *   WARP_SIZE     - threads per warp (fixed by architecture)
 *   THREADS_PER_BLOCK - work‑group size (WG), must be multiple of WARP_SIZE
 *   TS            - tile size (elements per thread) – can be varied
 *
 * Debug version with extensive printf.
 */

#define SM_COUNT        2
#define WARP_SIZE       2
#define THREADS_PER_BLOCK 4
#define WARPS_PER_SM    (THREADS_PER_BLOCK / WARP_SIZE)
#define TOTAL_WARPS     (SM_COUNT * WARPS_PER_SM)

#define REG_COUNT       58
#define PRED_COUNT      7
#define SHARED_SIZE     256
#define GLOBAL_SIZE     1024
#define MASK_STACK_DEPTH 8

// ----------------------------------------------------------------------
// Warp state
// ----------------------------------------------------------------------
typedef Warp {
    int pc;                             // program counter
    int mask;                            // active threads (bitmask)
    int stack_mask[MASK_STACK_DEPTH];     // saved masks
    int stack_pc[MASK_STACK_DEPTH];       // saved PCs
    int sp;                               // stack pointer (-1 if empty)
    int r[TOTAL_WARPS * WARP_SIZE * REG_COUNT]; // flattened registers
    int p[TOTAL_WARPS * WARP_SIZE * PRED_COUNT]; // flattened predicates
    int wait_cycles;                       // cycles until ready (for memory)
    bit finished;                          // all threads in warp retired
};

Warp warps[TOTAL_WARPS];

// Memory
int global_mem[GLOBAL_SIZE];
int shared_mem[SM_COUNT*SHARED_SIZE];

// Kernel parameters (addresses and constants)
int param[4];   // 0: input array base, 1: output array base, 2: shared base, 3: TS

// Per‑thread constants (set at launch)
int tid[SM_COUNT*TOTAL_WARPS*WARP_SIZE];
int ctaid[SM_COUNT*TOTAL_WARPS*WARP_SIZE];
int ntid[SM_COUNT*TOTAL_WARPS*WARP_SIZE];
int envreg3[SM_COUNT*TOTAL_WARPS*WARP_SIZE];

// Scheduler state
int last_warp[SM_COUNT];
int idle_ticks[SM_COUNT];
int total_ticks[SM_COUNT];

// Global cycle counter
int cycle = 0;
bit all_done = 0;

bit sum_done = 0;
int summ = 0;

bit barrier_reached[TOTAL_WARPS];  

// ----------------------------------------------------------------------
// Helper macros for register/predicate access
// ----------------------------------------------------------------------
#define R(warp, thread, reg) \
    warps[warp].r[(warp)*WARP_SIZE*REG_COUNT + (thread)*REG_COUNT + (reg)]

#define P(warp, thread, pred) \
    warps[warp].p[(warp)*WARP_SIZE*PRED_COUNT + (thread)*PRED_COUNT + (pred)]

// ----------------------------------------------------------------------
// Stack operations for divergence
// ----------------------------------------------------------------------
inline push_mask_pc(warp, saved_mask, saved_pc) {
    atomic {
        warps[warp].sp = warps[warp].sp + 1;
        assert(warps[warp].sp < MASK_STACK_DEPTH);
        warps[warp].stack_mask[warps[warp].sp] = saved_mask;
        warps[warp].stack_pc[warps[warp].sp] = saved_pc;
        printf("Warp %d: PUSH mask=0x%x pc=%d (sp=%d)\n",
               warp, saved_mask, saved_pc, warps[warp].sp);
    }
}

inline pop_mask_pc(warp) {
    atomic {
        assert(warps[warp].sp >= 0);
        warps[warp].mask = warps[warp].stack_mask[warps[warp].sp];
        warps[warp].pc = warps[warp].stack_pc[warps[warp].sp];
        printf("Warp %d: POP  mask=0x%x pc=%d (sp=%d)\n",
               warp, warps[warp].mask, warps[warp].pc, warps[warp].sp);
        warps[warp].sp = warps[warp].sp - 1;
    }
}

// ----------------------------------------------------------------------
// Divergent conditional branch
// ----------------------------------------------------------------------
/*inline bra_cond(warp, pred_reg, target_pc) {
    int true_mask = 0;
    int false_mask = 0;
    //int i;   
    atomic {
        // Evaluate predicate for each active thread
        for (i : 0 .. WARP_SIZE-1) {
            if
                :: (warps[warp].mask & (1<<i)) ->
                    if
                        :: (P(warp, i, pred_reg) != 0) ->
                            true_mask = true_mask | (1<<i);
                        :: else ->
                            false_mask = false_mask | (1<<i);
                    fi;
                :: else -> skip;
            fi;
        }
        printf("Warp %d: bra_cond pred=%d true=0x%x false=0x%x\n",
               warp, pred_reg, true_mask, false_mask);
        // If both sets non‑empty, diverge
        if
            :: (true_mask != 0 && false_mask != 0) ->
                push_mask_pc(warp, false_mask, warps[warp].pc + 1);
                warps[warp].mask = true_mask;
                warps[warp].pc = target_pc;
                printf("Warp %d: DIVERGE -> target %d, new mask=0x%x\n",
                       warp, target_pc, warps[warp].mask);
            :: (true_mask != 0) ->
                warps[warp].pc = target_pc;
                printf("Warp %d: BRANCH (all true) -> target %d\n",
                       warp, target_pc);
            :: else -> // all false
                warps[warp].pc = warps[warp].pc + 1;
                printf("Warp %d: FALLTHROUGH (all false) -> pc=%d\n",
                       warp, warps[warp].pc);
        fi;
    }
}*/

// ----------------------------------------------------------------------
// Unconditional branch (no divergence)
// ----------------------------------------------------------------------
inline bra_uni(warp, target_pc) {
    atomic {
        warps[warp].pc = target_pc;
        printf("Warp %d: BRA.UNI -> target %d\n", warp, target_pc);
    }
}

// ----------------------------------------------------------------------
// Pop at convergence points (e.g., BB0_4)
// ----------------------------------------------------------------------
inline convergence_pop(warp) {
    atomic {
        if
            :: (warps[warp].sp >= 0) ->
                pop_mask_pc(warp);
            :: else -> skip;
        fi;
    }
}

// ----------------------------------------------------------------------
// Memory access with cache simulation
// ----------------------------------------------------------------------
inline ld_global(warp, thread, dest_reg, addr_reg) {
    int addr;
    atomic {
        addr = R(warp, thread, addr_reg);
        // Assume word‑aligned address
        R(warp, thread, dest_reg) = global_mem[addr/4];
        printf("Warp %d thread %d: ld.global addr=%d (word %d) -> %d\n",
               warp, thread, addr, addr/4, global_mem[addr/4]);
        // Non‑deterministic hit (1 cycle) or miss (10 cycles)
        if
            :: true -> warps[warp].wait_cycles = 1;
            :: true -> warps[warp].wait_cycles = 10;
        fi;
        printf("Warp %d: ld.global latency=%d cycles\n",
               warp, warps[warp].wait_cycles);
    }
}

// ----------------------------------------------------------------------
// PTX instruction execution for a given warp
// ----------------------------------------------------------------------
inline execute_instruction(warp) {
    int tm, fm;  
    //int i;   // loop index over threads 
    printf("DEBUG: Warp %d PC=%d mask=0x%x sp=%d\n",
       warp, warps[warp].pc, warps[warp].mask, warps[warp].sp);
    atomic {
        printf("Warp %d: EXECUTING PC=%d mask=0x%x\n",
               warp, warps[warp].pc, warps[warp].mask);
        if
        // --------------------------------------------------------------
        :: warps[warp].pc == 0 ->   // ld.param.u32 %r4, [sum_even_param_3]
            for (i : 0 .. WARP_SIZE-1) {
                if :: (warps[warp].mask & (1<<i)) ->
                    R(warp, i, 4) = param[3];
                    printf("Warp %d thread %d: %%r4 = %d (TS)\n",
                           warp, i, param[3]);
                :: else -> skip;
                fi;
            }
            warps[warp].pc = 1;

        :: warps[warp].pc == 1 ->   // mov.u32 %r14, %tid.x
            for (i : 0 .. WARP_SIZE-1) {
                if :: (warps[warp].mask & (1<<i)) ->
                    R(warp, i, 14) = tid[warp*TOTAL_WARPS+i];
                    printf("Warp %d thread %d: %%r14 = %d (tid)\n",
                           warp, i, tid[warp*TOTAL_WARPS+i]);
                :: else -> skip;
                fi;
            }
            warps[warp].pc = 2;

        :: warps[warp].pc == 2 ->   // mov.u32 %r15, %envreg3
            for (i : 0 .. WARP_SIZE-1) {
                if :: (warps[warp].mask & (1<<i)) ->
                    R(warp, i, 15) = envreg3[warp*TOTAL_WARPS+i];
                    printf("Warp %d thread %d: %%r15 = %d (envreg3)\n",
                           warp, i, envreg3[warp*TOTAL_WARPS+i]);
                :: else -> skip;
                fi;
            }
            warps[warp].pc = 3;

        :: warps[warp].pc == 3 ->   // mov.u32 %r16, %ntid.x
            for (i : 0 .. WARP_SIZE-1) {
                if :: (warps[warp].mask & (1<<i)) ->
                    R(warp, i, 16) = ntid[warp*TOTAL_WARPS+i];
                    printf("Warp %d thread %d: %%r16 = %d (ntid)\n",
                           warp, i, ntid[warp*TOTAL_WARPS+i]);
                :: else -> skip;
                fi;
            }
            warps[warp].pc = 4;

        :: warps[warp].pc == 4 ->   // mov.u32 %r17, %ctaid.x
            for (i : 0 .. WARP_SIZE-1) {
                if :: (warps[warp].mask & (1<<i)) ->
                    R(warp, i, 17) = ctaid[warp*TOTAL_WARPS+i];
                    printf("Warp %d thread %d: %%r17 = %d (ctaid)\n",
                           warp, i, ctaid[warp*TOTAL_WARPS+i]);
                :: else -> skip;
                fi;
            }
            warps[warp].pc = 5;

        :: warps[warp].pc == 5 ->   // mov.u32 %r18, %tid.x
            for (i : 0 .. WARP_SIZE-1) {
                if :: (warps[warp].mask & (1<<i)) ->
                    R(warp, i, 18) = tid[warp*TOTAL_WARPS+i];
                    printf("Warp %d thread %d: %%r18 = %d (tid)\n",
                           warp, i, tid[warp*TOTAL_WARPS+i]);
                :: else -> skip;
                fi;
            }
            warps[warp].pc = 6;

        :: warps[warp].pc == 6 ->   // add.s32 %r20, %r18, %r15
            for (i : 0 .. WARP_SIZE-1) {
                if :: (warps[warp].mask & (1<<i)) ->
                    R(warp, i, 20) = R(warp, i, 18) + R(warp, i, 15);
                    printf("Warp %d thread %d: %%r20 = %%r18+%%r15 = %d\n",
                           warp, i, R(warp, i, 20));
                :: else -> skip;
                fi;
            }
            warps[warp].pc = 7;

        :: warps[warp].pc == 7 ->   // mad.lo.s32 %r21, %r17, %r16, %r20
            for (i : 0 .. WARP_SIZE-1) {
                if :: (warps[warp].mask & (1<<i)) ->
                    R(warp, i, 21) = R(warp, i, 17) * R(warp, i, 16) + R(warp, i, 20);
                    printf("Warp %d thread %d: %%r21 = %%r17*%%r16+%%r20 = %d\n",
                           warp, i, R(warp, i, 21));
                :: else -> skip;
                fi;
            }
            warps[warp].pc = 8;

        :: warps[warp].pc == 8 ->   // mov.u32 %r19, %ntid.x
            for (i : 0 .. WARP_SIZE-1) {
                if :: (warps[warp].mask & (1<<i)) ->
                    R(warp, i, 19) = ntid[warp*TOTAL_WARPS+i];
                    printf("Warp %d thread %d: %%r19 = %d (ntid)\n",
                           warp, i, ntid[warp*TOTAL_WARPS+i]);
                :: else -> skip;
                fi;
            }
            warps[warp].pc = 9;

        :: warps[warp].pc == 9 ->   // div.u32 %r6, %r21, %r19
            for (i : 0 .. WARP_SIZE-1) {
                if :: (warps[warp].mask & (1<<i)) ->
                    R(warp, i, 6) = R(warp, i, 21) / R(warp, i, 19);
                    printf("Warp %d thread %d: %%r6 = %%r21/%%r19 = %d (my_unit)\n",
                           warp, i, R(warp, i, 6));
                :: else -> skip;
                fi;
            }
            warps[warp].pc = 10;

        :: warps[warp].pc == 10 ->  // setp.gt.s32 %p1, %r4, 0
            for (i : 0 .. WARP_SIZE-1) {
                if :: (warps[warp].mask & (1<<i)) ->
                    if :: (R(warp, i, 4) > 0) -> P(warp, i, 1) = 1
                       :: else ->   P(warp, i, 1) = 0
                    fi;
                    printf("Warp %d thread %d: %%p1 = %d (%%r4=%d)\n",
                           warp, i, P(warp,i,1), R(warp,i,4));
                :: else -> skip;
                fi;
            }
            warps[warp].pc = 11;

        :: warps[warp].pc == 11 ->  // @%p1 bra BB0_2 (target 14)
            atomic {
                tm = 0; fm = 0;
                for (i : 0 .. WARP_SIZE-1) {
                    if
                    :: (warps[warp].mask & (1<<i)) ->
                        if
                        :: (P(warp, i, 1) != 0) -> tm = tm | (1<<i);
                        :: else -> fm = fm | (1<<i);
                        fi;
                    :: else -> skip;
                    fi;
                }
                if
                :: (tm != 0 && fm != 0) ->
                    warps[warp].sp = warps[warp].sp + 1;
                    assert(warps[warp].sp < MASK_STACK_DEPTH);
                    warps[warp].stack_mask[warps[warp].sp] = fm;
                    warps[warp].stack_pc[warps[warp].sp] = warps[warp].pc + 1;
                    warps[warp].mask = tm;
                    warps[warp].pc = 14;
                :: (tm != 0) ->
                    warps[warp].mask = tm;
                    warps[warp].pc = 14;
                :: else ->
                    warps[warp].mask = fm;
                    warps[warp].pc = warps[warp].pc + 1;
                fi;
            }

        :: warps[warp].pc == 12 ->  // mov.u32 %r56, 0
            for (i : 0 .. WARP_SIZE-1) {
                if :: (warps[warp].mask & (1<<i)) ->
                    R(warp, i, 56) = 0;
                    printf("Warp %d thread %d: %%r56 = 0 (sum)\n", warp, i);
                :: else -> skip;
                fi;
            }
            warps[warp].pc = 13;

        :: warps[warp].pc == 13 ->  // bra.uni BB0_4 (target 35)
            bra_uni(warp, 35);

        :: warps[warp].pc == 14 ->  // BB0_2: mov.u32 %r56, 0
            for (i : 0 .. WARP_SIZE-1) {
                if :: (warps[warp].mask & (1<<i)) ->
                    R(warp, i, 56) = 0;
                    printf("Warp %d thread %d: %%r56 = 0 (sum)\n", warp, i);
                :: else -> skip;
                fi;
            }
            warps[warp].pc = 15;

        :: warps[warp].pc == 15 ->  // mov.u32 %r55, %r56
            for (i : 0 .. WARP_SIZE-1) {
                if :: (warps[warp].mask & (1<<i)) ->
                    R(warp, i, 55) = R(warp, i, 56);
                    printf("Warp %d thread %d: %%r55 = %%r56 = %d\n",
                           warp, i, R(warp,i,55));
                :: else -> skip;
                fi;
            }
            warps[warp].pc = 16;

        :: warps[warp].pc == 16 ->  // BB0_3: mov.u32 %r25, %envreg3
            for (i : 0 .. WARP_SIZE-1) {
                if :: (warps[warp].mask & (1<<i)) ->
                    R(warp, i, 25) = envreg3[warp*TOTAL_WARPS+i];
                    printf("Warp %d thread %d: %%r25 = %d (envreg3)\n",
                           warp, i, envreg3[warp*TOTAL_WARPS+i]);
                :: else -> skip;
                fi;
            }
            warps[warp].pc = 17;

        :: warps[warp].pc == 17 ->  // mov.u32 %r26, %ntid.x
            for (i : 0 .. WARP_SIZE-1) {
                if :: (warps[warp].mask & (1<<i)) ->
                    R(warp, i, 26) = ntid[warp*TOTAL_WARPS+i];
                    printf("Warp %d thread %d: %%r26 = %d (ntid)\n",
                           warp, i, ntid[warp*TOTAL_WARPS+i]);
                :: else -> skip;
                fi;
            }
            warps[warp].pc = 18;

        :: warps[warp].pc == 18 ->  // mov.u32 %r27, %ctaid.x
            for (i : 0 .. WARP_SIZE-1) {
                if :: (warps[warp].mask & (1<<i)) ->
                    R(warp, i, 27) = ctaid[warp*TOTAL_WARPS+i];
                    printf("Warp %d thread %d: %%r27 = %d (ctaid)\n",
                           warp, i, ctaid[warp*TOTAL_WARPS+i]);
                :: else -> skip;
                fi;
            }
            warps[warp].pc = 19;

        :: warps[warp].pc == 19 ->  // mov.u32 %r28, %tid.x
            for (i : 0 .. WARP_SIZE-1) {
                if :: (warps[warp].mask & (1<<i)) ->
                    R(warp, i, 28) = tid[warp*TOTAL_WARPS+i];
                    printf("Warp %d thread %d: %%r28 = %d (tid)\n",
                           warp, i, tid[warp*TOTAL_WARPS+i]);
                :: else -> skip;
                fi;
            }
            warps[warp].pc = 20;

        :: warps[warp].pc == 20 ->  // add.s32 %r29, %r28, %r25
            for (i : 0 .. WARP_SIZE-1) {
                if :: (warps[warp].mask & (1<<i)) ->
                    R(warp, i, 29) = R(warp, i, 28) + R(warp, i, 25);
                    printf("Warp %d thread %d: %%r29 = %%r28+%%r25 = %d\n",
                           warp, i, R(warp,i,29));
                :: else -> skip;
                fi;
            }
            warps[warp].pc = 21;

        :: warps[warp].pc == 21 ->  // mad.lo.s32 %r30, %r27, %r26, %r29
            for (i : 0 .. WARP_SIZE-1) {
                if :: (warps[warp].mask & (1<<i)) ->
                    R(warp, i, 30) = R(warp, i, 27) * R(warp, i, 26) + R(warp, i, 29);
                    printf("Warp %d thread %d: %%r30 = %%r27*%%r26+%%r29 = %d\n",
                           warp, i, R(warp,i,30));
                :: else -> skip;
                fi;
            }
            warps[warp].pc = 22;

        :: warps[warp].pc == 22 ->  // ld.param.u32 %r54, [sum_even_param_3]
            for (i : 0 .. WARP_SIZE-1) {
                if :: (warps[warp].mask & (1<<i)) ->
                    R(warp, i, 54) = param[3];
                    printf("Warp %d thread %d: %%r54 = %d (TS)\n",
                           warp, i, param[3]);
                :: else -> skip;
                fi;
            }
            warps[warp].pc = 23;

        :: warps[warp].pc == 23 ->  // mad.lo.s32 %r31, %r54, %r30, %r55
            for (i : 0 .. WARP_SIZE-1) {
                if :: (warps[warp].mask & (1<<i)) ->
                    R(warp, i, 31) = R(warp, i, 54) * R(warp, i, 30) + R(warp, i, 55);
                    printf("Warp %d thread %d: %%r31 = %%r54*%%r30+%%r55 = %d\n",
                           warp, i, R(warp,i,31));
                :: else -> skip;
                fi;
            }
            warps[warp].pc = 24;

        :: warps[warp].pc == 24 ->  // shl.b32 %r32, %r31, 2
            for (i : 0 .. WARP_SIZE-1) {
                if :: (warps[warp].mask & (1<<i)) ->
                    R(warp, i, 32) = R(warp, i, 31) << 2;
                    printf("Warp %d thread %d: %%r32 = %%r31<<2 = %d\n",
                           warp, i, R(warp,i,32));
                :: else -> skip;
                fi;
            }
            warps[warp].pc = 25;

        :: warps[warp].pc == 25 ->  // ld.param.u32 %r49, [sum_even_param_0]
            for (i : 0 .. WARP_SIZE-1) {
                if :: (warps[warp].mask & (1<<i)) ->
                    R(warp, i, 49) = param[0];
                    printf("Warp %d thread %d: %%r49 = %d (input base)\n",
                           warp, i, param[0]);
                :: else -> skip;
                fi;
            }
            warps[warp].pc = 26;

        :: warps[warp].pc == 26 ->  // add.s32 %r33, %r49, %r32
            for (i : 0 .. WARP_SIZE-1) {
                if :: (warps[warp].mask & (1<<i)) ->
                    R(warp, i, 33) = R(warp, i, 49) + R(warp, i, 32);
                    printf("Warp %d thread %d: %%r33 = %%r49+%%r32 = %d (address)\n",
                           warp, i, R(warp,i,33));
                :: else -> skip;
                fi;
            }
            warps[warp].pc = 27;

        :: warps[warp].pc == 27 ->  // ld.global.u32 %r34, [%r33]
            for (i : 0 .. WARP_SIZE-1) {
                if :: (warps[warp].mask & (1<<i)) ->
                    ld_global(warp, i, 34, 33);
                :: else -> skip;
                fi;
            }
            // ld_global sets wait_cycles; advance pc now, warp will stall later
            warps[warp].pc = 28;

        :: warps[warp].pc == 28 ->  // and.b32 %r35, %r34, 1
            for (i : 0 .. WARP_SIZE-1) {
                if :: (warps[warp].mask & (1<<i)) ->
                    R(warp, i, 35) = R(warp, i, 34) & 1;
                    printf("Warp %d thread %d: %%r35 = %%r34 & 1 = %d\n",
                           warp, i, R(warp,i,35));
                :: else -> skip;
                fi;
            }
            warps[warp].pc = 29;

        :: warps[warp].pc == 29 ->  // setp.eq.s32 %p2, %r35, 0
            for (i : 0 .. WARP_SIZE-1) {
                if :: (warps[warp].mask & (1<<i)) ->
                    if
                        ::(R(warp, i, 35) == 0) -> P(warp, i, 2) = 1
                        ::else ->  P(warp, i, 2) = 0
                    fi;
                    printf("Warp %d thread %d: %%p2 = %d (even?)\n",
                           warp, i, P(warp,i,2));
                :: else -> skip;
                fi;
            }
            warps[warp].pc = 30;

        :: warps[warp].pc == 30 ->  // selp.b32 %r36, %r34, 0, %p2
            for (i : 0 .. WARP_SIZE-1) {
                if :: (warps[warp].mask & (1<<i)) ->
                    if
                        :: (P(warp, i, 2) != 0) ->
                            R(warp, i, 36) = R(warp, i, 34);
                        :: else ->
                            R(warp, i, 36) = 0;
                    fi;
                    printf("Warp %d thread %d: %%r36 = %d (selected value)\n",
                           warp, i, R(warp,i,36));
                :: else -> skip;
                fi;
            }
            warps[warp].pc = 31;

        :: warps[warp].pc == 31 ->  // add.s32 %r56, %r56, %r36
            for (i : 0 .. WARP_SIZE-1) {
                if :: (warps[warp].mask & (1<<i)) ->
                    R(warp, i, 56) = R(warp, i, 56) + R(warp, i, 36);
                    printf("Warp %d thread %d: %%r56 = %d (sum updated)\n",
                           warp, i, R(warp,i,56));
                :: else -> skip;
                fi;
            }
            warps[warp].pc = 32;

        :: warps[warp].pc == 32 ->  // add.s32 %r55, %r55, 1
            for (i : 0 .. WARP_SIZE-1) {
                if :: (warps[warp].mask & (1<<i)) ->
                    R(warp, i, 55) = R(warp, i, 55) + 1;
                    printf("Warp %d thread %d: %%r55 = %d (i++)\n",
                           warp, i, R(warp,i,55));
                :: else -> skip;
                fi;
            }
            warps[warp].pc = 33;

        :: warps[warp].pc == 33 ->  // setp.lt.s32 %p3, %r55, %r54
            for (i : 0 .. WARP_SIZE-1) {
                if :: (warps[warp].mask & (1<<i)) ->
                    if 
                        ::(R(warp, i, 55) < R(warp, i, 54)) ->  P(warp, i, 3) = 1
                        ::else -> P(warp, i, 3) = 0;
                    fi;
                    printf("Warp %d thread %d: %%p3 = %d (i < TS?)\n",
                           warp, i, P(warp,i,3));
                :: else -> skip;
                fi;
            }
            warps[warp].pc = 34;

        :: warps[warp].pc == 34 ->  // @%p3 bra BB0_3 (target 16)
            atomic {
                tm = 0; fm = 0;
                for (i : 0 .. WARP_SIZE-1) {
                    if
                    :: (warps[warp].mask & (1<<i)) ->
                        if
                        :: (P(warp, i, 3) != 0) -> tm = tm | (1<<i);
                        :: else -> fm = fm | (1<<i);
                        fi;
                    :: else -> skip;
                    fi;
                }
                if
                :: (tm != 0 && fm != 0) ->
                    warps[warp].sp = warps[warp].sp + 1;
                    assert(warps[warp].sp < MASK_STACK_DEPTH);
                    warps[warp].stack_mask[warps[warp].sp] = fm;
                    warps[warp].stack_pc[warps[warp].sp] = warps[warp].pc + 1;
                    warps[warp].mask = tm;
                    warps[warp].pc = 16;
                :: (tm != 0) ->
                    warps[warp].mask = tm;
                    warps[warp].pc = 16;
                :: else ->
                    warps[warp].mask = fm;
                    warps[warp].pc = warps[warp].pc + 1;
                fi;
            }

        :: warps[warp].pc == 35 ->  // BB0_4: bar.sync 0
        int wi;
         atomic {
            int sm1 = warp / WARPS_PER_SM;
            barrier_reached[warp] = 1;
        
            // Проверяем, достигли ли ВСЕ варпы этого SM барьера
            bit all_reached = 1;
            for (wi : 0 .. WARPS_PER_SM-1) {
                int wid = sm1 * WARPS_PER_SM + wi;
                if :: (barrier_reached[wid] == 0) -> all_reached = 0; break;
                :: else -> skip; fi;
            }
        
            if :: (all_reached == 1) ->
                // Все достигли: сбрасываем и продвигаем всех
                for (wi : 0 .. WARPS_PER_SM-1) {
                    int wid = sm1 * WARPS_PER_SM + wi;
                    barrier_reached[wid] = 0;
                    if :: (warps[wid].pc == 35) ->
                        // Поп стека, если есть
                        if :: (warps[wid].sp >= 0) ->
                            warps[wid].mask = warps[wid].stack_mask[warps[wid].sp];
                            warps[wid].pc = warps[wid].stack_pc[warps[wid].sp];
                            warps[wid].sp = warps[wid].sp - 1;
                        :: else -> skip; fi;
                        warps[wid].pc = 36;  // Следующая инструкция
                    :: else -> skip; fi;
                }
            :: else ->
                // Этот варп ждёт: не меняем PC, просто выходим
                skip;
            fi;
            
        }

        :: warps[warp].pc == 36 ->  // shl.b32 %r37, %r14, 2
            for (i : 0 .. WARP_SIZE-1) {
                if :: (warps[warp].mask & (1<<i)) ->
                    R(warp, i, 37) = R(warp, i, 14) << 2;
                    printf("Warp %d thread %d: %%r37 = %%r14<<2 = %d\n",
                           warp, i, R(warp,i,37));
                :: else -> skip;
                fi;
            }
            warps[warp].pc = 37;

        :: warps[warp].pc == 37 ->  // ld.param.u32 %r53, [sum_even_param_2]
            for (i : 0 .. WARP_SIZE-1) {
                if :: (warps[warp].mask & (1<<i)) ->
                    R(warp, i, 53) = param[2];
                    printf("Warp %d thread %d: %%r53 = %d (shared base)\n",
                           warp, i, param[2]);
                :: else -> skip;
                fi;
            }
            warps[warp].pc = 38;

        :: warps[warp].pc == 38 ->  // add.s32 %r38, %r53, %r37
            for (i : 0 .. WARP_SIZE-1) {
                if :: (warps[warp].mask & (1<<i)) ->
                    R(warp, i, 38) = R(warp, i, 53) + R(warp, i, 37);
                    printf("Warp %d thread %d: %%r38 = %%r53+%%r37 = %d (shared addr)\n",
                           warp, i, R(warp,i,38));
                :: else -> skip;
                fi;
            }
            warps[warp].pc = 39;

        :: warps[warp].pc == 39 ->  // st.shared.u32 [%r38], %r56
            for (i : 0 .. WARP_SIZE-1) {
                if :: (warps[warp].mask & (1<<i)) -> {
                    int addr = R(warp, i, 38);
                    int sm1 = warp / WARPS_PER_SM;
                    shared_mem[sm1*SHARED_SIZE+addr/4] = R(warp, i, 56);
                    printf("Warp %d thread %d: st.shared [%d] = %d (shared_mem[%d])\n",
                           warp, i, addr, R(warp,i,56), sm1*SHARED_SIZE+addr/4);
                }
                :: else -> skip;
                fi;
            }
            warps[warp].wait_cycles = 1;
            warps[warp].pc = 40;

        :: warps[warp].pc == 40 ->  // setp.eq.s32 %p4, %r14, 0
            for (i : 0 .. WARP_SIZE-1) {
                if :: (warps[warp].mask & (1<<i)) ->
                    if
                        ::(R(warp, i, 14) == 0) -> P(warp, i, 4) = 1
                        ::else -> P(warp, i, 4) = 0
                    fi;
                    printf("Warp %d thread %d: %%p4 = %d (thread 0?)\n",
                           warp, i, P(warp,i,4));
                :: else -> skip;
                fi;
            }
            warps[warp].pc = 41;

        :: warps[warp].pc == 41 ->  // @%p4 bra BB0_6 (target 43)
            atomic {
                tm = 0; fm = 0;
                for (i : 0 .. WARP_SIZE-1) {
                    if
                    :: (warps[warp].mask & (1<<i)) ->
                        if
                        :: (P(warp, i, 4) != 0) -> tm = tm | (1<<i);
                        :: else -> fm = fm | (1<<i);
                        fi;
                    :: else -> skip;
                    fi;
                }
            
                if
                :: (tm != 0 && fm != 0) ->  // Дивергенция
                    warps[warp].sp = warps[warp].sp + 1;
                    assert(warps[warp].sp < MASK_STACK_DEPTH);
                    warps[warp].stack_mask[warps[warp].sp] = fm;
                    warps[warp].stack_pc[warps[warp].sp] = warps[warp].pc + 1;
                    warps[warp].mask = tm;  // ✓ Активны только true-потоки
                    warps[warp].pc = 43;
                :: (tm != 0) ->  // Все true
                    warps[warp].mask = tm;  // ✓ ← ← ← КЛЮЧЕВОЕ: обновляем mask!
                    warps[warp].pc = 43;
                :: else ->  // Все false
                    warps[warp].mask = fm;  // ✓ ← ← ← КЛЮЧЕВОЕ: обновляем mask!
                    warps[warp].pc = warps[warp].pc + 1;
                fi;
            }

        :: warps[warp].pc == 42 ->  // ret
            // All active threads retire
            warps[warp].mask = 0;
            warps[warp].finished = 1;
            printf("Warp %d: RET (finished)\n", warp);

        :: warps[warp].pc == 43 ->  // BB0_6: mov.u32 %r39, %ntid.x
            for (i : 0 .. WARP_SIZE-1) {
                if :: (warps[warp].mask & (1<<i)) ->
                    R(warp, i, 39) = ntid[warp*TOTAL_WARPS+i];
                    printf("Warp %d thread %d: %%r39 = %d (ntid)\n",
                           warp, i, ntid[warp*TOTAL_WARPS+i]);
                :: else -> skip;
                fi;
            }
            warps[warp].pc = 44;

        :: warps[warp].pc == 44 ->  // setp.lt.u32 %p5, %r39, 2
            for (i : 0 .. WARP_SIZE-1) {
                if :: (warps[warp].mask & (1<<i)) ->
                    if
                        ::(R(warp, i, 39) < 2) -> P(warp, i, 5) = 1
                        ::else -> P(warp, i, 5) = 0
                    fi;
                    printf("Warp %d thread %d: %%p5 = %d (ntid<2?)\n",
                           warp, i, P(warp,i,5));
                :: else -> skip;
                fi;
            }
            warps[warp].pc = 45;

        :: warps[warp].pc == 45 ->  // @%p5 bra BB0_9 (target 56)
            atomic {
                tm = 0; fm = 0;
                for (i : 0 .. WARP_SIZE-1) {
                    if
                    :: (warps[warp].mask & (1<<i)) ->
                        if
                        :: (P(warp, i, 5) != 0) -> tm = tm | (1<<i);
                        :: else -> fm = fm | (1<<i);
                        fi;
                    :: else -> skip;
                    fi;
                }
                if
                :: (tm != 0 && fm != 0) ->
                    warps[warp].sp = warps[warp].sp + 1;
                    assert(warps[warp].sp < MASK_STACK_DEPTH);
                    warps[warp].stack_mask[warps[warp].sp] = fm;
                    warps[warp].stack_pc[warps[warp].sp] = warps[warp].pc + 1;
                    warps[warp].mask = tm;
                    warps[warp].pc = 56;
                :: (tm != 0) ->
                    warps[warp].mask = tm;
                    warps[warp].pc = 56;
                :: else ->
                    warps[warp].mask = fm;
                    warps[warp].pc = warps[warp].pc + 1;
                fi;
            }

        :: warps[warp].pc == 46 ->  // mov.u32 %r57, 1
            for (i : 0 .. WARP_SIZE-1) {
                if :: (warps[warp].mask & (1<<i)) ->
                    R(warp, i, 57) = 1;
                    printf("Warp %d thread %d: %%r57 = 1 (j)\n", warp, i);
                :: else -> skip;
                fi;
            }
            warps[warp].pc = 47;

        :: warps[warp].pc == 47 ->  // BB0_8: shl.b32 %r42, %r57, 2
            for (i : 0 .. WARP_SIZE-1) {
                if :: (warps[warp].mask & (1<<i)) ->
                    R(warp, i, 42) = R(warp, i, 57) << 2;
                    printf("Warp %d thread %d: %%r42 = %%r57<<2 = %d\n",
                           warp, i, R(warp,i,42));
                :: else -> skip;
                fi;
            }
            warps[warp].pc = 48;

        :: warps[warp].pc == 48 ->  // ld.param.u32 %r52, [sum_even_param_2]
            for (i : 0 .. WARP_SIZE-1) {
                if :: (warps[warp].mask & (1<<i)) ->
                    R(warp, i, 52) = param[2];
                    printf("Warp %d thread %d: %%r52 = %d (shared base)\n",
                           warp, i, param[2]);
                :: else -> skip;
                fi;
            }
            warps[warp].pc = 49;

        :: warps[warp].pc == 49 ->  // add.s32 %r43, %r52, %r42
            for (i : 0 .. WARP_SIZE-1) {
                if :: (warps[warp].mask & (1<<i)) ->
                    R(warp, i, 43) = R(warp, i, 52) + R(warp, i, 42);
                    printf("Warp %d thread %d: %%r43 = %%r52+%%r42 = %d (shared addr)\n",
                           warp, i, R(warp,i,43));
                :: else -> skip;
                fi;
            }
            warps[warp].pc = 50;

        :: warps[warp].pc == 50 ->  // ld.shared.u32 %r44, [%r43]
            for (i : 0 .. WARP_SIZE-1) {
                if :: (warps[warp].mask & (1<<i)) -> {
                    int addr = R(warp, i, 43);
                    int sm1 = warp / WARPS_PER_SM;
                    R(warp, i, 44) = shared_mem[sm1*SHARED_SIZE+addr/4];
                    printf("Warp %d thread %d: ld.shared [%d] -> %d (shared_mem[%d])\n",
                           warp, i, addr, R(warp,i,44), sm1*SM_COUNT+addr/4);
                }
                :: else -> skip;
                fi;
            }
            warps[warp].wait_cycles = 1;
            warps[warp].pc = 51;

        :: warps[warp].pc == 51 ->  // atom.shared.add.u32 %r45, [%r52], %r44
            for (i : 0 .. WARP_SIZE-1) {
                if :: (warps[warp].mask & (1<<i)) -> {
                    int sm1 = warp / WARPS_PER_SM;
                    int base = param[2];  // same for all threads
                    R(warp, i, 45) = shared_mem[sm1*SHARED_SIZE+base/4];
                    shared_mem[sm1*SHARED_SIZE+base/4] = shared_mem[sm1*SHARED_SIZE+base/4] + R(warp, i, 44);
                    printf("Warp %d thread %d: atom.add [%d] + %d -> new value %d, old=%d\n",
                           warp, i, base, R(warp,i,44),
                           shared_mem[sm1*SHARED_SIZE+base/4], R(warp,i,45));
                }
                :: else -> skip;
                fi;
            }
            warps[warp].wait_cycles = 1;
            warps[warp].pc = 52;

        :: warps[warp].pc == 52 ->  // mov.u32 %r41, %ntid.x
            for (i : 0 .. WARP_SIZE-1) {
                if :: (warps[warp].mask & (1<<i)) ->
                    R(warp, i, 41) = ntid[warp*TOTAL_WARPS+i];
                    printf("Warp %d thread %d: %%r41 = %d (ntid)\n",
                           warp, i, ntid[warp*TOTAL_WARPS+i]);
                :: else -> skip;
                fi;
            }
            warps[warp].pc = 53;

        :: warps[warp].pc == 53 ->  // add.s32 %r57, %r57, 1
            for (i : 0 .. WARP_SIZE-1) {
                if :: (warps[warp].mask & (1<<i)) ->
                    R(warp, i, 57) = R(warp, i, 57) + 1;
                    printf("Warp %d thread %d: %%r57 = %d (j++)\n",
                           warp, i, R(warp,i,57));
                :: else -> skip;
                fi;
            }
            warps[warp].pc = 54;

        :: warps[warp].pc == 54 ->  // setp.lt.u32 %p6, %r57, %r41
            for (i : 0 .. WARP_SIZE-1) {
                if :: (warps[warp].mask & (1<<i)) ->
                    if
                        ::(R(warp, i, 57) < R(warp, i, 41)) -> P(warp, i, 6) = 1;
                        ::else -> P(warp, i, 6) = 0;
                    fi;
                    printf("Warp %d thread %d: %%p6 = %d (j<ntid?)\n",
                           warp, i, P(warp,i,6));
                :: else -> skip;
                fi;
            }
            warps[warp].pc = 55;

        :: warps[warp].pc == 55 ->  // @%p6 bra BB0_8 (target 47)
           atomic {
                tm = 0; fm = 0;
                for (i : 0 .. WARP_SIZE-1) {
                    if
                    :: (warps[warp].mask & (1<<i)) ->
                        if
                        :: (P(warp, i, 6) != 0) -> tm = tm | (1<<i);
                        :: else -> fm = fm | (1<<i);
                        fi;
                    :: else -> skip;
                    fi;
                }
                if
                :: (tm != 0 && fm != 0) ->
                    warps[warp].sp = warps[warp].sp + 1;
                    assert(warps[warp].sp < MASK_STACK_DEPTH);
                    warps[warp].stack_mask[warps[warp].sp] = fm;
                    warps[warp].stack_pc[warps[warp].sp] = warps[warp].pc + 1;
                    warps[warp].mask = tm;
                    warps[warp].pc = 47;
                :: (tm != 0) ->
                    warps[warp].mask = tm;
                    warps[warp].pc = 47;
                :: else ->
                    warps[warp].mask = fm;
                    warps[warp].pc = warps[warp].pc + 1;
                fi;
            }

        :: warps[warp].pc == 56 ->  // BB0_9: ld.param.u32 %r51, [sum_even_param_2]
            for (i : 0 .. WARP_SIZE-1) {
                if :: (warps[warp].mask & (1<<i)) ->
                    R(warp, i, 51) = param[2];
                    printf("Warp %d thread %d: %%r51 = %d (shared base)\n",
                           warp, i, param[2]);
                :: else -> skip;
                fi;
            }
            warps[warp].pc = 57;

        :: warps[warp].pc == 57 ->  // ld.shared.u32 %r46, [%r51]
            for (i : 0 .. WARP_SIZE-1) {
                if :: (warps[warp].mask & (1<<i)) -> {
                    int addr = R(warp, i, 51);
                    int sm1 = warp / WARPS_PER_SM;
                    R(warp, i, 46) = shared_mem[sm1*SHARED_SIZE+addr/4];
                    printf("Warp %d thread %d: ld.shared [%d] -> %d (final sum for block)\n",
                           warp, i, addr, R(warp,i,46));
                }
                :: else -> skip;
                fi;
            }
            warps[warp].wait_cycles = 1;
            warps[warp].pc = 58;

        :: warps[warp].pc == 58 ->  // (additional) store to global output
            // Not in original PTX but needed: output[my_unit] = sum
            // We'll assume param[1] is output base, and %r6 holds my_unit
            for (i : 0 .. WARP_SIZE-1) {
                if :: (warps[warp].mask & (1<<i)) -> {
                    int addr = param[1] + R(warp, i, 6)*4;

                    int word_idx = addr/4;
           
                    // <<< ЛОГИРОВАНИЕ ДО ЗАПИСИ >>>
                    printf("DEBUG: Warp %d thread %d: ABOUT TO WRITE global_mem[%d] = %d (was %d)\n",
                        warp, i, word_idx, R(warp, i, 46), global_mem[word_idx]);
                
                    global_mem[word_idx] = R(warp, i, 46);
                
                    // <<< ЛОГИРОВАНИЕ ПОСЛЕ ЗАПИСИ >>>
                    printf("DEBUG: Warp %d thread %d: WRITTEN global_mem[%d] = %d\n",
                        warp, i, word_idx, global_mem[word_idx]);

                    //global_mem[addr/4] = R(warp, i, 46);
                    printf("Warp %d thread %d: st.global [%d] = %d (output[%d])\n",
                           warp, i, addr, R(warp,i,46), R(warp,i,6));
                }
                :: else -> skip;
                fi;
            }
            warps[warp].wait_cycles = 1;
            warps[warp].pc = 59;

        :: warps[warp].pc == 59 ->  // ret (end of kernel)
            warps[warp].mask = 0;
            warps[warp].finished = 1;
            printf("Warp %d: finished (end of kernel)\n", warp);

        :: else -> // Unknown pc – should not happen
            printf("ERROR: invalid PC %d for warp %d\n", warps[warp].pc, warp);
            break;
        fi;
    }
}

// ----------------------------------------------------------------------
// Scheduler process
// ----------------------------------------------------------------------
proctype scheduler() {
    int sm, w, warp;
    int i;
    printf("Scheduler started, SM_COUNT=%d, WARPS_PER_SM=%d, TOTAL_WARPS=%d\n",
           SM_COUNT, WARPS_PER_SM, TOTAL_WARPS);
    do
        :: atomic {
            cycle = cycle + 1;
            printf("========== CYCLE %d ==========\n", cycle);
            // For each SM, try to schedule a ready warp
            for (sm : 0 .. SM_COUNT-1) {
                int start = last_warp[sm];
                int found = 0;
                for (i : 0 .. WARPS_PER_SM-1) {
                    w = (start + i) % WARPS_PER_SM;
                    warp = sm * WARPS_PER_SM + w;
                    if ::(warps[warp].wait_cycles == 0 && warps[warp].mask != 0) -> {
                        printf("SM %d: selects warp %d (pc=%d, mask=0x%x)\n",
                               sm, warp, warps[warp].pc, warps[warp].mask);
                        execute_instruction(warp);
                        last_warp[sm] = (w + 1) % WARPS_PER_SM;
                        found = 1;
                        total_ticks[sm] = total_ticks[sm] + 1;
                        break;
                    } ::else -> skip
                    fi
                }
                if ::(found == 0) -> {
                    printf("SM %d: idle\n", sm);
                    idle_ticks[sm] = idle_ticks[sm] + 1;
                    total_ticks[sm] = total_ticks[sm] + 1;
                }  ::else -> skip
                fi
            }
            // Decrement wait_cycles for all warps
            for (warp : 0 .. TOTAL_WARPS-1) {
                if ::(warps[warp].wait_cycles > 0) -> {
                    warps[warp].wait_cycles = warps[warp].wait_cycles - 1;
                    printf("Warp %d: wait_cycles now %d\n",
                           warp, warps[warp].wait_cycles);
                }  ::else -> skip
                fi
            }

            // Check if all warps finished
            all_done = 1;
            for (warp : 0 .. TOTAL_WARPS-1) {
                if ::(warps[warp].mask != 0)-> {
                    all_done = 0;
                    break;
                }  ::else -> skip;
                fi
            }
        } // end atomic
        if ::(all_done) -> break
           ::else -> skip;
        fi
    od;
    // Print statistics and results
    printf("\n=== ALL WARPS FINISHED ===\n");
    for (sm : 0 .. SM_COUNT-1) {
        printf("SM %d: idle %d%%, total cycles %d\n", sm,
               100 - (idle_ticks[sm]*100 / total_ticks[sm]), total_ticks[sm]);
    }
    printf("\n--- RESULTS (sums per block) ---\n");
    summ = 0;
    for (sm : 0 .. SM_COUNT-1) {
        int base_addr = param[1]/4;  
        printf("Block %d (SM %d): sum = %d\n",
               sm, sm, global_mem[base_addr + sm]);
        summ = summ +  global_mem[base_addr + sm];     
    }
    sum_done = 1;
}

// ----------------------------------------------------------------------
// Initialization
// ----------------------------------------------------------------------
init {
    int warp_id = 0;
    int sm, w, t, i;

    // Set up threads: block 0 on SM0, block 1 on SM1
    for (sm : 0 .. SM_COUNT-1) {
        for (w : 0 .. WARPS_PER_SM-1) {
            for (t : 0 .. WARP_SIZE-1) {
                printf("setup: sm = %d w = %d t = %d, warp_id=%d\n ",
                       sm, w, t, warp_id);
                int global_thread = w * WARP_SIZE + t;
                tid[warp_id*TOTAL_WARPS+t] = global_thread;
                ctaid[warp_id*TOTAL_WARPS+t] = sm;
                ntid[warp_id*TOTAL_WARPS+t] = THREADS_PER_BLOCK;
                envreg3[warp_id*(TOTAL_WARPS-1)+t] = 0;
            }
            // Initialize warp state
            warps[warp_id].pc = 0;
            warps[warp_id].mask = (1 << WARP_SIZE) - 1;
            warps[warp_id].sp = -1;
            warps[warp_id].wait_cycles = 0;
            warps[warp_id].finished = 0;
            // Registers and predicates are zero by default
            warp_id = warp_id + 1;
        }
    }

    int N = 32; //N < GLOBAL_SIZE

    // Set kernel parameters (example values)
    param[0] = 0;          // input array base address
    param[1] = 1000;       // output array base
    param[2] = 0;          // shared memory base (offset)
    param[3] = N / (SM_COUNT*THREADS_PER_BLOCK);          // TS (tile size) – can be varied

    // Initialize global memory with some test data (even/odd numbers)
    for (i : 0 .. N-1) {
        global_mem[i] = i;   // simple pattern: 0,1,2,3,...
    }

    for (w : 0 .. TOTAL_WARPS-1) { barrier_reached[w] = 0; }

    // Start scheduler
    run scheduler();
}

ltl sum_correct {
    [] (all_done -> (global_mem[250] == 56 && global_mem[251] == 184))
}

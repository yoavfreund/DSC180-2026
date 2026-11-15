/*
 * vj_scan.h: only for scan templates (never for a submitted rule).
 * P0..P3 are integer parameters. The scan fills them with every combination
 * in the ranges you give and reports the best concrete rules.
 */
#ifndef VJ_SCAN_H
#define VJ_SCAN_H
extern int vj_p[4];
#define P0 (vj_p[0])
#define P1 (vj_p[1])
#define P2 (vj_p[2])
#define P3 (vj_p[3])
#endif

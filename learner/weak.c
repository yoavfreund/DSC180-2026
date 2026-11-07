// learner/weak.c
// Weak learners: decision stump, depth-limited tree, and L2-regularized logistic regression.
// Thread-safe: no global or static mutable state. All memory is allocated per call.

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "weak.h"

#define FAM_STUMP  0
#define FAM_TREE   1
#define FAM_LOGREG 2

#define MAX_TREE_DEPTH 10
#define SMOOTH 1e-6

enum { OK = 0, ERR_SPEC = -1, ERR_ARG = -2, ERR_IO = -3, ERR_MEM = -4, ERR_FORMAT = -5 };

/* ======================= tree and stump ======================= */

typedef struct {
    int feat;          /* -1 for a leaf */
    double thr;        /* go left if x[feat] <= thr */
    int left, right;   /* child node ids; children always have larger ids than their parent */
    int sign;          /* leaf label, +1 or -1 */
    double prob;       /* leaf P(y = +1) with light smoothing */
} Node;

typedef struct {
    double v;
    int r;
} Pair;

typedef struct {
    const double *X;
    int n_feat;
    const int *y;
    const double *w;
    int max_depth;
    int min_leaf;
    Node *nodes;
    int n_nodes;
    Pair *buf;         /* scratch space for sorting, size n_train */
} Builder;

static int cmp_pair(const void *a, const void *b) {
    const Pair *p = a, *q = b;
    if (p->v < q->v) return -1;
    if (p->v > q->v) return 1;
    return (p->r < q->r) ? -1 : (p->r > q->r);   /* deterministic tie-break by row index */
}

/* Weighted Gini impurity times total weight: t - (wp^2 + wn^2) / t */
static double side_score(double wp, double wn) {
    double t = wp + wn;
    if (t <= 0.0) return 0.0;
    return t - (wp * wp + wn * wn) / t;
}

static void leaf_fill(Node *nd, double wp, double wn) {
    nd->feat = -1;
    nd->thr = 0.0;
    nd->left = nd->right = -1;
    nd->sign = (wp >= wn) ? 1 : -1;
    nd->prob = (wp + SMOOTH) / (wp + wn + 2.0 * SMOOTH);
}

static int build(Builder *B, int *rows, int n, int depth) {
    if (B->n_nodes >= (1 << (B->max_depth + 1)) - 1) return -1;
    int id = B->n_nodes++;
    Node *nd = &B->nodes[id];

    double wp = 0.0, wn = 0.0;
    for (int i = 0; i < n; i++) {
        int r = rows[i];
        if (B->y[r] > 0) wp += B->w[r];
        else             wn += B->w[r];
    }
    leaf_fill(nd, wp, wn);

    if (depth >= B->max_depth || n < 2 * B->min_leaf || wp <= 0.0 || wn <= 0.0)
        return id;

    int best_f = -1;
    double best_thr = 0.0, best_score = INFINITY;

    for (int f = 0; f < B->n_feat; f++) {
        for (int i = 0; i < n; i++) {
            B->buf[i].v = B->X[(size_t)rows[i] * B->n_feat + f];
            B->buf[i].r = rows[i];
        }
        qsort(B->buf, (size_t)n, sizeof(Pair), cmp_pair);

        double lp = 0.0, ln = 0.0;
        for (int k = 0; k + 1 < n; k++) {
            int r = B->buf[k].r;
            if (B->y[r] > 0) lp += B->w[r];
            else             ln += B->w[r];

            int nl = k + 1, nr = n - nl;
            double v = B->buf[k].v, vnext = B->buf[k + 1].v;
            if (v == vnext) continue;                       /* split only between distinct values */
            if (nl < B->min_leaf || nr < B->min_leaf) continue;

            double score = side_score(lp, ln) + side_score(wp - lp, wn - ln);
            if (score < best_score) {                       /* strict <: lowest feature, then lowest threshold */
                double thr = 0.5 * (v + vnext);
                if (thr >= vnext) thr = v;                  /* guard against rounding */
                best_score = score;
                best_f = f;
                best_thr = thr;
            }
        }
    }

    if (best_f < 0) return id;                              /* no valid split: stay a leaf */

    int *left = malloc(sizeof(int) * (size_t)n);
    int *right = malloc(sizeof(int) * (size_t)n);
    if (!left || !right) { free(left); free(right); return -1; }

    int nl = 0, nr = 0;
    for (int i = 0; i < n; i++) {
        int r = rows[i];
        if (B->X[(size_t)r * B->n_feat + best_f] <= best_thr) left[nl++] = r;
        else                                                  right[nr++] = r;
    }

    nd->feat = best_f;
    nd->thr = best_thr;
    int lid = build(B, left, nl, depth + 1);
    int rid = build(B, right, nr, depth + 1);
    free(left);
    free(right);
    if (lid < 0 || rid < 0) return -1;

    B->nodes[id].left = lid;
    B->nodes[id].right = rid;
    return id;
}

static int fit_tree(int max_depth, int min_leaf, int n_feat, const double *X,
                    const int *y, const double *w, int n_train, const int *train_idx,
                    Node **out_nodes, int *out_n) {
    int cap = (1 << (max_depth + 1)) - 1;
    Node *nodes = malloc(sizeof(Node) * (size_t)cap);
    int *rows = malloc(sizeof(int) * (size_t)n_train);
    Pair *buf = malloc(sizeof(Pair) * (size_t)n_train);
    if (!nodes || !rows || !buf) { free(nodes); free(rows); free(buf); return ERR_MEM; }

    memcpy(rows, train_idx, sizeof(int) * (size_t)n_train);
    Builder B = { .X = X, .n_feat = n_feat, .y = y, .w = w,
                  .max_depth = max_depth, .min_leaf = min_leaf,
                  .nodes = nodes, .n_nodes = 0, .buf = buf };

    int root = build(&B, rows, n_train, 0);
    free(rows);
    free(buf);
    if (root < 0) { free(nodes); return ERR_MEM; }

    *out_nodes = nodes;
    *out_n = B.n_nodes;
    return OK;
}

/* ======================= logistic regression ======================= */

typedef struct {
    int n_feat;
    double bias;
    double *coef, *mean, *std;
} LogModel;

static void lr_free(LogModel *m) {
    free(m->coef); free(m->mean); free(m->std);
    m->coef = m->mean = m->std = NULL;
}

static double sigmoid(double z) {
    if (z > 500.0) z = 500.0;
    if (z < -500.0) z = -500.0;
    return 1.0 / (1.0 + exp(-z));
}

static double lr_logit(const LogModel *m, const double *x_row) {
    double z = m->bias;
    for (int f = 0; f < m->n_feat; f++)
        z += m->coef[f] * ((x_row[f] - m->mean[f]) / m->std[f]);
    return z;
}

static int fit_logreg(const Spec *s, int n_feat, const double *X, const int *y,
                      const double *w, int n_train, const int *train_idx, LogModel *m) {
    int F = n_feat;
    m->n_feat = F;
    m->bias = 0.0;
    m->coef = calloc((size_t)F, sizeof(double));
    m->mean = calloc((size_t)F, sizeof(double));
    m->std  = calloc((size_t)F, sizeof(double));
    double *v = malloc(sizeof(double) * (size_t)n_train);
    double *grad = malloc(sizeof(double) * (size_t)F);
    if (!m->coef || !m->mean || !m->std || !v || !grad) {
        free(v); free(grad); lr_free(m);
        return ERR_MEM;
    }

    /* Standardize using unweighted training statistics. */
    for (int f = 0; f < F; f++) {
        double sum = 0.0, sq = 0.0;
        for (int i = 0; i < n_train; i++) {
            double x = X[(size_t)train_idx[i] * F + f];
            sum += x;
            sq += x * x;
        }
        double mu = sum / n_train;
        double var = sq / n_train - mu * mu;
        m->mean[f] = mu;
        m->std[f] = (var > 1e-12) ? sqrt(var) : 1.0;
    }

    /* Normalize sample weights over the training slice. */
    double tot = 0.0;
    for (int i = 0; i < n_train; i++) tot += w[train_idx[i]];
    if (tot <= 0.0) { free(v); free(grad); lr_free(m); return ERR_ARG; }
    for (int i = 0; i < n_train; i++) v[i] = w[train_idx[i]] / tot;

    /* Full-batch gradient descent on the weighted log-loss. Deterministic: zero init. */
    for (int it = 0; it < s->iters; it++) {
        for (int f = 0; f < F; f++) grad[f] = 0.0;
        double gb = 0.0;
        for (int i = 0; i < n_train; i++) {
            int r = train_idx[i];
            const double *xr = X + (size_t)r * F;
            double p = sigmoid(lr_logit(m, xr));
            double t = (y[r] > 0) ? 1.0 : 0.0;
            double g = v[i] * (p - t);
            gb += g;
            for (int f = 0; f < F; f++)
                grad[f] += g * ((xr[f] - m->mean[f]) / m->std[f]);
        }
        for (int f = 0; f < F; f++)
            m->coef[f] -= s->lr * (grad[f] + s->l2 * m->coef[f]);
        m->bias -= s->lr * gb;
    }

    free(v);
    free(grad);
    return OK;
}

/* ======================= model files ======================= */

typedef struct {
    int family;
    int n_feat;
    Node *nodes;       /* stump and tree */
    int n_nodes;
    LogModel lr;       /* logreg */
} Model;

static void model_free(Model *m) {
    free(m->nodes);
    m->nodes = NULL;
    lr_free(&m->lr);
}

static int save_tree(const char *path, int family, int n_feat, const Node *nodes, int n_nodes) {
    FILE *fp = fopen(path, "w");
    if (!fp) return ERR_IO;
    fprintf(fp, "WEAK 1\nfamily %d\nn_feat %d\nn_nodes %d\n", family, n_feat, n_nodes);
    for (int i = 0; i < n_nodes; i++)
        fprintf(fp, "%d %.17g %d %d %d %.17g\n", nodes[i].feat, nodes[i].thr,
                nodes[i].left, nodes[i].right, nodes[i].sign, nodes[i].prob);
    if (fclose(fp) != 0) return ERR_IO;
    return OK;
}

static int save_logreg(const char *path, const LogModel *m) {
    FILE *fp = fopen(path, "w");
    if (!fp) return ERR_IO;
    fprintf(fp, "WEAK 1\nfamily %d\nn_feat %d\nbias %.17g\n",
            FAM_LOGREG, m->n_feat, m->bias);
    fprintf(fp, "coef");
    for (int f = 0; f < m->n_feat; f++) fprintf(fp, " %.17g", m->coef[f]);
    fprintf(fp, "\nmean");
    for (int f = 0; f < m->n_feat; f++) fprintf(fp, " %.17g", m->mean[f]);
    fprintf(fp, "\nstd");
    for (int f = 0; f < m->n_feat; f++) fprintf(fp, " %.17g", m->std[f]);
    fprintf(fp, "\n");
    if (fclose(fp) != 0) return ERR_IO;
    return OK;
}

static int expect_tag(FILE *fp, const char *tag) {
    char buf[32];
    if (fscanf(fp, "%31s", buf) != 1) return 0;
    return strcmp(buf, tag) == 0;
}

static int load_model(const char *path, Model *m) {
    memset(m, 0, sizeof *m);
    FILE *fp = fopen(path, "r");
    if (!fp) return ERR_IO;

    int rc = ERR_FORMAT;
    int ver = 0;
    if (!expect_tag(fp, "WEAK") || fscanf(fp, "%d", &ver) != 1 || ver != 1) goto done;
    if (!expect_tag(fp, "family") || fscanf(fp, "%d", &m->family) != 1) goto done;
    if (!expect_tag(fp, "n_feat") || fscanf(fp, "%d", &m->n_feat) != 1 || m->n_feat <= 0) goto done;

    if (m->family == FAM_LOGREG) {
        LogModel *L = &m->lr;
        L->n_feat = m->n_feat;
        L->coef = calloc((size_t)L->n_feat, sizeof(double));
        L->mean = calloc((size_t)L->n_feat, sizeof(double));
        L->std  = calloc((size_t)L->n_feat, sizeof(double));
        if (!L->coef || !L->mean || !L->std) { rc = ERR_MEM; goto done; }

        if (!expect_tag(fp, "bias") || fscanf(fp, "%lf", &L->bias) != 1) goto done;
        if (!expect_tag(fp, "coef")) goto done;
        for (int f = 0; f < L->n_feat; f++)
            if (fscanf(fp, "%lf", &L->coef[f]) != 1) goto done;
        if (!expect_tag(fp, "mean")) goto done;
        for (int f = 0; f < L->n_feat; f++)
            if (fscanf(fp, "%lf", &L->mean[f]) != 1) goto done;
        if (!expect_tag(fp, "std")) goto done;
        for (int f = 0; f < L->n_feat; f++) {
            if (fscanf(fp, "%lf", &L->std[f]) != 1) goto done;
            if (!(L->std[f] > 0.0)) goto done;
        }
        rc = OK;
    } else if (m->family == FAM_STUMP || m->family == FAM_TREE) {
        if (!expect_tag(fp, "n_nodes") || fscanf(fp, "%d", &m->n_nodes) != 1 || m->n_nodes < 1)
            goto done;
        m->nodes = malloc(sizeof(Node) * (size_t)m->n_nodes);
        if (!m->nodes) { rc = ERR_MEM; goto done; }

        for (int i = 0; i < m->n_nodes; i++) {
            Node *nd = &m->nodes[i];
            if (fscanf(fp, "%d %lf %d %d %d %lf",
                       &nd->feat, &nd->thr, &nd->left, &nd->right, &nd->sign, &nd->prob) != 6)
                goto done;
            if (nd->feat < -1 || nd->feat >= m->n_feat) goto done;
            if (nd->sign != 1 && nd->sign != -1) goto done;
            if (nd->feat >= 0) {
                /* children must come after the parent; this rules out cycles */
                if (nd->left <= i || nd->left >= m->n_nodes) goto done;
                if (nd->right <= i || nd->right >= m->n_nodes) goto done;
            }
        }
        rc = OK;
    }

done:
    fclose(fp);
    if (rc != OK) model_free(m);
    return rc;
}

/* ======================= public API ======================= */

int train_predict(const Spec *s, const char *model_path,
                  int n_rows, int n_feat, const double *X, const int *y,
                  const double *w, int n_train, const int *train_idx,
                  int *pred_out, double *prob_out) {
    if (!s || !model_path || n_rows <= 0 || n_feat <= 0 || !X || !y || !w ||
        n_train <= 0 || !train_idx || !pred_out || !prob_out)
        return ERR_ARG;

    for (int i = 0; i < n_train; i++) {
        int r = train_idx[i];
        if (r < 0 || r >= n_rows) return ERR_ARG;
        if (!isfinite(w[r]) || w[r] < 0.0) return ERR_ARG;
    }

    int rc;
    if (s->family == FAM_LOGREG) {
        if (s->iters <= 0 || !(s->lr > 0.0) || s->l2 < 0.0) return ERR_SPEC;
        LogModel L;
        rc = fit_logreg(s, n_feat, X, y, w, n_train, train_idx, &L);
        if (rc == OK) rc = save_logreg(model_path, &L);
        lr_free(&L);
    } else if (s->family == FAM_TREE || s->family == FAM_STUMP) {
        int depth = (s->family == FAM_STUMP) ? 1 : s->max_depth;
        int min_leaf = (s->family == FAM_STUMP) ? 1 : s->min_leaf;
        if (depth < 1 || depth > MAX_TREE_DEPTH || min_leaf < 1) return ERR_SPEC;

        Node *nodes = NULL;
        int n_nodes = 0;
        rc = fit_tree(depth, min_leaf, n_feat, X, y, w, n_train, train_idx, &nodes, &n_nodes);
        if (rc == OK) rc = save_tree(model_path, s->family, n_feat, nodes, n_nodes);
        free(nodes);
    } else {
        return ERR_SPEC;
    }
    if (rc != OK) return rc;

    /* Predict through the saved file so training and inference use the same code path. */
    return predict_model(model_path, n_rows, n_feat, X, pred_out, prob_out);
}

int predict_model(const char *model_path, int n_rows, int n_feat, const double *X,
                  int *pred_out, double *prob_out) {
    if (!model_path || n_rows <= 0 || n_feat <= 0 || !X || !pred_out || !prob_out)
        return ERR_ARG;

    Model m;
    int rc = load_model(model_path, &m);
    if (rc != OK) return rc;
    if (m.n_feat != n_feat) { model_free(&m); return ERR_ARG; }

    for (int r = 0; r < n_rows; r++) {
        const double *xr = X + (size_t)r * n_feat;
        if (m.family == FAM_LOGREG) {
            double p = sigmoid(lr_logit(&m.lr, xr));
            prob_out[r] = p;
            pred_out[r] = (p >= 0.5) ? 1 : -1;
        } else {
            int id = 0;
            while (m.nodes[id].feat >= 0) {
                const Node *nd = &m.nodes[id];
                id = (xr[nd->feat] <= nd->thr) ? nd->left : nd->right;
            }
            prob_out[r] = m.nodes[id].prob;
            pred_out[r] = m.nodes[id].sign;
        }
    }

    model_free(&m);
    return OK;
}
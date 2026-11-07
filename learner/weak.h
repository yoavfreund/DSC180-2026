// weak.h
typedef struct {
    int family;        // 0 = stump, 1 = tree, 2 = logreg
    int max_depth;     // tree
    int min_leaf;      // tree
    int iters;         // logreg
    double lr;         // logreg
    double l2;         // logreg
} Spec;

typedef struct {
    double err;        // weighted error on val_idx
    double fpr, fnr;   // weighted false-positive and false-negative rates
    double w_pos_val;  // share of validation weight on y=+1
} EvalResult;

// Train on train_idx using sample weights w (any scale; C does not normalize).
// Writes the model to model_path and predictions/probabilities for ALL rows.
int train_predict(const Spec *s, const char *model_path,
                  int n_rows, int n_feat, const double *X, const int *y,
                  const double *w, int n_train, const int *train_idx,
                  int *pred_out, double *prob_out);

// Predict new rows from a saved model (used by the combiner on unseen data).
int predict_model(const char *model_path, int n_rows, int n_feat,
                  const double *X, int *pred_out, double *prob_out);

EvalResult evaluate(int n_rows, const int *pred, const int *y,
                    const double *w, int n_val, const int *val_idx);
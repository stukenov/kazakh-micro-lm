#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

typedef struct {
    int rows;
    int cols;
    double **data;
} Matrix;

Matrix* create_matrix(int rows, int cols) {
    Matrix* mat = (Matrix*)malloc(sizeof(Matrix));
    if (!mat) {
        return NULL;
    }
    mat->rows = rows;
    mat->cols = cols;
    mat->data = (double**)malloc(rows * sizeof(double*));
    if (!mat->data) {
        free(mat);
        return NULL;
    }
    for (int i = 0; i < rows; i++) {
        mat->data[i] = (double*)calloc(cols, sizeof(double));
        if (!mat->data[i]) {
            for (int j = 0; j < i; j++) {
                free(mat->data[j]);
            }
            free(mat->data);
            free(mat);
            return NULL;
        }
    }
    return mat;
}

void free_matrix(Matrix* mat) {
    if (!mat) {
        return;
    }
    for (int i = 0; i < mat->rows; i++) {
        free(mat->data[i]);
    }
    free(mat->data);
    free(mat);
}

void randomize_matrix(Matrix* mat) {
    if (!mat) {
        return;
    }
    for (int i = 0; i < mat->rows; i++) {
        for (int j = 0; j < mat->cols; j++) {
            mat->data[i][j] = ((double)rand() / RAND_MAX) * 2 - 1; // Random values between -1 and 1
        }
    }
}

Matrix* matmul(Matrix* A, Matrix* B) {
    if (!A || !B || A->cols != B->rows) {
        fprintf(stderr, "Matrix multiplication dimension mismatch.\n");
        return NULL;
    }
    Matrix* C = create_matrix(A->rows, B->cols);
    if (!C) {
        return NULL;
    }
    for (int i = 0; i < A->rows; i++) {
        for (int j = 0; j < B->cols; j++) {
            double sum = 0.0;
            for (int k = 0; k < A->cols; k++) {
                sum += A->data[i][k] * B->data[k][j];
            }
            C->data[i][j] = sum;
        }
    }
    return C;
}

Matrix* matadd(Matrix* A, Matrix* B) {
    if (!A || !B || A->rows != B->rows || A->cols != B->cols) {
        fprintf(stderr, "Matrix addition dimension mismatch.\n");
        return NULL;
    }
    Matrix* C = create_matrix(A->rows, A->cols);
    if (!C) {
        return NULL;
    }
    for (int i = 0; i < A->rows; i++) {
        for (int j = 0; j < A->cols; j++) {
            C->data[i][j] = A->data[i][j] + B->data[i][j];
        }
    }
    return C;
}

Matrix* sigmoid(Matrix* A) {
    if (!A) {
        return NULL;
    }
    Matrix* C = create_matrix(A->rows, A->cols);
    if (!C) {
        return NULL;
    }
    for (int i = 0; i < A->rows; i++) {
        for (int j = 0; j < A->cols; j++) {
            C->data[i][j] = 1.0 / (1.0 + exp(-A->data[i][j]));
        }
    }
    return C;
}

Matrix* sigmoid_derivative(Matrix* A) {
    if (!A) {
        return NULL;
    }
    Matrix* C = create_matrix(A->rows, A->cols);
    if (!C) {
        return NULL;
    }
    for (int i = 0; i < A->rows; i++) {
        for (int j = 0; j < A->cols; j++) {
            C->data[i][j] = A->data[i][j] * (1 - A->data[i][j]);
        }
    }
    return C;
}

Matrix* mat_subtract(Matrix* A, Matrix* B) {
    if (!A || !B || A->rows != B->rows || A->cols != B->cols) {
        fprintf(stderr, "Matrix subtraction dimension mismatch.\n");
        return NULL;
    }
    Matrix* C = create_matrix(A->rows, A->cols);
    if (!C) {
        return NULL;
    }
    for (int i = 0; i < A->rows; i++) {
        for (int j = 0; j < A->cols; j++) {
            C->data[i][j] = A->data[i][j] - B->data[i][j];
        }
    }
    return C;
}

Matrix* transpose(Matrix* A) {
    if (!A) {
        return NULL;
    }
    Matrix* T = create_matrix(A->cols, A->rows);
    if (!T) {
        return NULL;
    }
    for (int i = 0; i < A->rows; i++) {
        for (int j = 0; j < A->cols; j++) {
            T->data[j][i] = A->data[i][j];
        }
    }
    return T;
}

typedef struct {
    int input_size;
    int hidden_size;
    int output_size;
    Matrix* Wxh;
    Matrix* Whh;
    Matrix* Why;
    Matrix* bh;
    Matrix* by;
} RNN;

RNN* init_rnn(int input_size, int hidden_size, int output_size) {
    RNN* rnn = (RNN*)malloc(sizeof(RNN));
    if (!rnn) {
        return NULL;
    }
    rnn->input_size = input_size;
    rnn->hidden_size = hidden_size;
    rnn->output_size = output_size;

    rnn->Wxh = create_matrix(hidden_size, input_size);
    rnn->Whh = create_matrix(hidden_size, hidden_size);
    rnn->Why = create_matrix(output_size, hidden_size);
    rnn->bh = create_matrix(hidden_size, 1);
    rnn->by = create_matrix(output_size, 1);

    if (!rnn->Wxh || !rnn->Whh || !rnn->Why || !rnn->bh || !rnn->by) {
        free_rnn(rnn);
        return NULL;
    }

    randomize_matrix(rnn->Wxh);
    randomize_matrix(rnn->Whh);
    randomize_matrix(rnn->Why);
    randomize_matrix(rnn->bh);
    randomize_matrix(rnn->by);

    return rnn;
}

void free_rnn(RNN* rnn) {
    if (!rnn) {
        return;
    }
    free_matrix(rnn->Wxh);
    free_matrix(rnn->Whh);
    free_matrix(rnn->Why);
    free_matrix(rnn->bh);
    free_matrix(rnn->by);
    free(rnn);
}

Matrix* rnn_forward(RNN* rnn, Matrix* x, Matrix* h_prev) {
    if (!rnn || !x || !h_prev) {
        return NULL;
    }

    Matrix* Wxhx = matmul(rnn->Wxh, x);
    if (!Wxhx) {
        return NULL;
    }

    Matrix* Whh_h = matmul(rnn->Whh, h_prev);
    if (!Whh_h) {
        free_matrix(Wxhx);
        return NULL;
    }

    Matrix* sum1 = matadd(Wxhx, Whh_h);
    if (!sum1) {
        free_matrix(Wxhx);
        free_matrix(Whh_h);
        return NULL;
    }

    Matrix* h = matadd(sum1, rnn->bh);
    if (!h) {
        free_matrix(Wxhx);
        free_matrix(Whh_h);
        free_matrix(sum1);
        return NULL;
    }

    Matrix* h_sigmoid = sigmoid(h);
    if (!h_sigmoid) {
        free_matrix(Wxhx);
        free_matrix(Whh_h);
        free_matrix(sum1);
        free_matrix(h);
        return NULL;
    }

    free_matrix(Wxhx);
    free_matrix(Whh_h);
    free_matrix(sum1);
    free_matrix(h);
    return h_sigmoid;
}

double compute_loss(Matrix* y_pred, Matrix* y_true) {
    if (!y_pred || !y_true) {
        return -1.0;
    }
    double loss = 0.0;
    for (int i = 0; i < y_pred->rows; i++) {
        for (int j = 0; j < y_pred->cols; j++) {
            double diff = y_pred->data[i][j] - y_true->data[i][j];
            loss += diff * diff;
        }
    }
    return loss / (y_pred->rows * y_pred->cols);
}

void train_rnn(RNN* rnn, Matrix** inputs, Matrix** targets, int sequence_length, int epochs, double learning_rate) {
    if (!rnn || !inputs || !targets || sequence_length <= 0 || epochs <= 0 || learning_rate <= 0) {
        return;
    }

    for (int epoch = 0; epoch < epochs; epoch++) {
        Matrix* h_prev = create_matrix(rnn->hidden_size, 1);
        if (!h_prev) {
            return;
        }

        Matrix* loss_total = create_matrix(1, 1);
        if (!loss_total) {
            free_matrix(h_prev);
            return;
        }
        loss_total->data[0][0] = 0.0;

        for (int t = 0; t < sequence_length; t++) {
            Matrix* h = rnn_forward(rnn, inputs[t], h_prev);
            if (!h) {
                free_matrix(h_prev);
                free_matrix(loss_total);
                return;
            }

            Matrix* y_pred = matmul(rnn->Why, h);
            if (!y_pred) {
                free_matrix(h_prev);
                free_matrix(loss_total);
                free_matrix(h);
                return;
            }

            Matrix* y_with_bias = matadd(y_pred, rnn->by);
            if (!y_with_bias) {
                free_matrix(h_prev);
                free_matrix(loss_total);
                free_matrix(h);
                free_matrix(y_pred);
                return;
            }

            double loss = compute_loss(y_with_bias, targets[t]);
            loss_total->data[0][0] += loss;

            free_matrix(h_prev);
            free_matrix(y_pred);
            free_matrix(y_with_bias);
            h_prev = h;
        }

        printf("Epoch %d, Loss: %f\n", epoch + 1, loss_total->data[0][0] / sequence_length);
        free_matrix(loss_total);
        free_matrix(h_prev);
    }
}

int main(void) {
    srand((unsigned int)time(NULL));
    int input_size = 5;
    int hidden_size = 10;
    int output_size = 5;
    int sequence_length = 20;
    int epochs = 100;
    double learning_rate = 0.01;

    RNN* rnn = init_rnn(input_size, hidden_size, output_size);
    if (!rnn) {
        fprintf(stderr, "Failed to initialize RNN\n");
        return 1;
    }

    Matrix** inputs = (Matrix**)malloc(sequence_length * sizeof(Matrix*));
    Matrix** targets = (Matrix**)malloc(sequence_length * sizeof(Matrix*));
    if (!inputs || !targets) {
        free_rnn(rnn);
        free(inputs);
        free(targets);
        fprintf(stderr, "Memory allocation failed\n");
        return 1;
    }

    for (int t = 0; t < sequence_length; t++) {
        inputs[t] = create_matrix(input_size, 1);
        targets[t] = create_matrix(output_size, 1);
        if (!inputs[t] || !targets[t]) {
            for (int i = 0; i < t; i++) {
                free_matrix(inputs[i]);
                free_matrix(targets[i]);
            }
            free(inputs);
            free(targets);
            free_rnn(rnn);
            fprintf(stderr, "Memory allocation failed\n");
            return 1;
        }
        for (int i = 0; i < input_size; i++) {
            inputs[t]->data[i][0] = ((double)rand() / RAND_MAX);
        }
        for (int i = 0; i < output_size; i++) {
            targets[t]->data[i][0] = ((double)rand() / RAND_MAX);
        }
    }

    train_rnn(rnn, inputs, targets, sequence_length, epochs, learning_rate);

    for (int t = 0; t < sequence_length; t++) {
        free_matrix(inputs[t]);
        free_matrix(targets[t]);
    }
    free(inputs);
    free(targets);
    free_rnn(rnn);

    return 0;
}
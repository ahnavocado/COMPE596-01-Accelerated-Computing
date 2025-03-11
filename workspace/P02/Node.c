#include "Node.h"
#include <omp.h>
#include <time.h>
#include <stdlib.h>

Node *head = NULL;
omp_lock_t list_lock; // Global lock for thread safety

// Function to insert a node in order
void insert_in_order(int value) {
    Node *newNode = (Node *)calloc(1, sizeof(Node));
    newNode->value = value;

    omp_set_lock(&list_lock); // Locking before modification

    Node *prev = head;
    Node *p = head->next;

    while (p != NULL) {
        if (p->value >= value) break;
        prev = p;
        p = p->next;
    }

    newNode->next = p;
    newNode->prev = prev;
    prev->next = newNode;

    if (p) {
        p->prev = newNode;
    }

    omp_unset_lock(&list_lock); // Unlock after modification
}

// Function to free the list
void free_list() {
    Node *p = head;
    while (p) {
        Node *temp = p;
        p = p->next;
        free(temp);
    }
}

int main() {
    // N values in powers of 2
    int N_values[] = {1, 4, 16, 64, 256, 1024, 4096, 16384, 65536, 262144};
    int num_N = sizeof(N_values) / sizeof(N_values[0]);

    srand(time(NULL));

    // Iterate over different N values
    for (int i = 0; i < num_N; i++) {
        int N = N_values[i];

        head = (Node *)calloc(1, sizeof(Node)); // Dummy head node
        head->value = -1;
        head->next = NULL;
        head->prev = NULL;
        omp_init_lock(&list_lock);

        double start_time = omp_get_wtime();

        // 변수 선언을 루프 바깥으로 이동
        int k, value;
        Node *newNode, *p, *prev;

        #pragma omp parallel for private(k, value, newNode, p, prev)
        for (k = 1; k <= N; k++) {
            unsigned int seed = time(NULL) ^ omp_get_thread_num();
            value = rand_r(&seed) % 1000 + 1;
            newNode = (Node *)calloc(1, sizeof(Node));

            prev = head;
            p = head->next;

            while (p != NULL) {
                if (p->value >= value) break;
                prev = p;
                p = p->next;
            }

            newNode->value = value;
            newNode->next = p;
            newNode->prev = prev;
            prev->next = newNode;

            if (p) {
                p->prev = newNode;
            }
        }

        double end_time = omp_get_wtime();

        printf("N = %d, Time taken: %f seconds\n", N, end_time - start_time);

        free_list();
        omp_destroy_lock(&list_lock);
    }

    return 0;
}


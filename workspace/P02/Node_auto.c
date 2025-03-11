#include "Node.h"
#include <omp.h>
#include <time.h>

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

    // Thread counts in powers of 2
    int thread_counts[] = {4, 8, 16, 32, 64, 128, 256};
    int num_threads = sizeof(thread_counts) / sizeof(thread_counts[0]);

    srand(time(NULL));

    // Iterate over different N values
    for (int i = 0; i < num_N; i++) {
        int N = N_values[i];

        // Iterate over different thread counts
        for (int j = 0; j < num_threads; j++) {
            int num_thread = thread_counts[j];

            head = (Node *)calloc(1, sizeof(Node)); // Dummy head node
            head->value = -1;
            head->next = NULL;
            head->prev = NULL;
            omp_init_lock(&list_lock);

            double start_time = omp_get_wtime();

            #pragma omp parallel for num_threads(num_thread)
            for (int k = 1; k <= N; k++) {
                unsigned int seed = time(NULL) ^ omp_get_thread_num();
                int value = rand_r(&seed) % 1000 + 1;
                insert_in_order(value);
            }

            double end_time = omp_get_wtime();

            printf("N = %d, Threads = %d, Time taken: %f seconds\n", N, num_thread, end_time - start_time);

            free_list();
            omp_destroy_lock(&list_lock);
        }
    }

    return 0;
}


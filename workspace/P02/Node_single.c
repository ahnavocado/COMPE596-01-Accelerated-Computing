#include "Node.h"
#include <time.h>
#include <stdlib.h>
#include <stdio.h>

Node *head = NULL;

// Function to insert a node in order (Serial Version)
void insert_in_order(int value) {
    Node *newNode = (Node *)calloc(1, sizeof(Node));
    newNode->value = value;

    if (!head || head->value >= value) {
        newNode->next = head;
        newNode->prev = NULL;
        if (head) head->prev = newNode;
        head = newNode;
        return;
    }

    Node *p = head;
    while (p->next && p->next->value < value) {
        p = p->next;
    }

    newNode->next = p->next;
    newNode->prev = p;
    if (p->next) p->next->prev = newNode;
    p->next = newNode;
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

        head = NULL;  // Reset head before each test

        clock_t start = clock();

        for (int k = 1; k <= N; k++) {
            int value = rand() % 1000 + 1;
            insert_in_order(value);
        }

        clock_t end = clock();

        printf("N = %d, Time taken: %f seconds\n", N, (double)(end - start) / CLOCKS_PER_SEC);

        free_list();  // Free memory after each test
    }

    return 0;
}


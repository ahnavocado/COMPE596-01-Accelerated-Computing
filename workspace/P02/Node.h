#ifndef NODE_H
#define NODE_H

#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

// Node structure for the doubly linked list
typedef struct node {
    int value;
    struct node *next, *prev;
    omp_lock_t lock;
} Node;

// Function declarations
void insert_in_order(int value);
void print_list();
void free_list();

#endif // NODE_H


#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct node {
    int value;
    struct node *next;
} node_t;

static node_t *node_new(int value) {
    node_t *node = malloc(sizeof(*node));
    if (node == NULL) {
        fprintf(stderr, "out of memory\n");
        exit(EXIT_FAILURE);
    }
    node->value = value;
    node->next = NULL;
    return node;
}

static int clamp(int value, int min, int max) {
    if (value < min) {
        return min;
    }
    if (value > max) {
        return max;
    }
    return value;
}

int main(void) {
    const int values[] = {1, 2, 3, 4};
    node_t *head = NULL;
    for (size_t i = 0; i < sizeof(values) / sizeof(values[0]); ++i) {
        node_t *node = node_new(clamp(values[i], 0, 10));
        node->next = head;
        head = node;
    }

    int total = 0;
    for (node_t *cur = head; cur != NULL; cur = cur->next) {
        total += cur->value;
    }
    while (head != NULL) {
        node_t *next = head->next;
        free(head);
        head = next;
    }
    printf("total=%d\n", total);
    return total == 10 ? EXIT_SUCCESS : EXIT_FAILURE;
}

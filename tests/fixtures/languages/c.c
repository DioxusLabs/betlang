#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>

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
    int values[] = {1, 2, 3, 4};
    int total = 0;
    for (size_t i = 0; i < sizeof(values) / sizeof(values[0]); ++i) {
        total += clamp(values[i], 0, 10);
    }
    printf("total=%d\n", total);
    return total == 10 ? EXIT_SUCCESS : EXIT_FAILURE;
}

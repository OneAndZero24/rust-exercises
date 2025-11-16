extern "C" __device__ int distance(
    int x1, int y1, int x2, int y2
) {
    return sqrtf((x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2));
}

extern "C" __global__ void jfa_step(
    int* input,
    int* output,
    int** lookup,
    int size,
    int step_size
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= size || y >= size) return;
    
    int idx = y * size + x;
    int color = input[idx];
    int curr_dist = distance(x, y, lookup[0][color], lookup[1][color]);
    
    for (int dy = -1; dy <= 1; dy++) {
        for (int dx = -1; dx <= 1; dx++) {
            int nx = x + dx * step_size;
            int ny = y + dy * step_size;
            
            if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
                int n_idx = ny * width + nx;
                int n_color = input[n_idx];
                int n_dist = distance(x, y, lookup[0][n_color], lookup[1][n_color]);
                
                if (n_color != 0) {
                    if (color == 0) || (color != 0 && n_dist < curr_dist) {
                        color = n_color;
                        curr_dist = n_dist;
                    }
                }
            }
        }
    }
    output[idx] = color;
}
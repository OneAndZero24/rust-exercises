extern "C" __device__ float distance(
    int x1, int y1, int x2, int y2
) {
    int dx = x1 - x2;
    int dy = y1 - y2;
    return sqrtf((float)(dx * dx + dy * dy));
}

extern "C" __global__ void jfa_step(
    int* input,
    int* output,
    int* lookup,
    int size,
    int step_size
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= size || y >= size) return;
    
    int n = 10; // Number of seeds
    int idx = y * size + x;
    int color = input[idx];
    float curr_dist = (color >= 0) ? distance(x, y, lookup[color], lookup[color + n]) : 1e10f;
    
    for (int dy = -1; dy <= 1; dy++) {
        for (int dx = -1; dx <= 1; dx++) {
            int nx = x + dx * step_size;
            int ny = y + dy * step_size;
            
            if (nx >= 0 && nx < size && ny >= 0 && ny < size) {
                int n_idx = ny * size + nx;
                int n_color = input[n_idx];
                
                if (n_color >= 0) {
                    float n_dist = distance(x, y, lookup[n_color], lookup[n_color + n]);
                    if ((color < 0) || (n_dist < curr_dist)) {
                        color = n_color;
                        curr_dist = n_dist;
                    }
                }
            }
        }
    }
    output[idx] = color;
}
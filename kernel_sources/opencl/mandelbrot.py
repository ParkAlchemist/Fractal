mandelbrot_kernel_cl_f32 = """
__kernel void mandelbrot_kernel(
    const float min_x, const float max_x,
    const float min_y, const float max_y,
    __global float *iter_buf,
    const int width, const int height,
    const int max_iter,
    const int samples)
{
    int x = get_global_id(0);
    int y = get_global_id(1);
    if (x >= width || y >= height) return;
    float pixel_size_x = (max_x - min_x) / (float)width;
    float pixel_size_y = (max_y - min_y) / (float)height;
    float accum = 0.0f;
    float total_samples = (float)(samples * samples);
    for (int sx = 0; sx < samples; sx++)
    for (int sy = 0; sy < samples; sy++)
    {
        float ox = ((float)sx + 0.5f) / (float)samples;
        float oy = ((float)sy + 0.5f) / (float)samples;
        float real = min_x + (x + ox) * pixel_size_x;
        float imag = min_y + (y + oy) * pixel_size_y;
        float zr = real, zi = imag;
        int i = 0;
        for (i = 0; i < max_iter; i++)
        {
            float zr2 = zr * zr;
            float zi2 = zi * zi;
            if (zr2 + zi2 >= 4.0f) break;
            float tmp = zr2 - zi2 + real;
            zi = 2.0f * zr * zi + imag;
            zr = tmp;
        }
        float mag2 = zr*zr + zi*zi;
        if (mag2 < 1e-20f) mag2 = 1e-20f;
        float log_zn = 0.5f * log(mag2);
        float nu = log(log_zn / 0.69314718f) / 0.69314718f;
        float smooth = i < max_iter ? (float)(i + 1) - nu : (float)i;
        accum += smooth / (float)max_iter;
    }
    iter_buf[y * width + x] = accum / total_samples;
}
"""

mandelbrot_kernel_cl_f64 = """
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
__kernel void mandelbrot_kernel(
    const double min_x, const double max_x,
    const double min_y, const double max_y,
    __global double *iter_buf,
    const int width, const int height,
    const int max_iter,
    const int samples)
{
    int x = get_global_id(0);
    int y = get_global_id(1);
    if (x >= width || y >= height) return;
    double pixel_size_x = (max_x - min_x) / (double)width;
    double pixel_size_y = (max_y - min_y) / (double)height;
    double accum = 0.0;
    double total_samples = (double)(samples * samples);
    for (int sx = 0; sx < samples; sx++)
    for (int sy = 0; sy < samples; sy++)
    {
        double ox = ((double)sx + 0.5) / (double)samples;
        double oy = ((double)sy + 0.5) / (double)samples;
        double real = min_x + (x + ox) * pixel_size_x;
        double imag = min_y + (y + oy) * pixel_size_y;
        double zr = real, zi = imag;
        int i = 0;
        for (i = 0; i < max_iter; i++)
        {
            double zr2 = zr * zr;
            double zi2 = zi * zi;
            if (zr2 + zi2 >= 4.0) break;
            double tmp = zr2 - zi2 + real;
            zi = 2.0 * zr * zi + imag;
            zr = tmp;
        }
        double mag2 = zr*zr + zi*zi;
        if (mag2 < 1e-20) mag2 = 1e-20;
        double log_zn = 0.5 * log(mag2);
        double nu = log(log_zn / 0.6931471805599453) / 0.6931471805599453;
        double smooth = i < max_iter ? (double)(i + 1) - nu : (double)i;
        accum += smooth / (double)max_iter;
    }
    iter_buf[y * width + x] = accum / total_samples;
}
"""

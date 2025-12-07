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

# --- Perturbation ------------------------

mandelbrot_kernel_cl_perturb_f32 = r"""
__kernel void mandelbrot_perturb_kernel(
    const float cRef_r, const float cRef_i,
    __global const float* zref_r,
    __global const float* zref_i,
    const int refLen,
    const float min_x, const float max_x,
    const float min_y, const float max_y,
    __global float* iter_buf,
    const int width, const int height,
    const int max_iter,
    const int samples,
    const int order,
    const float wFallbackThresh
){
    int x = get_global_id(0);
    int y = get_global_id(1);
    if (x >= width || y >= height) return;

    float pix_x = (max_x - min_x) / (float)width;
    float pix_y = (max_y - min_y) / (float)height;
    float accum = 0.0f;
    float total = (float)(samples * samples);

    for (int sx = 0; sx < samples; ++sx)
    for (int sy = 0; sy < samples; ++sy) {
        float offx = ((float)sx + 0.5f) / (float)samples;
        float offy = ((float)sy + 0.5f) / (float)samples;

        float c_r = min_x + (x + offx) * pix_x;
        float c_i = min_y + (y + offy) * pix_y;
        float dc_r = c_r - cRef_r;
        float dc_i = c_i - cRef_i;

        float wr = 0.0f, wi = 0.0f;
        int n = 0;
        float zF_r = 0.0f, zF_i = 0.0f;

        while (n < max_iter) {
            float zr = zref_r[n];
            float zi = zref_i[n];
            float zrx = zr + wr;
            float zry = zi + wi;
            if (zrx*zrx + zry*zry > 4.0f) { zF_r = zrx; zF_i = zry; break; }

            float t_r = 2.0f * zr * wr - 2.0f * zi * wi;
            float t_i = 2.0f * zr * wi + 2.0f * zi * wr;
            if (order >= 2) {
                float w2r = wr*wr - wi*wi;
                float w2i = 2.0f * wr * wi;
                t_r += w2r; t_i += w2i;
            }
            wr = t_r + dc_r;
            wi = t_i + dc_i;

            if (fabs(wr) > wFallbackThresh || fabs(wi) > wFallbackThresh) {
                float zt_r = zrx, zt_i = zry;
                n += 1;
                while (n < max_iter) {
                    float zr2 = zt_r*zt_r - zt_i*zt_i + c_r;
                    float zi2 = 2.0f*zt_r*zt_i + c_i;
                    zt_r = zr2; zt_i = zi2;
                    if (zt_r*zt_r + zt_i*zt_i > 4.0f) break;
                    n += 1;
                }
                zF_r = zt_r; zF_i = zt_i;
                break;
            }
            n += 1;
        }

        float mag2 = zF_r*zF_r + zF_i*zF_i; if (mag2 < 1e-20f) mag2 = 1e-20f;
        float log_zn = 0.5f * log(mag2);
        float nu = log(log_zn / 0.69314718f) / 0.69314718f;
        float smooth = (n < max_iter) ? ((float)(n + 1) - nu) : (float)n;
        accum += smooth / (float)max_iter;
    }

    iter_buf[y * width + x] = accum / total;
}
"""

mandelbrot_kernel_cl_perturb_f64 = r"""
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
__kernel void mandelbrot_perturb_kernel(
    const double cRef_r, const double cRef_i,
    __global const double* zref_r,
    __global const double* zref_i,
    const int refLen,
    const double min_x, const double max_x,
    const double min_y, const double max_y,
    __global double* iter_buf,
    const int width, const int height,
    const int max_iter,
    const int samples,
    const int order,
    const double wFallbackThresh
){
    int x = get_global_id(0);
    int y = get_global_id(1);
    if (x >= width || y >= height) return;

    double pix_x = (max_x - min_x) / (double)width;
    double pix_y = (max_y - min_y) / (double)height;
    double accum = 0.0;
    double total = (double)(samples * samples);

    for (int sx = 0; sx < samples; ++sx)
    for (int sy = 0; sy < samples; ++sy) {
        double offx = ((double)sx + 0.5) / (double)samples;
        double offy = ((double)sy + 0.5) / (double)samples;

        double c_r = min_x + (x + offx) * pix_x;
        double c_i = min_y + (y + offy) * pix_y;
        double dc_r = c_r - cRef_r;
        double dc_i = c_i - cRef_i;

        double wr = 0.0, wi = 0.0;
        int n = 0;
        double zF_r = 0.0, zF_i = 0.0;

        while (n < max_iter) {
            double zr = zref_r[n];
            double zi = zref_i[n];
            double zrx = zr + wr;
            double zry = zi + wi;
            if (zrx*zrx + zry*zry > 4.0) { zF_r = zrx; zF_i = zry; break; }

            double t_r = 2.0 * zr * wr - 2.0 * zi * wi;
            double t_i = 2.0 * zr * wi + 2.0 * zi * wr;
            if (order >= 2) {
                double w2r = wr*wr - wi*wi;
                double w2i = 2.0 * wr * wi;
                t_r += w2r; t_i += w2i;
            }
            wr = t_r + dc_r;
            wi = t_i + dc_i;

            if (fabs(wr) > wFallbackThresh || fabs(wi) > wFallbackThresh) {
                double zt_r = zrx, zt_i = zry;
                n += 1;
                while (n < max_iter) {
                    double zr2 = zt_r*zt_r - zt_i*zt_i + c_r;
                    double zi2 = 2.0*zt_r*zt_i + c_i;
                    zt_r = zr2; zt_i = zi2;
                    if (zt_r*zt_r + zt_i*zt_i > 4.0) break;
                    n += 1;
                }
                zF_r = zt_r; zF_i = zt_i;
                break;
            }
            n += 1;
        }

        double mag2 = zF_r*zF_r + zF_i*zF_i; if (mag2 < 1e-20) mag2 = 1e-20;
        double log_zn = 0.5 * log(mag2);
        double nu = log(log_zn / 0.6931471805599453) / 0.6931471805599453;
        double smooth = (n < max_iter) ? ((double)(n + 1) - nu) : (double)n;
        accum += smooth / (double)max_iter;
    }

    iter_buf[y * width + x] = accum / total;
}
"""

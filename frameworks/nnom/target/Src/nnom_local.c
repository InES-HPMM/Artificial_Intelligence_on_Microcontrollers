/*
 * Copyright (c) 2018-2019
 * Jianjia Ma, Wearable Bio-Robotics Group (WBR)
 * majianjia@live.com
 *
 * SPDX-License-Identifier: Apache-2.0
 * 
 * Notice: 
 * Code in this file inlcudes derivative works from CMSIS, which is released under alternative license.
 * Please check the LICENSE file for detial.
 *
 * Change Logs:
 * Date           Author       Notes
 * 2019-02-05     Jianjia Ma   The first version
 * 2019-03-19     Jianjia Ma   Local C implementation partly from CMSIS-NN
 */

#include "nnom.h"
#include "nnom_local.h"

// modified from CMSIS-NN test_ref
void local_avepool_q7_HWC(const q7_t *Im_in,           // input image
                          const uint16_t dim_im_in_x,  // input image dimension x or W
                          const uint16_t dim_im_in_y,  // input image dimension y or H
                          const uint16_t ch_im_in,     // number of input image channels
                          const uint16_t dim_kernel_x, // window kernel size
                          const uint16_t dim_kernel_y, // window kernel size
                          const uint16_t padding_x,    // padding sizes
                          const uint16_t padding_y,    // padding sizes
                          const uint16_t stride_x,     // stride
                          const uint16_t stride_y,     // stride
                          const uint16_t dim_im_out_x, // output image dimension x or W
                          const uint16_t dim_im_out_y, // output image dimension y or H
                          q7_t *bufferA,               // a buffer for local storage, NULL by now
                          q7_t *Im_out)
{
    int16_t i_ch_in, i_x, i_y;
    int16_t k_x, k_y;

    for (i_ch_in = 0; i_ch_in < ch_im_in; i_ch_in++)
    {
        for (i_y = 0; i_y < dim_im_out_y; i_y++)
        {
            for (i_x = 0; i_x < dim_im_out_x; i_x++)
            {
                int sum = 0;
                int count = 0;
                for (k_y = i_y * stride_y - padding_y; k_y < i_y * stride_y - padding_y + dim_kernel_y; k_y++)
                {
                    for (k_x = i_x * stride_x - padding_x; k_x < i_x * stride_x - padding_x + dim_kernel_x; k_x++)
                    {
                        if (k_y >= 0 && k_x >= 0 && k_y < dim_im_in_y && k_x < dim_im_in_x)
                        {
                            sum += Im_in[i_ch_in + ch_im_in * (k_x + k_y * dim_im_in_x)];
                            count++;
                        }
                    }
                }
                Im_out[i_ch_in + ch_im_in * (i_x + i_y * dim_im_out_x)] = sum / count;
            }
        }
    }
}

// modified from CMSIS-NN test_ref
void local_maxpool_q7_HWC(const q7_t *Im_in,           // input image
                          const uint16_t dim_im_in_x,  // input image dimension x or W
                          const uint16_t dim_im_in_y,  // input image dimension y or H
                          const uint16_t ch_im_in,     // number of input image channels
                          const uint16_t dim_kernel_x, // window kernel size
                          const uint16_t dim_kernel_y, // window kernel size
                          const uint16_t padding_x,    // padding sizes
                          const uint16_t padding_y,    // padding sizes
                          const uint16_t stride_x,     // stride
                          const uint16_t stride_y,     // stride
                          const uint16_t dim_im_out_x, // output image dimension x or W
                          const uint16_t dim_im_out_y, // output image dimension y or H
                          q7_t *bufferA,               // a buffer for local storage, NULL by now
                          q7_t *Im_out)
{
    int16_t i_ch_in, i_x, i_y;
    int16_t k_x, k_y;

    for (i_ch_in = 0; i_ch_in < ch_im_in; i_ch_in++)
    {
        for (i_y = 0; i_y < dim_im_out_y; i_y++)
        {
            for (i_x = 0; i_x < dim_im_out_x; i_x++)
            {
                int max = -129;
                for (k_y = i_y * stride_y - padding_y; k_y < i_y * stride_y - padding_y + dim_kernel_y; k_y++)
                {
                    for (k_x = i_x * stride_x - padding_x; k_x < i_x * stride_x - padding_x + dim_kernel_x; k_x++)
                    {
                        if (k_y >= 0 && k_x >= 0 && k_y < dim_im_in_y && k_x < dim_im_in_x)
                        {
                            if (Im_in[i_ch_in + ch_im_in * (k_x + k_y * dim_im_in_x)] > max)
                            {
                                max = Im_in[i_ch_in + ch_im_in * (k_x + k_y * dim_im_in_x)];
                            }
                        }
                    }
                }
                Im_out[i_ch_in + ch_im_in * (i_x + i_y * dim_im_out_x)] = max;
            }
        }
    }
}

// temporary for the thesis
// shift according to the maximum
int32_t local_sumpool_q7_HWC(const q7_t *Im_in,           // input image
                             const uint16_t dim_im_in_x,  // input image dimension x or W
                             const uint16_t dim_im_in_y,  // input image dimension y or H
                             const uint16_t ch_im_in,     // number of input image channels
                             const uint16_t dim_kernel_x, // window kernel size
                             const uint16_t dim_kernel_y, // window kernel size
                             const uint16_t padding_x,    // padding sizes
                             const uint16_t padding_y,    // padding sizes
                             const uint16_t stride_x,     // stride
                             const uint16_t stride_y,     // stride
                             const uint16_t dim_im_out_x, // output image dimension x or W
                             const uint16_t dim_im_out_y, // output image dimension y or H
                             q7_t *bufferA,               // a buffer for local storage, size = 4*output_size
                             q7_t *Im_out)
{
    int16_t i_ch_in, i_x, i_y;
    int16_t k_x, k_y;
    int32_t *buf = (int32_t *)bufferA;
	// stage2
    int32_t max_abs = 0;
    int32_t output_shift;
    size_t output_size = dim_im_out_x * dim_im_out_x * ch_im_in;

    // save in 32bit
    for (i_ch_in = 0; i_ch_in < ch_im_in; i_ch_in++)
    {
        for (i_y = 0; i_y < dim_im_out_y; i_y++)
        {
            for (i_x = 0; i_x < dim_im_out_x; i_x++)
            {
                int sum = 0;
                for (k_y = i_y * stride_y - padding_y; k_y < i_y * stride_y - padding_y + dim_kernel_y; k_y++)
                {
                    for (k_x = i_x * stride_x - padding_x; k_x < i_x * stride_x - padding_x + dim_kernel_x; k_x++)
                    {
                        if (k_y >= 0 && k_x >= 0 && k_y < dim_im_in_y && k_x < dim_im_in_x)
                        {
                            sum += Im_in[i_ch_in + ch_im_in * (k_x + k_y * dim_im_in_x)];
                        }
                    }
                }
                // 32bit
                buf[i_ch_in + ch_im_in * (i_x + i_y * dim_im_out_x)] = sum;
            }
        }
    }

    // find max amount results
    for (int i = 0; i < output_size; i++)
    {
        int32_t val = buf[i];
        if (val < 0)
            val = -val;
        if (val > max_abs)
            max_abs = val;
    }
    // find best shift to cover the max
    for (output_shift = 0;; output_shift++)
    {
        if (127 * (1 + output_shift) >= max_abs)
            break;
    }

    // shift the results
    for (int i = 0; i < output_size; i++)
    {
        Im_out[i] = buf[i] >> output_shift;
    }
    return output_shift;
}

// customised up sample pooling
void local_up_sampling_q7_HWC(const q7_t *Im_in,       // input image
                          const uint16_t dim_im_in_x,  // input image dimension x or W
                          const uint16_t dim_im_in_y,  // input image dimension y or H
                          const uint16_t ch_im_in,     // number of input image channels
                          const uint16_t dim_kernel_x, // window kernel size
                          const uint16_t dim_kernel_y, // window kernel size
                          const uint16_t dim_im_out_x, // output image dimension x or W
                          const uint16_t dim_im_out_y, // output image dimension y or H
                          q7_t *bufferA,               // a buffer for local storage, NULL by now
                          q7_t *Im_out)
{
    int16_t i_x, i_y;


    // for loop for each pixel in input image.
    for (i_y = 0; i_y < dim_im_in_x; i_y++)
    {
        for (i_x = 0; i_x < dim_im_in_y; i_x++)
        {
            // copy all the channels together. 
            const q7_t *pin = Im_in + (i_y * dim_im_in_x + i_x ) * ch_im_in;
            q7_t *pout = Im_out + (i_y * dim_im_in_x * dim_kernel_x * dim_kernel_y + i_x * dim_kernel_y) * ch_im_in;

            // cpy along x axis
            for(int i = 0; i<dim_kernel_x; i++)
                memcpy(pout + i*ch_im_in, pin, ch_im_in);
            // duplicate the copied x data into y axis. 
            for(int i = 0; i<dim_kernel_y-1; i++)
                memcpy(pout + (i+1) * ch_im_in * dim_im_in_x * dim_kernel_x, pout, ch_im_in * dim_kernel_x);
        }
    }
}



//
/// ******************* convolutional *************************
//

void local_convolve_HWC_q7_nonsquare(const q7_t *Im_in,                                                 // input image
                                     const uint16_t dim_im_in_x,                                        // input image dimention x
                                     const uint16_t dim_im_in_y,                                        // input image dimention y
                                     const uint16_t ch_im_in,                                           // number of input image channels
                                     const q7_t *wt,                                                    // kernel weights
                                     const uint16_t ch_im_out,                                          // number of filters, i.e., output image channels
                                     const uint16_t dim_kernel_x,                                       // filter kernel size x
                                     const uint16_t dim_kernel_y,                                       // filter kernel size y
                                     const uint16_t padding_x,                                          // padding sizes x
                                     const uint16_t padding_y,                                          // padding sizes y
                                     const uint16_t stride_x,                                           // stride x
                                     const uint16_t stride_y,                                           // stride y
                                     const q7_t *bias,                                                  // bias
                                     const uint16_t bias_shift, const uint16_t out_shift, q7_t *Im_out, // output image
                                     const uint16_t dim_im_out_x,                                       // output image dimension x
                                     const uint16_t dim_im_out_y,                                       // output image dimension y
                                     q15_t *bufferA,                                                    //buffer space for input
                                     q7_t *bufferB                                                      //buffer space for output
)
{
    int i, j, k, l, m, n;
    int conv_out;
    int in_row, in_col;

    for (i = 0; i < ch_im_out; i++)
    {
        for (j = 0; j < dim_im_out_y; j++)
        {
            for (k = 0; k < dim_im_out_x; k++)
            {
#ifndef NNOM_TRUNCATE
                conv_out = ((q31_t)(bias[i]) << bias_shift) + (0x1 << (out_shift - 1));
#else
                conv_out = bias[i] << bias_shift;
#endif
                for (m = 0; m < dim_kernel_y; m++)
                {
                    for (n = 0; n < dim_kernel_x; n++)
                    {
                        // if-for implementation
                        in_row = stride_y * j + m - padding_y;
                        in_col = stride_x * k + n - padding_x;
                        if (in_row >= 0 && in_col >= 0 && in_row < dim_im_in_y && in_col < dim_im_in_x)
                        {
                            for (l = 0; l < ch_im_in; l++)
                            {
                                conv_out += Im_in[(in_row * dim_im_in_x + in_col) * ch_im_in + l] *
                                            wt[i * ch_im_in * dim_kernel_y * dim_kernel_x + (m * dim_kernel_x + n) * ch_im_in +
                                               l];
                            }
                        }
                    }
                }
                Im_out[i + (j * dim_im_out_x + k) * ch_im_out] = (q7_t)__NNOM_SSAT((conv_out >> out_shift), 8);
            }
        }
    }
}

void local_depthwise_separable_conv_HWC_q7_nonsquare(const q7_t *Im_in,           // input image
                                                     const uint16_t dim_im_in_x,  // input image dimention x
                                                     const uint16_t dim_im_in_y,  // input image dimention y
                                                     const uint16_t ch_im_in,     // number of input image channels
                                                     const q7_t *wt,              // kernel weights
                                                     const uint16_t ch_im_out,    // number of filters, i.e., output image channels
                                                     const uint16_t dim_kernel_x, // filter kernel size x
                                                     const uint16_t dim_kernel_y, // filter kernel size y
                                                     const uint16_t padding_x,    // padding sizes x
                                                     const uint16_t padding_y,    // padding sizes y
                                                     const uint16_t stride_x,     // stride x
                                                     const uint16_t stride_y,     // stride y
                                                     const q7_t *bias,            // bias
                                                     const uint16_t bias_shift,   // amount of left-shift for bias
                                                     const uint16_t out_shift,    // amount of right-shift for output
                                                     q7_t *Im_out,                // output image
                                                     const uint16_t dim_im_out_x, // output image dimension x
                                                     const uint16_t dim_im_out_y, // output image dimension y
                                                     q15_t *bufferA,              //buffer space for input
                                                     q7_t *bufferB                //buffer space for output
)
{
    int i_out_y, i_out_x, i_ch_out;
    int i_ker_y, i_ker_x;
    for (i_out_y = 0; i_out_y < dim_im_out_y; i_out_y++)
    {
        for (i_out_x = 0; i_out_x < dim_im_out_x; i_out_x++)
        {
            for (i_ch_out = 0; i_ch_out < ch_im_out; i_ch_out++)
            {
                // for each output
#ifndef NNOM_TRUNCATE
                int conv_out = (bias[i_ch_out] << bias_shift) + (0x1 << (out_shift - 1));
#else
                int conv_out = bias[i_ch_out] << bias_shift;
#endif
                for (i_ker_y = 0; i_ker_y < dim_kernel_y; i_ker_y++)
                {
                    for (i_ker_x = 0; i_ker_x < dim_kernel_x; i_ker_x++)
                    {
                        int in_row = stride_y * i_out_y + i_ker_y - padding_y;
                        int in_col = stride_x * i_out_x + i_ker_x - padding_x;
                        if (in_row >= 0 && in_col >= 0 && in_row < dim_im_in_y && in_col < dim_im_in_x)
                        {
                            conv_out += Im_in[(in_row * dim_im_in_x + in_col) * ch_im_in + i_ch_out] *
                                        wt[(i_ker_y * dim_kernel_x + i_ker_x) * ch_im_out + i_ch_out];
                        }
                    }
                }
                Im_out[(i_out_y * dim_im_out_x + i_out_x) * ch_im_out + i_ch_out] =
                    (q7_t)__NNOM_SSAT((conv_out >> out_shift), 8);
            }
        }
    }
}

void local_fully_connected_q7_opt(const q7_t *pV,               // pointer to vector
                                  const q7_t *pM,               // pointer to matrix
                                  const uint16_t dim_vec,       // length of the vector
                                  const uint16_t num_of_rows,   // numCol of A
                                  const uint16_t bias_shift,    // amount of left-shift for bias
                                  const uint16_t out_shift,     // amount of right-shift for output
                                  const q7_t *bias, q7_t *pOut, // output operand
                                  q15_t *vec_buffer)
{

    uint16_t rowCnt = num_of_rows >> 2;
    const q7_t *pB = pM;
    const q7_t *pA;
    q7_t *pO = pOut;
    const q7_t *pBias = bias;

    while (rowCnt)
    {
        pA = pV;
#ifndef NNOM_TRUNCATE
        q31_t sum = (*pBias++ << bias_shift) + (0x1 << (out_shift - 1));
        q31_t sum2 = (*pBias++ << bias_shift) + (0x1 << (out_shift - 1));
        q31_t sum3 = (*pBias++ << bias_shift) + (0x1 << (out_shift - 1));
        q31_t sum4 = (*pBias++ << bias_shift) + (0x1 << (out_shift - 1));
#else
        q31_t sum = *pBias++ << bias_shift;
        q31_t sum2 = *pBias++ << bias_shift;
        q31_t sum3 = *pBias++ << bias_shift;
        q31_t sum4 = *pBias++ << bias_shift;
#endif

        uint16_t colCnt = dim_vec >> 2;

        while (colCnt)
        {
            q7_t inA1 = *pA++;
            q7_t inA3 = *pA++;
            q7_t inA2 = *pA++;
            q7_t inA4 = *pA++;

            q7_t inB1 = *pB++;
            q7_t inB3 = *pB++;
            q7_t inB2 = *pB++;
            q7_t inB4 = *pB++;

            sum += inA1 * inB1 + inA2 * inB2;
            sum2 += inA1 * inB3 + inA2 * inB4;

            inB1 = *pB++;
            inB3 = *pB++;
            inB2 = *pB++;
            inB4 = *pB++;

            sum3 += inA1 * inB1 + inA2 * inB2;
            sum4 += inA1 * inB3 + inA2 * inB4;

            inB1 = *pB++;
            inB3 = *pB++;
            inB2 = *pB++;
            inB4 = *pB++;

            sum += inA3 * inB1 + inA4 * inB2;
            sum2 += inA3 * inB3 + inA4 * inB4;

            inB1 = *pB++;
            inB3 = *pB++;
            inB2 = *pB++;
            inB4 = *pB++;

            sum3 += inA3 * inB1 + inA4 * inB2;
            sum4 += inA3 * inB3 + inA4 * inB4;

            colCnt--;
        }
        colCnt = dim_vec & 0x3;
        while (colCnt)
        {
            q7_t inA = *pA++;
            q7_t inB = *pB++;
            sum += inA * inB;
            inB = *pB++;
            sum2 += inA * inB;
            inB = *pB++;
            sum3 += inA * inB;
            inB = *pB++;
            sum4 += inA * inB;

            colCnt--;
        }
        *pO++ = (q7_t)__NNOM_SSAT((sum >> out_shift), 8);
        *pO++ = (q7_t)__NNOM_SSAT((sum2 >> out_shift), 8);
        *pO++ = (q7_t)__NNOM_SSAT((sum3 >> out_shift), 8);
        *pO++ = (q7_t)__NNOM_SSAT((sum4 >> out_shift), 8);

        rowCnt--;
    }

    rowCnt = num_of_rows & 0x3;

    while (rowCnt)
    {
        pA = pV;
#ifndef NNOM_TRUNCATE
        int ip_out = (*pBias++ << bias_shift) + (0x1 << (out_shift - 1));
#else
        int ip_out = *pBias++ << bias_shift;
#endif
        for (int j = 0; j < dim_vec; j++)
        {
            q7_t inA = *pA++;
            q7_t inB = *pB++;
            ip_out += inA * inB;
        }
        *pO++ = (q7_t)__NNOM_SSAT((ip_out >> out_shift), 8);

        rowCnt--;
    }
}

void local_fully_connected_q7(const q7_t *pV,               // pointer to vector
                              const q7_t *pM,               // pointer to matrix
                              const uint16_t dim_vec,       // length of the vector
                              const uint16_t num_of_rows,   // numCol of A
                              const uint16_t bias_shift,    // amount of left-shift for bias
                              const uint16_t out_shift,     // amount of right-shift for output
                              const q7_t *bias, q7_t *pOut, // output operand
                              q15_t *vec_buffer)
{
    for (int i = 0; i < num_of_rows; i++)
    {
#ifndef NNOM_TRUNCATE
        int ip_out = (bias[i] << bias_shift) + (0x1 << (out_shift - 1));
#else
        int ip_out = bias[i] << bias_shift;
#endif
        for (int j = 0; j < dim_vec; j++)
        {
            ip_out += pV[j] * pM[i * dim_vec + j];
        }
        pOut[i] = (q7_t)__NNOM_SSAT((ip_out >> out_shift), 8);
    }
}


void local_softmax_q7(const q7_t *vec_in, const uint32_t dim_vec, q7_t *p_out)
{
    q31_t sum;
    int32_t i;
    uint8_t shift;
    q15_t base;
    base = -257;

    /* We first search for the maximum */
    for (i = 0; i < dim_vec; i++)
    {
        if (vec_in[i] > base)
        {
            base = vec_in[i];
        }
    }

    /* 
     * So the base is set to max-8, meaning 
     * that we ignore really small values. 
     * anyway, they will be 0 after shrinking to q7_t.
     */
    base = base - 8;

    sum = 0;

    for (i = 0; i < dim_vec; i++)
    {
        if (vec_in[i] > base)
        {
            shift = (uint8_t)__NNOM_USAT(vec_in[i] - base, 5);
            sum += 0x1 << shift;
        }
    }

    /* This is effectively (0x1 << 20) / sum */
    int output_base = 0x100000 / sum;

    /* 
     * Final confidence will be output_base >> ( 13 - (vec_in[i] - base) )
     * so 128 (0x1<<7) -> 100% confidence when sum = 0x1 << 8, output_base = 0x1 << 12 
     * and vec_in[i]-base = 8
     */
    for (i = 0; i < dim_vec; i++)
    {
        if (vec_in[i] > base)
        {
            /* Here minimum value of 13+base-vec_in[i] will be 5 */
            shift = (uint8_t)__NNOM_USAT(13 + base - vec_in[i], 5);
            p_out[i] = (q7_t)__NNOM_SSAT((output_base >> shift), 8);
        }
        else
        {
            p_out[i] = 0;
        }
    }
}

void local_sigmoid_q7(q7_t *data, uint32_t size, uint16_t int_width)
{
    uint32_t i = size;
    q7_t *pIn = data;
    q7_t *pOut = data;
    q7_t in;
    q7_t out;
    uint16_t shift_size = 3 - int_width;
	// saturation if int bit too large
	if(int_width > 3)
	{
		while (i)
		{
			if(*pIn++ > 0)
				*pOut++ = 127;
			else
				*pOut++ = 0;
			i--;
		}
	}
	// otherwise search table
	else
	{
		while (i)
		{
			in = *pIn++;
			out = nnom_sigmoid_table_q7[(uint8_t)(in >> shift_size)];
			*pOut++ = out;
			i--;
		}
    }
}

void local_tanh_q7(q7_t *data, uint32_t size, uint16_t int_width)
{
    uint32_t i = size;
    q7_t *pIn = data;
    q7_t *pOut = data;
    q7_t in;
	q7_t out;
	uint16_t shift_size = 3 - int_width;
	
	// saturation if int bit too large
	if(int_width > 3)
	{
		while (i)
		{
			in = *pIn++;
			if(in > 0)
				*pOut++ = 127;
			else if ( in == 0)
				*pOut++ = 0;
			else
				*pOut++ = -128;
			i--;
		}
	}
	// otherwise search table
	else
	{
		while (i)
		{
			in = *pIn++;
			out = nnom_tanh_table_q7[(uint8_t)(in >> shift_size)];
			*pOut++ = out;
			i--;
		}
	}
}
void local_relu_q7(q7_t *data, uint32_t size)
{
    uint32_t i;

    for (i = 0; i < size; i++)
    {
        if (data[i] < 0)
            data[i] = 0;
    }
}

// matrix ops
void local_mult_q7(q7_t *pSrcA,
                   q7_t *pSrcB,
                   q7_t *pDst,
                   const uint16_t out_shift,
                   uint32_t blockSize)
{
    uint32_t i;

    for (i = 0; i < blockSize; i++)
    {
        q31_t product = pSrcA[i] * pSrcB[i];
#ifndef NNOM_TRUNCATE
        pDst[i] = (q7_t)__NNOM_SSAT((product + (0x1 << (out_shift - 1))) >> out_shift, 8);
#else
        pDst[i] = (q7_t)__NNOM_SSAT(product >> out_shift, 8);
#endif
    }
}

// own
void local_add_q7(q7_t *pSrcA,
                  q7_t *pSrcB,
                  q7_t *pDst,
                  const uint16_t out_shift,
                  uint32_t blockSize)
{
    uint32_t i;

    for (i = 0; i < blockSize; i++)
    {
        q31_t sum = pSrcA[i] + pSrcB[i];
#ifndef NNOM_TRUNCATE
        pDst[i] = (q7_t)__NNOM_SSAT((sum + (0x1 << (out_shift - 1))) >> out_shift, 8);
#else
        pDst[i] = (q7_t)__NNOM_SSAT(sum >> out_shift, 8);
#endif
    }
}

// own
void local_sub_q7(q7_t *pSrcA,
                  q7_t *pSrcB,
                  q7_t *pDst,
                  const uint16_t out_shift,
                  uint32_t blockSize)
{
    uint32_t i;

    for (i = 0; i < blockSize; i++)
    {
        q31_t sub = pSrcA[i] - pSrcB[i];
#ifndef NNOM_TRUNCATE
        pDst[i] = (q7_t)__NNOM_SSAT((sub + (0x1 << (out_shift - 1))) >> out_shift, 8);
#else
        pDst[i] = (q7_t)__NNOM_SSAT(sub >> out_shift, 8);
#endif
    }
}

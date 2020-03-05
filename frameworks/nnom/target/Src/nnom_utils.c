/*
 * Copyright (c) 2018-2019
 * Jianjia Ma, Wearable Bio-Robotics Group (WBR)
 * majianjia@live.com
 *
 * SPDX-License-Identifier: Apache-2.0
 *
 * Change Logs:
 * Date           Author       Notes
 * 2019-02-05     Jianjia Ma   The first version
 */

#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <stdbool.h>
#include "nnom.h"
#include "nnom_utils.h"

static nnom_predic_t *_predic_create_instance(nnom_model_t *m, size_t label_num, size_t top_k_size)
{
	nnom_predic_t *pre;
	uint8_t *p;
	size_t mem_size = 0;
	
	mem_size += nnom_alignto(label_num * label_num * 2, 4); // confusion_mat
	mem_size += top_k_size * 4;						   		// top_k
	mem_size += nnom_alignto(sizeof(nnom_predic_t), 4);

	// we dont use nnom_mem(), we dont count the memory
	p = nnom_malloc(mem_size);
	if (!p)
		return NULL;
	nnom_memset(p, 0, mem_size);

	pre = (nnom_predic_t *)p;
	pre->confusion_mat = (uint16_t *)(p + nnom_alignto(sizeof(nnom_predic_t), 4));
	pre->top_k = (uint32_t *)(p + nnom_alignto(sizeof(nnom_predic_t), 4) + nnom_alignto(label_num * label_num * 2, 4));

	// config
	pre->label_num = label_num;
	pre->top_k_size = top_k_size;
	pre->predic_count = 0;

	// run
	pre->model = m;

	pre->t_run_total = 0;	// model running time in total
	pre->t_predic_start = 0; // when it is initial
	pre->t_predic_total = 0; // total time of the whole test

	return pre;
}

// create a prediction
// input model, the buf pointer to the softwmax output (Temporary, this can be extract from model)
// the size of softmax output (the num of lable)
// the top k that wants to record.
nnom_predic_t *prediction_create(nnom_model_t *m, int8_t *buf_prediction, size_t label_num, size_t top_k_size)
{
	nnom_predic_t *pre = _predic_create_instance(m, label_num, top_k_size);
	if (!pre)
		return NULL;
	if (!m)
	{
		nnom_free(pre);
		return NULL;
	}

	// set the output buffer of model to the prediction instance
	pre->buf_prediction = buf_prediction;

	// mark start time.
	pre->t_predic_start = nnom_ms_get();

	return pre;
}

// after a new data is set in input
// feed data to prediction
// input the current label, (range from 0 to total number of label -1)
// (the current input data should be set by user manully to the input buffer of the model.)
int32_t prediction_run(nnom_predic_t *pre, uint32_t label)
{
	int max_val;
	int max_index;
	uint32_t true_ranking = 0;
	uint32_t start;

	if (!pre)
		return NN_ARGUMENT_ERROR;

	// now run model
	start = nnom_ms_get();
	model_run(pre->model);
	pre->t_run_total += nnom_ms_get() - start;

	// find how many prediction is bigger than the ground true.
	// Raning rules, same as tensorflow. however, predictions in MCU is more frequencly to have equal probability since it is using fixed-point.
	// if ranking is 1, 2, =2(true), 4, 5, 6. the result will be top 3.
	// if ranking is 1, 2(true), =2, 4, 5, 6. the result will be top 2.
	// find the ranking of the prediced label.
	for (uint32_t j = 0; j < pre->label_num; j++)
	{
		if (j == label)
			continue;
		if (pre->buf_prediction[label] < pre->buf_prediction[j])
			true_ranking++;
		// while value[label] = value[j]. only when label > j, label is the second of j
		else if (pre->buf_prediction[label] == pre->buf_prediction[j] && j < label)
			true_ranking++;
	}

	if (true_ranking < pre->top_k_size)
		pre->top_k[true_ranking]++;

	// Find top 1 and return the current prediction.
	// If there are several maximum prediction, return the first one.
	max_val = pre->buf_prediction[0];
	max_index = 0;
	for (uint32_t j = 1; j < pre->label_num; j++)
	{
		if (pre->buf_prediction[j] > max_val)
		{
			max_val = pre->buf_prediction[j];
			max_index = j;
		}
	}

	// fill confusion matrix
	pre->confusion_mat[label * pre->label_num + max_index] += 1;

	// prediction count
	pre->predic_count++;

	// return the prediction
	return max_index;
}

void prediction_end(nnom_predic_t *pre)
{
	if (!pre)
		return;
	pre->t_predic_total = nnom_ms_get() - pre->t_predic_start;
}

void predicetion_delete(nnom_predic_t *pre)
{
	if (!pre)
		return;
	nnom_free(pre);
}

void prediction_matrix(nnom_predic_t *pre)
{
	if (!pre)
		return;
	// print titles
	NNOM_LOG("\nConfusion matrix:\n");
	NNOM_LOG("predic");
	for (int i = 0; i < pre->label_num; i++)
	{
		NNOM_LOG("%6d", i);
	}
	NNOM_LOG("\n");
	NNOM_LOG("actual\n");
	// print the matrix
	for (int i = 0; i < pre->label_num; i++)
	{
		uint32_t row_total = 0;

		NNOM_LOG(" %3d |", i);
		for (int j = 0; j < pre->label_num; j++)
		{
			row_total += pre->confusion_mat[i * pre->label_num + j];
			NNOM_LOG("%6d", pre->confusion_mat[i * pre->label_num + j]);
		}
		NNOM_LOG("   |%4d%%\n", pre->confusion_mat[i * pre->label_num + i] * 100 / row_total);
		row_total = 0;
	}
	NNOM_LOG("\n");
}

// top-k
void prediction_top_k(nnom_predic_t *pre)
{
	uint32_t top = 0;
	if (!pre)
		return;

	for (int i = 0; i < pre->top_k_size; i++)
	{
		top += pre->top_k[i];
		if (top != pre->predic_count)
			NNOM_LOG("Top %d Accuracy: %d.%02d%% \n", i + 1, (top * 100) / pre->predic_count,
					((top * 100 * 100) / pre->predic_count)%100);
		else
			NNOM_LOG("Top %d Accuracy: 100%% \n", i + 1);
	}
}

// this function is to print sumarry
void prediction_summary(nnom_predic_t *pre)
{
	if (!pre)
		return;
	// sumamry
	NNOM_LOG("\nPrediction summary:\n");
	NNOM_LOG("Test frames: %d\n", pre->predic_count);
	NNOM_LOG("Test running time: %d sec\n", pre->t_predic_total / 1000);
	NNOM_LOG("Model running time: %d ms\n", pre->t_run_total);
	NNOM_LOG("Average prediction time: %d us\n", (pre->t_run_total * 1000) / pre->predic_count);
	NNOM_LOG("Average effeciency: %d.%02d ops/us\n", (int)(((uint64_t)pre->model->total_ops * pre->predic_count) / (pre->t_run_total * 1000)),
			(int)(((uint64_t)pre->model->total_ops * pre->predic_count)*100 / (pre->t_run_total * 1000))%100);
	NNOM_LOG("Average frame rate: %d.%d Hz\n", 1000 / (pre->t_run_total / pre->predic_count),
			(1000*10 / (pre->t_run_total / pre->predic_count))%10);

	// print top-k
	prediction_top_k(pre);

	// print confusion matrix
	prediction_matrix(pre);
}

// stand alone prediction API
// this api test one set of data, return the prediction
nnom_status_t nnom_predic(nnom_model_t *m, uint32_t *label, float *prob)
{
	int32_t max_val, max_index, sum;
	int8_t *output;

	if (!m)
		return NN_ARGUMENT_ERROR;

	model_run(m);

	// get the output memory
	output = m->tail->out->mem->blk;

	// Top 1
	max_val = output[0];
	max_index = 0;
	sum = max_val;
	for (uint32_t i = 1; i < shape_size(&m->tail->out->shape); i++)
	{
		if (output[i] > max_val)
		{
			max_val = output[i];
			max_index = i;
		}
		sum += output[i];
	}
	// send results
	*label = max_index;
	*prob  = (float)max_val/(float)sum; 
	
	return NN_SUCCESS;
}

static void layer_stat(nnom_layer_t *layer)
{
	// layer stat
	NNOM_LOG(" %10s -  %8d      ",
		   (char *)&default_layer_names[layer->type],
		   layer->stat.time);

	// MAC operation
	if(layer->stat.macc == 0)
		NNOM_LOG("            ");
	else if (layer->stat.macc < 10000)
		NNOM_LOG("%7d     ", layer->stat.macc);
	else if (layer->stat.macc < 1000*1000)
		NNOM_LOG("%6dk     ", layer->stat.macc/1000);
	else if (layer->stat.macc < 1000*1000*1000)
		NNOM_LOG("%3d.%02dM     ", layer->stat.macc/(1000*1000), layer->stat.macc%(1000*1000)/(10*1000)); // xxx.xx M
	else
		NNOM_LOG("%3d.%02dG     ", layer->stat.macc/(1000*1000*1000), layer->stat.macc%(1000*1000*1000)/(10*1000*1000)); // xxx.xx G

	// layer efficiency
	if (layer->stat.macc != 0)
		NNOM_LOG("%d.%02d\n", layer->stat.macc / layer->stat.time, (layer->stat.macc * 100) / (layer->stat.time) % 100);
	else
		NNOM_LOG("\n");
}

void model_stat(nnom_model_t *m)
{
	size_t total_ops = 0;
	size_t total_time = 0;
	nnom_layer_t *layer;
	size_t run_num = 0;

	if (!m)
		return;

	layer = m->head;

	NNOM_LOG("\nPrint running stat..\n");
	NNOM_LOG("Layer(#)        -   Time(us)     ops(MACs)   ops/us \n");
	NNOM_LOG("--------------------------------------------------------\n");
	while (layer)
	{
		run_num++;
		NNOM_LOG("#%-3d", run_num);
		total_ops += layer->stat.macc;
		total_time += layer->stat.time;
		layer_stat(layer);
		if (layer->shortcut == NULL)
			break;
		layer = layer->shortcut;
	}
	NNOM_LOG("\nSummary:\n");
	NNOM_LOG("Total ops (MAC): %d", total_ops);
	NNOM_LOG("(%d.%02dM)\n", total_ops/(1000*1000), total_ops%(1000*1000)/(10000));
	NNOM_LOG("Prediction time :%dus\n", total_time);
	NNOM_LOG("Efficiency %d.%02d ops/us\n",
		   (total_ops / total_time),
		   (total_ops * 100) / (total_time) % 100);
}

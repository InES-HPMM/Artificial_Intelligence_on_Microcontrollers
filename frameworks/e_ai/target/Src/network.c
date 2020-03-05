/***********************************************************************************************************************
* DISCLAIMER
* This software is supplied by Renesas Electronics Corporation and is only intended for use with Renesas products. No 
* other uses are authorized. This software is owned by Renesas Electronics Corporation and is protected under all 
* applicable laws, including copyright laws. 
* THIS SOFTWARE IS PROVIDED 'AS IS' AND RENESAS MAKES NO WARRANTIES REGARDING
* THIS SOFTWARE, WHETHER EXPRESS, IMPLIED OR STATUTORY, INCLUDING BUT NOT LIMITED TO WARRANTIES OF MERCHANTABILITY, 
* FITNESS FOR A PARTICULAR PURPOSE AND NON-INFRINGEMENT. ALL SUCH WARRANTIES ARE EXPRESSLY DISCLAIMED. TO THE MAXIMUM 
* EXTENT PERMITTED NOT PROHIBITED BY LAW, NEITHER RENESAS ELECTRONICS CORPORATION NOR ANY OF ITS AFFILIATED COMPANIES 
* SHALL BE LIABLE FOR ANY DIRECT, INDIRECT, SPECIAL, INCIDENTAL OR CONSEQUENTIAL DAMAGES FOR ANY REASON RELATED TO THIS 
* SOFTWARE, EVEN IF RENESAS OR ITS AFFILIATES HAVE BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGES.
* Renesas reserves the right, without notice, to make changes to this software and to discontinue the availability of 
* this software. By using this software, you agree to the additional terms and conditions found by accessing the 
* following link:
* http://www.renesas.com/disclaimer 
*
* Copyright (C) 2017 Renesas Electronics Corporation. All rights reserved.    
***********************************************************************************************************************/
/***********************************************************************************************************************
* File Name    : network.c
* Version      : 1.00
* Description  : Definitions of all functions
***********************************************************************************************************************/
/**********************************************************************************************************************
* History : DD.MM.YYYY Version  Description
*         : 16.06.2017 1.00     First Release
***********************************************************************************************************************/

#include <math.h>
#include "Typedef.h"
/*
transpose :
	Performs matrix transpose by TsInterchanging rows and columns
Parameters :
	dData	- array of input data
	dOut	- placeholder for the output
	iShapes	- dimensions of input array (i.e., data)
*/
void transpose( TPrecision *dData,TPrecision *dOut,TsInt* iShapes){
	TsInt iRow,iColumn;
	TsInt row = iShapes[2];
	TsInt col = iShapes[3];
	
	for(iRow=0; iRow<row; iRow++)
	{
	  for(iColumn=0; iColumn<col; iColumn++)
	  {
		  dOut[(iColumn*row)+iRow] = dData[(iRow*col)+iColumn];
	  }
	}
}

/*
innerproduct :
	- Fully connected layer
	- Performs dot product of data and weights and add them up with biases
	  (Matrix Multiplication of data and weights and addition of biases)
Parameters :
	data		- Array of input data
	weight_trans	- Array of weights (transposed)
	biases 		- Array of biases
	out		- Placeholder for the output
	shapes		- Dimensions of data and weights (N, D, F, D)
*/
void innerproduct(TPrecision *data,TPrecision *weight_trans, TPrecision *biases,TPrecision *out,TsInt *shapes){
        TsInt iRow, iColumn;
		TsInt iInneritr;
        TsInt N = shapes[0];
        TsInt D = shapes[1];
        TsInt F = shapes[2];
        TPrecision dSum = 0;
        
        for(iRow=0; iRow<N; iRow++)
        {
          for(iColumn=0; iColumn<F; iColumn++)
          {
        	  dSum = 0;
        	  for(iInneritr=0; iInneritr<D;iInneritr++)
        	  {
            	dSum += data[(iRow*D)+iInneritr] * weight_trans[(iInneritr*F)+iColumn];
        	  }
            out[(iRow*F)+iColumn] = dSum + biases[iColumn];
          }
        }
}





/*
softmax :
	- Activation function
	- Squashes an array of arbitrary real values to an array of real values in the range (0, 1) that add up to 1	
Parameters :
	dData	- Array of input data
	iShapes	- Size of the input array
*/
void softmax( TPrecision *dData, TsInt iShapes )
{
    TPrecision dMax, dSum = 0;
    TsInt iRow;

    dMax = dData[0];
    for (iRow = 1; iRow < iShapes; iRow++)
    {
        if (dData[iRow] > dMax)
        {
        	dMax = dData[iRow];
        }
    }
    for (iRow = 0; iRow < iShapes; iRow++)
    {
    	dData[iRow] = dData[iRow] - dMax;
        dSum = dSum + exp(dData[iRow]);
    }
    for (iRow = 0; iRow < iShapes; iRow++)
    {
    	dData[iRow] = exp(dData[iRow])/dSum;
    }
}




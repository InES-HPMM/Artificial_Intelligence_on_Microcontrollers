/**
 * @file main.cc
 * @author Raphael Zingg / zing@zhaw.ch
 * @copyright 2019 ZHAW Institute of Embedded Systems
 * @date 17/12/2019
 *
 * @brief Example application with a mnist classifier. Used a tfLite for microcontrollers
 * quanzized neural network
 *
 * @references
 * [1] tensorflow, Get started with microcontrollers, https://www.tensorflow.org/lite/microcontrollers/get_started, Accessed 17.12.2019
 */

/* TfLite includes */
#include "model_settings.h"
#include "model_data.h"
#include "../../kernels/all_ops_resolver.h"
#include "../../micro_error_reporter.h"
#include "../../micro_interpreter.h"
#include "../../../schema/schema_generated.h"
#include "../../../version.h"
#include "tensorflow/lite/micro/kernels/micro_ops.h"

/* Board includes */
#include "stm32f4xx_hal.h"

/* Std includes */
#include "stdint.h"

#define IMAGE_SIZE 28 * 28 /* MMNIST images have 28*28 pixels */
#define currentFW 3        /* tflite */

/* GPIO */
#define OTG_FS_PowerSwitchOn_Pin GPIO_PIN_0
#define OTG_FS_PowerSwitchOn_GPIO_Port GPIOC
#define ai_timing_Pin GPIO_PIN_0
#define ai_timing_Pin_GPIO_Port GPIOB
#define BOOT1_Pin GPIO_PIN_2
#define BOOT1_GPIO_Port GPIOB
#define LD4_Pin GPIO_PIN_12
#define LD4_GPIO_Port GPIOD
#define LD3_Pin GPIO_PIN_13
#define LD3_GPIO_Port GPIOD
#define LD5_Pin GPIO_PIN_14
#define LD5_GPIO_Port GPIOD
#define LD6_Pin GPIO_PIN_15
#define LD6_GPIO_Port GPIOD
#define CATEGORIES 10

/* Private variables ---------------------------------------------------------*/
UART_HandleTypeDef huart4;

/* Private function prototypes -----------------------------------------------*/
void SystemClock_Config(void);
static void MX_GPIO_Init(void);
static void MX_UART4_Init(void);
void Error_Handler(void);

int main(void)
{
  uint8_t ui8_input_picture[IMAGE_SIZE];
  float category_score, top_category_score;
  uint8_t c_cmd, top_category_index;
  uint32_t predic_label, i;
  TfLiteStatus invoke_status;

  /* Reset of all peripherals, Initializes the Flash interface and the Systick. */
  HAL_Init();

  /* Configure the system clock */
  SystemClock_Config();

  /* Initialize all configured peripherals */
  MX_GPIO_Init();
  MX_UART4_Init();

  /* init error reporter, not used in embedded code (no printf/console) 
   * but interpreter needs reference */
  tflite::MicroErrorReporter micro_error_reporter;
  tflite::ErrorReporter *error_reporter = &micro_error_reporter;

  /* Map the model into a usable data structure. This doesn't involve any
     copying or parsing, it's a very lightweight operation. */
  const tflite::Model *model = ::tflite::GetModel(mnist_model_tflite_tflite);
  if (model->version() != TFLITE_SCHEMA_VERSION)
    return -1;

  /* This pulls in all the operation implementations we need, the source code of AllOpsResolver
   * is modified such that only the used operations are pulled! Another option would be to use MicroMutableOpResolver, 
   * However this does not work with current tf versions */
  tflite::ops::micro::AllOpsResolver resolver;

  /* Create an area of memory to use for input, output, and intermediate arrays.
  * The size required will depend on the model you are using, and may need to be determined by experimentation. [1] */
  const int tensor_arena_size = 50 * 1024;
  uint8_t tensor_arena[tensor_arena_size];

  /* Build an interpreter to run the model with */
  tflite::MicroInterpreter interpreter(model, resolver, tensor_arena,
                                       tensor_arena_size, error_reporter);

  /* Allocate memory from the tensor_arena for the model's tensors: */
  TfLiteStatus status = interpreter.AllocateTensors();

  /* Get information about the memory area to use for the model's input. */
  TfLiteTensor *input = interpreter.input(0);
  TfLiteTensor *output = interpreter.output(0);

  while (1)
  {

    /* Receive command from serial decive */
    if (HAL_UART_Receive(&huart4, &c_cmd, 1, 200) != HAL_OK)
    {
      continue;
    }

    /* Parse command */
    switch (c_cmd - 0)
    {
    case 's':

      /* Handshake */
      c_cmd = 'X' + 0;
      HAL_UART_Transmit(&huart4, &c_cmd, 1, 1000);
      c_cmd = currentFW;
      HAL_UART_Transmit(&huart4, &c_cmd, 1, 1000);
      break;

    case 'c':

      /* Receive a mnist image (784 values) feed it to the inference, and return the prediction */
      if (HAL_UART_Receive(&huart4, ui8_input_picture, IMAGE_SIZE, 1000) == HAL_OK)
      {

        /* Copy image into model input layer */
        memcpy(input->data.uint8, ui8_input_picture, IMAGE_SIZE);

        /* Start time measurement */
        HAL_GPIO_WritePin(GPIOB, ai_timing_Pin, GPIO_PIN_SET);

        /* Run the model on the current image */
        invoke_status = interpreter.Invoke();

        /* Stop time measurement */
        HAL_GPIO_WritePin(GPIOB, ai_timing_Pin, GPIO_PIN_RESET);

        if (invoke_status != kTfLiteOk)
        {
          c_cmd = (invoke_status == kTfLiteOk) ? 0 : 1;
          HAL_UART_Transmit(&huart4, &c_cmd, 1, 1000);
          break;
        }

        /* The output from the model is a vector containing the scores for each
         * kind of prediction, so figure out what the highest scoring category was. */
        top_category_score = 0;
        top_category_index = 0;
        for (int category_index = 0; category_index < CATEGORIES; category_index++)
        {
          category_score = output->data.uint8[category_index];
          if (category_score > top_category_score)
          {
            top_category_score = category_score;
            top_category_index = category_index;
          }
        }

        /* Return the prediction to evaluate the accuracy */
        c_cmd = (uint8_t)top_category_index;
        HAL_UART_Transmit(&huart4, &c_cmd, 1, 1000);
        break;
      }
    }
  }
}

/**
  * @brief System Clock Configuration
  * @retval None
  */
void SystemClock_Config(void)
{
  RCC_OscInitTypeDef RCC_OscInitStruct = {0};
  RCC_ClkInitTypeDef RCC_ClkInitStruct = {0};

  /** Configure the main internal regulator output voltage 
  */
  __HAL_RCC_PWR_CLK_ENABLE();
  __HAL_PWR_VOLTAGESCALING_CONFIG(PWR_REGULATOR_VOLTAGE_SCALE1);
  /** Initializes the CPU, AHB and APB busses clocks 
  */
  RCC_OscInitStruct.OscillatorType = RCC_OSCILLATORTYPE_HSE;
  RCC_OscInitStruct.HSEState = RCC_HSE_ON;
  RCC_OscInitStruct.PLL.PLLState = RCC_PLL_ON;
  RCC_OscInitStruct.PLL.PLLSource = RCC_PLLSOURCE_HSE;
  RCC_OscInitStruct.PLL.PLLM = 8;
  RCC_OscInitStruct.PLL.PLLN = 336;
  RCC_OscInitStruct.PLL.PLLP = RCC_PLLP_DIV2;
  RCC_OscInitStruct.PLL.PLLQ = 7;
  if (HAL_RCC_OscConfig(&RCC_OscInitStruct) != HAL_OK)
  {
    Error_Handler();
  }
  /** Initializes the CPU, AHB and APB busses clocks 
  */
  RCC_ClkInitStruct.ClockType = RCC_CLOCKTYPE_HCLK | RCC_CLOCKTYPE_SYSCLK | RCC_CLOCKTYPE_PCLK1 | RCC_CLOCKTYPE_PCLK2;
  RCC_ClkInitStruct.SYSCLKSource = RCC_SYSCLKSOURCE_PLLCLK;
  RCC_ClkInitStruct.AHBCLKDivider = RCC_SYSCLK_DIV1;
  RCC_ClkInitStruct.APB1CLKDivider = RCC_HCLK_DIV4;
  RCC_ClkInitStruct.APB2CLKDivider = RCC_HCLK_DIV2;

  if (HAL_RCC_ClockConfig(&RCC_ClkInitStruct, FLASH_LATENCY_5) != HAL_OK)
  {
    Error_Handler();
  }
}

/**
  * @brief UART4 Initialization Function
  * @param None
  * @retval None
  */
static void MX_UART4_Init(void)
{

  huart4.Instance = UART4;
  huart4.Init.BaudRate = 256000;
  huart4.Init.WordLength = UART_WORDLENGTH_8B;
  huart4.Init.StopBits = UART_STOPBITS_1;
  huart4.Init.Parity = UART_PARITY_NONE;
  huart4.Init.Mode = UART_MODE_TX_RX;
  huart4.Init.HwFlowCtl = UART_HWCONTROL_NONE;
  huart4.Init.OverSampling = UART_OVERSAMPLING_16;
  if (HAL_UART_Init(&huart4) != HAL_OK)
  {
    Error_Handler();
  }
}

/**
  * @brief GPIO Initialization Function
  * @param None
  * @retval None
  */
static void MX_GPIO_Init(void)
{
  GPIO_InitTypeDef GPIO_InitStruct = {0};

  /* GPIO Ports Clock Enable */
  __HAL_RCC_GPIOC_CLK_ENABLE();
  __HAL_RCC_GPIOA_CLK_ENABLE();
  __HAL_RCC_GPIOB_CLK_ENABLE();
  __HAL_RCC_GPIOD_CLK_ENABLE();

  /*Configure GPIO pin Output Level */
  HAL_GPIO_WritePin(OTG_FS_PowerSwitchOn_GPIO_Port, OTG_FS_PowerSwitchOn_Pin, GPIO_PIN_SET);

  /*Configure GPIO pin Output Level */
  HAL_GPIO_WritePin(ai_timing_Pin_GPIO_Port, ai_timing_Pin, GPIO_PIN_RESET);

  /*Configure GPIO pin Output Level */
  HAL_GPIO_WritePin(GPIOD, LD4_Pin | LD3_Pin | LD5_Pin | LD6_Pin, GPIO_PIN_RESET);

  /*Configure GPIO pin : OTG_FS_PowerSwitchOn_Pin */
  GPIO_InitStruct.Pin = OTG_FS_PowerSwitchOn_Pin;
  GPIO_InitStruct.Mode = GPIO_MODE_OUTPUT_PP;
  GPIO_InitStruct.Pull = GPIO_NOPULL;
  GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_LOW;
  HAL_GPIO_Init(OTG_FS_PowerSwitchOn_GPIO_Port, &GPIO_InitStruct);

  /*Configure GPIO pin : ai_timing_Pin */
  GPIO_InitStruct.Pin = ai_timing_Pin;
  GPIO_InitStruct.Mode = GPIO_MODE_OUTPUT_PP;
  GPIO_InitStruct.Pull = GPIO_NOPULL;
  GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_VERY_HIGH;
  HAL_GPIO_Init(ai_timing_Pin_GPIO_Port, &GPIO_InitStruct);

  /*Configure GPIO pin : BOOT1_Pin */
  GPIO_InitStruct.Pin = BOOT1_Pin;
  GPIO_InitStruct.Mode = GPIO_MODE_INPUT;
  GPIO_InitStruct.Pull = GPIO_NOPULL;
  HAL_GPIO_Init(BOOT1_GPIO_Port, &GPIO_InitStruct);

  /*Configure GPIO pins : LD4_Pin LD3_Pin LD5_Pin LD6_Pin */
  GPIO_InitStruct.Pin = LD4_Pin | LD3_Pin | LD5_Pin | LD6_Pin;
  GPIO_InitStruct.Mode = GPIO_MODE_OUTPUT_PP;
  GPIO_InitStruct.Pull = GPIO_NOPULL;
  GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_LOW;
  HAL_GPIO_Init(GPIOD, &GPIO_InitStruct);
}
/**
  * @brief  This function is executed in case of error occurrence.
  * @retval None
  */
void Error_Handler(void)
{
  while (1)
    ;
}

/************************ (C) COPYRIGHT STMicroelectronics *****END OF FILE****/

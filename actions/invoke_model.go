// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

package actions

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"strings"

	"github.com/aws/aws-sdk-go-v2/aws"
	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime"
)

// snippet-start:[gov2.bedrock-runtime.InvokeModelWrapper.complete]
// snippet-start:[gov2.bedrock-runtime.InvokeModelWrapper.struct]

// InvokeModelWrapper encapsulates Amazon Bedrock actions used in the examples.
// It contains a Bedrock Runtime client that is used to invoke foundation models.
type InvokeModelWrapper struct {
	BedrockRuntimeClient *bedrockruntime.Client
}

// snippet-end:[gov2.bedrock-runtime.InvokeModelWrapper.struct]

type TitanEmbeddingRequest struct {
	ModelId     string             `json:"modelId"`
	ContentType string             `json:"contentType"`
	Accept      string             `json:"accept"`
	Body        TitanEmbeddingBody `json:"body"`
}

type TitanEmbeddingBody struct {
	InputText string `json:"inputText"`
}

type TitanEmbeddingResponse struct {
}

func (wrapper InvokeModelWrapper) InvokeTitanEmbedding(prompt string) (string, error) {
	modelId := "amazon.titan-embed-text-v1"

	inputText := "I am a sentence that needs to be embedded."

	body, err := json.Marshal(TitanEmbeddingBody{InputText: inputText})

	if err != nil {
		log.Fatal("failed to marshal", err)
		return "", err
	}

	output, err := wrapper.BedrockRuntimeClient.InvokeModel(context.Background(), &bedrockruntime.InvokeModelInput{
		ModelId:     aws.String(modelId),
		ContentType: aws.String("application/json"),
		Body:        body,
	})

	if err != nil {
		ProcessError(err, modelId)
		return "", err
	}

	fmt.Println(output.Body)

	//	var response TitanTextResponse
	//	if err := json.Unmarshal(output.Body, &response); err != nil {
	//		log.Fatal("failed to unmarshal", err)
	//	}

	//	return response.Results[0].OutputText, nil
	return "", nil
}

// snippet-start:[gov2.bedrock-runtime.InvokeTitanText]

// Each model provider has their own individual request and response formats.
// For the format, ranges, and default values for Amazon Titan Text, refer to:
// https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-titan-text.html
type TitanTextRequest struct {
	InputText            string               `json:"inputText"`
	TextGenerationConfig TextGenerationConfig `json:"textGenerationConfig"`
}

type TextGenerationConfig struct {
	Temperature   float64  `json:"temperature"`
	TopP          float64  `json:"topP"`
	MaxTokenCount int      `json:"maxTokenCount"`
	StopSequences []string `json:"stopSequences,omitempty"`
}

type TitanTextResponse struct {
	InputTextTokenCount int      `json:"inputTextTokenCount"`
	Results             []Result `json:"results"`
}

type Result struct {
	TokenCount       int    `json:"tokenCount"`
	OutputText       string `json:"outputText"`
	CompletionReason string `json:"completionReason"`
}

func (wrapper InvokeModelWrapper) InvokeTitanText(prompt string) (string, error) {
	modelId := "amazon.titan-text-express-v1"

	body, err := json.Marshal(TitanTextRequest{
		InputText: prompt,
		TextGenerationConfig: TextGenerationConfig{
			Temperature:   0,
			TopP:          1,
			MaxTokenCount: 4096,
		},
	})

	if err != nil {
		log.Fatal("failed to marshal", err)
	}

	output, err := wrapper.BedrockRuntimeClient.InvokeModel(context.Background(), &bedrockruntime.InvokeModelInput{
		ModelId:     aws.String(modelId),
		ContentType: aws.String("application/json"),
		Body:        body,
	})

	if err != nil {
		ProcessError(err, modelId)
	}

	var response TitanTextResponse
	if err := json.Unmarshal(output.Body, &response); err != nil {
		log.Fatal("failed to unmarshal", err)
	}

	return response.Results[0].OutputText, nil
}

// snippet-end:[gov2.bedrock-runtime.InvokeTitanText]

func ProcessError(err error, modelId string) {
	errMsg := err.Error()
	if strings.Contains(errMsg, "no such host") {
		log.Printf(`The Bedrock service is not available in the selected region.
                    Please double-check the service availability for your region at
                    https://aws.amazon.com/about-aws/global-infrastructure/regional-product-services/.\n`)
	} else if strings.Contains(errMsg, "Could not resolve the foundation model") {
		log.Printf(`Could not resolve the foundation model from model identifier: \"%v\".
                    Please verify that the requested model exists and is accessible
                    within the specified region.\n
                    `, modelId)
	} else {
		log.Printf("Couldn't invoke model: \"%v\". Here's why: %v\n", modelId, err)
	}
}

// snippet-end:[gov2.bedrock-runtime.InvokeModelWrapper.complete]

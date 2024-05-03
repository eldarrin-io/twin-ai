package main

import (
	"context"
	"fmt"
	"github.com/aws/aws-sdk-go-v2/config"
	"github.com/aws/aws-sdk-go-v2/service/bedrock"
	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime"
	"twin-ai/actions"
)

const region = "eu-central-1"

func main() {
	sdkConfig, err := config.LoadDefaultConfig(context.TODO(), config.WithRegion(region))
	if err != nil {
		fmt.Println("Couldn't load default configuration. Have you set up your AWS account?")
		fmt.Println(err)
		return
	}
	bedrockClient := bedrock.NewFromConfig(sdkConfig)
	result, err := bedrockClient.ListFoundationModels(context.TODO(), &bedrock.ListFoundationModelsInput{})
	if err != nil {
		fmt.Printf("Couldn't list foundation models. Here's why: %v\n", err)
		return
	}
	if len(result.ModelSummaries) == 0 {
		fmt.Println("There are no foundation models.")
	}
	for _, modelSummary := range result.ModelSummaries {
		fmt.Println(*modelSummary.ModelId)
	}

	client := bedrockruntime.NewFromConfig(sdkConfig)

	a := actions.InvokeModelWrapper{BedrockRuntimeClient: client}

	b, _ := a.InvokeTitanEmbedding("I am a sentence that needs to be embedded.")

	fmt.Println(b)
}

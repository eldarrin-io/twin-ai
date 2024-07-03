package main

import (
	"bufio"
	"context"
	"fmt"
	"github.com/aws/aws-sdk-go-v2/config"
	"github.com/aws/aws-sdk-go-v2/service/bedrock"
	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime"
	"os"
	"strings"
	"twin-ai/actions"
)

const region = "eu-central-1"

func titanEmbedded() {
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

// use titan model
func titan(request string) (string, error) {
	sdkConfig, err := config.LoadDefaultConfig(context.TODO(), config.WithRegion(region))
	if err != nil {
		fmt.Println("Couldn't load default configuration. Have you set up your AWS account?")
		fmt.Println(err)
		return "", err
	}
	bedrockClient := bedrock.NewFromConfig(sdkConfig)
	result, err := bedrockClient.ListFoundationModels(context.TODO(), &bedrock.ListFoundationModelsInput{})
	if err != nil {
		fmt.Printf("Couldn't list foundation models. Here's why: %v\n", err)
		return "", err
	}
	if len(result.ModelSummaries) == 0 {
		fmt.Println("There are no foundation models.")
	}
	//for _, modelSummary := range result.ModelSummaries {
	//	fmt.Println(*modelSummary.ModelId)
	//}

	client := bedrockruntime.NewFromConfig(sdkConfig)

	a := actions.InvokeModelWrapper{BedrockRuntimeClient: client}

	b, err := a.InvokeTitanText(request)
	if err != nil {
		fmt.Println("Couldn't invoke titan model.")
		return "", err
	}
	return b, nil
}

func readLines(path string) ([]string, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	var lines []string
	scanner := bufio.NewScanner(file)
	for scanner.Scan() {
		lines = append(lines, scanner.Text())
	}

	return lines, scanner.Err()
}

func main() {
	lines, err := readLines("hack/input.txt")
	if err != nil {
		fmt.Println("Error:", err)
		return
	}

	for _, line := range lines {
		request := "What is the company description for Accertify?: " + line
		mfr, e := titan(request)
		mfr = strings.ReplaceAll(mfr, "\n", "")

		if e == nil {
			fmt.Printf("%s\n%s\n", line, mfr)
		}

		request = "What products do Accertify sell?: " + line
		mfr, e = titan(request)
		mfr = strings.ReplaceAll(mfr, "\n", "")

		if e == nil {
			fmt.Printf("%s\n%s\n", line, mfr)
		}
	}
}

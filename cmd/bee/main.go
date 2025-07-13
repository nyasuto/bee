// Package main implements the Bee CLI tool
// Learning Goal: Understanding CLI design patterns and command processing
package main

import (
	"encoding/csv"
	"flag"
	"fmt"
	"io"
	"os"
	"strconv"
	"strings"

	"github.com/nyasuto/bee/phase1"
)

// CLIConfig holds configuration for the CLI application
type CLIConfig struct {
	Command      string  // train, infer, test
	Model        string  // model type (perceptron)
	DataPath     string  // path to training data
	ModelPath    string  // path to save/load model
	LearningRate float64 // learning rate
	Epochs       int     // training epochs
	InputData    string  // comma-separated input for inference
	Verbose      bool    // verbose output
}

func main() {
	config := parseArgs()

	switch config.Command {
	case "train":
		err := trainCommand(config)
		if err != nil {
			fmt.Fprintf(os.Stderr, "❌ Training failed: %v\n", err)
			os.Exit(1)
		}
	case "infer":
		err := inferCommand(config)
		if err != nil {
			fmt.Fprintf(os.Stderr, "❌ Inference failed: %v\n", err)
			os.Exit(1)
		}
	case "test":
		err := testCommand(config)
		if err != nil {
			fmt.Fprintf(os.Stderr, "❌ Testing failed: %v\n", err)
			os.Exit(1)
		}
	case "help", "":
		printUsage()
	default:
		fmt.Fprintf(os.Stderr, "❌ Unknown command: %s\n", config.Command)
		printUsage()
		os.Exit(1)
	}
}

// parseArgs parses command line arguments using flag package
// Learning Goal: Understanding argument parsing patterns
func parseArgs() CLIConfig {
	var config CLIConfig

	if len(os.Args) < 2 {
		printUsage()
		os.Exit(1)
	}

	config.Command = os.Args[1]

	// Create flag set for each command
	var flagSet *flag.FlagSet

	switch config.Command {
	case "train":
		flagSet = flag.NewFlagSet("train", flag.ExitOnError)
		flagSet.StringVar(&config.Model, "model", "perceptron", "Model type (perceptron)")
		flagSet.StringVar(&config.DataPath, "data", "", "Path to training data (CSV)")
		flagSet.StringVar(&config.ModelPath, "output", "model.json", "Output model path")
		flagSet.Float64Var(&config.LearningRate, "lr", 0.1, "Learning rate")
		flagSet.IntVar(&config.Epochs, "epochs", 1000, "Maximum training epochs")
		flagSet.BoolVar(&config.Verbose, "verbose", false, "Verbose output")

	case "infer":
		flagSet = flag.NewFlagSet("infer", flag.ExitOnError)
		flagSet.StringVar(&config.ModelPath, "model", "model.json", "Path to trained model")
		flagSet.StringVar(&config.InputData, "input", "", "Comma-separated input values")
		flagSet.BoolVar(&config.Verbose, "verbose", false, "Verbose output")

	case "test":
		flagSet = flag.NewFlagSet("test", flag.ExitOnError)
		flagSet.StringVar(&config.Model, "model", "perceptron", "Model type (perceptron)")
		flagSet.StringVar(&config.DataPath, "data", "", "Path to test data (CSV)")
		flagSet.StringVar(&config.ModelPath, "model-path", "model.json", "Path to trained model")
		flagSet.BoolVar(&config.Verbose, "verbose", false, "Verbose output")

	default:
		return config
	}

	err := flagSet.Parse(os.Args[2:])
	if err != nil {
		fmt.Fprintf(os.Stderr, "❌ Flag parsing error: %v\n", err)
		os.Exit(1)
	}
	return config
}

// trainCommand implements the training functionality
// Learning Goal: Understanding training pipeline and data management
func trainCommand(config CLIConfig) error {
	if config.DataPath == "" {
		return fmt.Errorf("data path is required for training")
	}

	if config.Verbose {
		fmt.Printf("🐝 Bee Training - %s Model\n", config.Model)
		fmt.Printf("📊 Data: %s\n", config.DataPath)
		fmt.Printf("⚙️  Learning Rate: %.3f\n", config.LearningRate)
		fmt.Printf("🔄 Max Epochs: %d\n", config.Epochs)
	}

	// Load training data
	inputs, targets, err := loadCSVData(config.DataPath)
	if err != nil {
		return fmt.Errorf("failed to load data: %w", err)
	}

	if len(inputs) == 0 {
		return fmt.Errorf("no training data found")
	}

	if config.Verbose {
		fmt.Printf("📈 Loaded %d training samples with %d features\n",
			len(inputs), len(inputs[0]))
	}

	// Create and train model
	switch config.Model {
	case "perceptron":
		perceptron := phase1.NewPerceptron(len(inputs[0]), config.LearningRate)

		if config.Verbose {
			fmt.Printf("🧠 Training perceptron...\n")
		}

		epochs, err := perceptron.TrainDataset(inputs, targets, config.Epochs)
		if err != nil {
			return fmt.Errorf("training failed: %w", err)
		}

		// Calculate final accuracy
		accuracy, err := perceptron.Accuracy(inputs, targets)
		if err != nil {
			return fmt.Errorf("accuracy calculation failed: %w", err)
		}

		fmt.Printf("✅ Training completed in %d epochs\n", epochs)
		fmt.Printf("📊 Training accuracy: %.2f%%\n", accuracy*100)

		if config.Verbose {
			fmt.Printf("🔧 Final weights: %v\n", perceptron.GetWeights())
			fmt.Printf("🔧 Final bias: %.4f\n", perceptron.GetBias())
		}

		// Save model
		err = saveModel(perceptron, config.ModelPath)
		if err != nil {
			return fmt.Errorf("failed to save model: %w", err)
		}

		fmt.Printf("💾 Model saved to: %s\n", config.ModelPath)

	default:
		return fmt.Errorf("unsupported model type: %s", config.Model)
	}

	return nil
}

// inferCommand implements the inference functionality
// Learning Goal: Understanding model loading and prediction
func inferCommand(config CLIConfig) error {
	if config.InputData == "" {
		return fmt.Errorf("input data is required for inference")
	}

	if config.Verbose {
		fmt.Printf("🐝 Bee Inference\n")
		fmt.Printf("🧠 Model: %s\n", config.ModelPath)
		fmt.Printf("📝 Input: %s\n", config.InputData)
	}

	// Parse input data
	inputs, err := parseInputData(config.InputData)
	if err != nil {
		return fmt.Errorf("failed to parse input: %w", err)
	}

	// Load model
	perceptron, err := loadModel(config.ModelPath)
	if err != nil {
		return fmt.Errorf("failed to load model: %w", err)
	}

	if config.Verbose {
		fmt.Printf("🔧 Model loaded with %d weights\n", len(perceptron.GetWeights()))
	}

	// Perform inference
	prediction, err := perceptron.Predict(inputs)
	if err != nil {
		return fmt.Errorf("prediction failed: %w", err)
	}

	fmt.Printf("🎯 Prediction: %.0f\n", prediction)

	if config.Verbose {
		fmt.Printf("📊 Input: %v\n", inputs)
		fmt.Printf("⚖️  Weights: %v\n", perceptron.GetWeights())
		fmt.Printf("⚖️  Bias: %.4f\n", perceptron.GetBias())
	}

	return nil
}

// testCommand implements the testing functionality
// Learning Goal: Understanding model evaluation on test data
func testCommand(config CLIConfig) error {
	if config.DataPath == "" {
		return fmt.Errorf("test data path is required")
	}

	if config.Verbose {
		fmt.Printf("🐝 Bee Testing\n")
		fmt.Printf("📊 Test Data: %s\n", config.DataPath)
		fmt.Printf("🧠 Model: %s\n", config.ModelPath)
	}

	// Load test data
	inputs, targets, err := loadCSVData(config.DataPath)
	if err != nil {
		return fmt.Errorf("failed to load test data: %w", err)
	}

	if len(inputs) == 0 {
		return fmt.Errorf("no test data found")
	}

	// Load model
	perceptron, err := loadModel(config.ModelPath)
	if err != nil {
		return fmt.Errorf("failed to load model: %w", err)
	}

	// Calculate accuracy
	accuracy, err := perceptron.Accuracy(inputs, targets)
	if err != nil {
		return fmt.Errorf("accuracy calculation failed: %w", err)
	}

	fmt.Printf("📊 Test Results:\n")
	fmt.Printf("   Samples: %d\n", len(inputs))
	fmt.Printf("   Accuracy: %.2f%%\n", accuracy*100)

	// Show detailed predictions if verbose
	if config.Verbose {
		fmt.Printf("\n📝 Detailed Predictions:\n")
		correct := 0
		for i, input := range inputs {
			prediction, _ := perceptron.Predict(input)
			status := "❌"
			if prediction == targets[i] {
				status = "✅"
				correct++
			}
			fmt.Printf("   %s Input: %v → Predicted: %.0f, Actual: %.0f\n",
				status, input, prediction, targets[i])
		}
		fmt.Printf("\n📈 Summary: %d/%d correct\n", correct, len(inputs))
	}

	return nil
}

// loadCSVData loads training data from CSV file
// CSV format: feature1,feature2,...,target
// Learning Goal: Understanding data loading patterns
func loadCSVData(filepath string) ([][]float64, []float64, error) {
	file, err := os.Open(filepath)
	if err != nil {
		return nil, nil, fmt.Errorf("cannot open file: %w", err)
	}
	defer file.Close()

	reader := csv.NewReader(file)
	var inputs [][]float64
	var targets []float64

	lineNum := 0
	for {
		record, err := reader.Read()
		if err == io.EOF {
			break
		}
		if err != nil {
			return nil, nil, fmt.Errorf("error reading CSV line %d: %w", lineNum+1, err)
		}

		lineNum++

		// Skip empty lines or comments
		if len(record) == 0 || strings.HasPrefix(record[0], "#") {
			continue
		}

		if len(record) < 2 {
			return nil, nil, fmt.Errorf("line %d: insufficient columns, need at least 2", lineNum)
		}

		// Parse features (all columns except last)
		features := make([]float64, len(record)-1)
		for i := 0; i < len(record)-1; i++ {
			features[i], err = strconv.ParseFloat(strings.TrimSpace(record[i]), 64)
			if err != nil {
				return nil, nil, fmt.Errorf("line %d, column %d: invalid number '%s'",
					lineNum, i+1, record[i])
			}
		}

		// Parse target (last column)
		target, err := strconv.ParseFloat(strings.TrimSpace(record[len(record)-1]), 64)
		if err != nil {
			return nil, nil, fmt.Errorf("line %d, target column: invalid number '%s'",
				lineNum, record[len(record)-1])
		}

		inputs = append(inputs, features)
		targets = append(targets, target)
	}

	return inputs, targets, nil
}

// parseInputData parses comma-separated input values
func parseInputData(data string) ([]float64, error) {
	parts := strings.Split(data, ",")
	inputs := make([]float64, len(parts))

	for i, part := range parts {
		value, err := strconv.ParseFloat(strings.TrimSpace(part), 64)
		if err != nil {
			return nil, fmt.Errorf("invalid number at position %d: '%s'", i+1, part)
		}
		inputs[i] = value
	}

	return inputs, nil
}

// saveModel saves a perceptron model to JSON file
func saveModel(perceptron *phase1.Perceptron, filepath string) error {
	// Create directory if it doesn't exist
	dir := filepath[:strings.LastIndex(filepath, "/")]
	if dir != "" && dir != filepath {
		err := os.MkdirAll(dir, 0755)
		if err != nil {
			return fmt.Errorf("failed to create directory: %w", err)
		}
	}

	data, err := perceptron.ToJSON()
	if err != nil {
		return fmt.Errorf("failed to serialize model: %w", err)
	}

	err = os.WriteFile(filepath, data, 0644)
	if err != nil {
		return fmt.Errorf("failed to write file: %w", err)
	}

	return nil
}

// loadModel loads a perceptron model from JSON file
func loadModel(filepath string) (*phase1.Perceptron, error) {
	data, err := os.ReadFile(filepath)
	if err != nil {
		return nil, fmt.Errorf("failed to read file: %w", err)
	}

	perceptron, err := phase1.FromJSON(data)
	if err != nil {
		return nil, fmt.Errorf("failed to deserialize model: %w", err)
	}

	return perceptron, nil
}

// printUsage displays usage information
func printUsage() {
	fmt.Printf(`🐝 Bee Neural Network CLI Tool

Usage:
  bee <command> [options]

Commands:
  train      Train a neural network model
  infer      Perform inference with a trained model
  test       Test a trained model on data
  help       Show this help message

Training:
  bee train -data <csv_file> [options]
    -model string     Model type (default "perceptron")
    -data string      Path to training data (CSV format)
    -output string    Output model path (default "model.json")
    -lr float         Learning rate (default 0.1)
    -epochs int       Maximum training epochs (default 1000)
    -verbose          Verbose output

Inference:
  bee infer -model <model_file> -input <values>
    -model string     Path to trained model (default "model.json")
    -input string     Comma-separated input values
    -verbose          Verbose output

Testing:
  bee test -data <csv_file> [options]
    -data string      Path to test data (CSV format)
    -model-path string Path to trained model (default "model.json")
    -verbose          Verbose output

Examples:
  # Train XOR perceptron
  bee train -data datasets/xor.csv -output models/xor.json -verbose

  # Test XOR patterns
  bee infer -model models/xor.json -input "1,1"
  bee infer -model models/xor.json -input "0,1"

  # Test model accuracy
  bee test -data datasets/xor_test.csv -model-path models/xor.json

Data Format (CSV):
  # XOR dataset example
  0,0,0
  0,1,1
  1,0,1
  1,1,0

Learning Resources:
  - Phase 1 focuses on perceptron fundamentals
  - Each command demonstrates different ML pipeline stages
  - Verbose mode shows internal weights and calculations
`)
}

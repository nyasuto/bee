// Package main implements comprehensive tests for the Bee CLI tool
// Learning Goal: Understanding CLI testing patterns and command validation
package main

import (
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"

	"github.com/nyasuto/bee/benchmark"
	"github.com/nyasuto/bee/phase1"
)

// TestCLIConfig tests the CLIConfig structure functionality
func TestCLIConfig(t *testing.T) {
	t.Run("DefaultValues", func(t *testing.T) {
		config := CLIConfig{}

		// Test default values
		if config.LearningRate != 0 {
			t.Errorf("Expected default learning rate 0, got %f", config.LearningRate)
		}
		if config.Epochs != 0 {
			t.Errorf("Expected default epochs 0, got %d", config.Epochs)
		}
		if config.Verbose != false {
			t.Errorf("Expected default verbose false, got %v", config.Verbose)
		}
	})

	t.Run("ConfigurationFields", func(t *testing.T) {
		config := CLIConfig{
			Command:      "train",
			Model:        "perceptron",
			DataPath:     "test.csv",
			ModelPath:    "model.json",
			LearningRate: 0.1,
			Epochs:       100,
			InputData:    "1,0",
			Verbose:      true,
			Dataset:      "xor",
			Iterations:   50,
			OutputPath:   "output.json",
			MLPHidden:    "4,2",
		}

		// Verify all fields are set correctly
		if config.Command != "train" {
			t.Errorf("Expected command 'train', got %s", config.Command)
		}
		if config.Model != "perceptron" {
			t.Errorf("Expected model 'perceptron', got %s", config.Model)
		}
		if config.LearningRate != 0.1 {
			t.Errorf("Expected learning rate 0.1, got %f", config.LearningRate)
		}
		if config.Epochs != 100 {
			t.Errorf("Expected epochs 100, got %d", config.Epochs)
		}
	})
}

// TestParseInputData tests the parseInputData function
func TestParseInputData(t *testing.T) {
	testCases := []struct {
		name     string
		input    string
		expected []float64
		hasError bool
	}{
		{
			name:     "SingleValue",
			input:    "1.0",
			expected: []float64{1.0},
			hasError: false,
		},
		{
			name:     "MultipleValues",
			input:    "1.0,0.5,2.3",
			expected: []float64{1.0, 0.5, 2.3},
			hasError: false,
		},
		{
			name:     "WithSpaces",
			input:    " 1.0 , 0.5 , 2.3 ",
			expected: []float64{1.0, 0.5, 2.3},
			hasError: false,
		},
		{
			name:     "IntegerValues",
			input:    "1,0,1",
			expected: []float64{1, 0, 1},
			hasError: false,
		},
		{
			name:     "NegativeValues",
			input:    "-1.5,0,-2.3",
			expected: []float64{-1.5, 0, -2.3},
			hasError: false,
		},
		{
			name:     "InvalidNumber",
			input:    "1.0,abc,2.3",
			expected: nil,
			hasError: true,
		},
		{
			name:     "EmptyValue",
			input:    "1.0,,2.3",
			expected: nil,
			hasError: true,
		},
		{
			name:     "EmptyString",
			input:    "",
			expected: []float64{},
			hasError: true,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			result, err := parseInputData(tc.input)

			if tc.hasError {
				if err == nil {
					t.Errorf("Expected error but got none")
				}
				return
			}

			if err != nil {
				t.Errorf("Unexpected error: %v", err)
				return
			}

			if !reflect.DeepEqual(result, tc.expected) {
				t.Errorf("Expected %v, got %v", tc.expected, result)
			}
		})
	}
}

// TestLoadCSVData tests the loadCSVData function
func TestLoadCSVData(t *testing.T) {
	// Create temporary test directory
	tmpDir := t.TempDir()

	t.Run("ValidCSVFile", func(t *testing.T) {
		// Create test CSV file
		csvContent := `1,0,1
0,1,1
0,0,0
1,1,0`
		csvFile := filepath.Join(tmpDir, "test.csv")
		err := os.WriteFile(csvFile, []byte(csvContent), 0600)
		if err != nil {
			t.Fatalf("Failed to create test CSV file: %v", err)
		}

		_, _, err = loadCSVData("test.csv")
		if err == nil {
			// Should fail due to path validation (no tmpDir prefix)
			t.Errorf("Expected error due to path validation")
		}
	})

	t.Run("RelativePath", func(t *testing.T) {
		// Test with relative path (should work)
		csvContent := `1.5,2.3,1
-0.5,1.2,0`
		csvFile := "relative_test.csv"
		err := os.WriteFile(csvFile, []byte(csvContent), 0600)
		if err != nil {
			t.Fatalf("Failed to create test CSV file: %v", err)
		}
		defer os.Remove(csvFile)

		inputs, targets, err := loadCSVData(csvFile)
		if err != nil {
			t.Errorf("Unexpected error: %v", err)
			return
		}

		expectedInputs := [][]float64{{1.5, 2.3}, {-0.5, 1.2}}
		expectedTargets := []float64{1, 0}

		if !reflect.DeepEqual(inputs, expectedInputs) {
			t.Errorf("Expected inputs %v, got %v", expectedInputs, inputs)
		}
		if !reflect.DeepEqual(targets, expectedTargets) {
			t.Errorf("Expected targets %v, got %v", expectedTargets, targets)
		}
	})

	t.Run("InvalidPath", func(t *testing.T) {
		// Test directory traversal protection
		_, _, err := loadCSVData("../test.csv")
		if err == nil {
			t.Errorf("Expected error for directory traversal attempt")
		}
	})

	t.Run("AbsolutePath", func(t *testing.T) {
		// Test absolute path protection
		_, _, err := loadCSVData("/tmp/test.csv")
		if err == nil {
			t.Errorf("Expected error for absolute path")
		}
	})

	t.Run("NonExistentFile", func(t *testing.T) {
		_, _, err := loadCSVData("nonexistent.csv")
		if err == nil {
			t.Errorf("Expected error for non-existent file")
		}
	})

	t.Run("CSVWithComments", func(t *testing.T) {
		// Test CSV with comments - comments cause field count mismatch
		csvContent := `# This is a comment
1,0,1
0,1,0
1,1,1`
		csvFile := "comment_test.csv"
		err := os.WriteFile(csvFile, []byte(csvContent), 0600)
		if err != nil {
			t.Fatalf("Failed to create test CSV file: %v", err)
		}
		defer os.Remove(csvFile)

		_, _, err = loadCSVData(csvFile)

		// CSV parser will fail due to field count mismatch between comment line and data lines
		if err == nil {
			t.Errorf("Expected error due to field count mismatch in CSV with comments")
		} else {
			// Verify it's the expected type of error
			expectedError := "wrong number of fields"
			if !strings.Contains(err.Error(), expectedError) {
				t.Errorf("Expected error related to field count, got: %v", err)
			}
		}
	})

	t.Run("InsufficientColumns", func(t *testing.T) {
		csvContent := `1
0,1`
		csvFile := "insufficient_test.csv"
		err := os.WriteFile(csvFile, []byte(csvContent), 0600)
		if err != nil {
			t.Fatalf("Failed to create test CSV file: %v", err)
		}
		defer os.Remove(csvFile)

		_, _, err = loadCSVData(csvFile)
		if err == nil {
			t.Errorf("Expected error for insufficient columns")
		}
	})

	t.Run("InvalidNumbers", func(t *testing.T) {
		csvContent := `1,abc,1
0,1,0`
		csvFile := "invalid_test.csv"
		err := os.WriteFile(csvFile, []byte(csvContent), 0600)
		if err != nil {
			t.Fatalf("Failed to create test CSV file: %v", err)
		}
		defer os.Remove(csvFile)

		_, _, err = loadCSVData(csvFile)
		if err == nil {
			t.Errorf("Expected error for invalid numbers")
		}
	})
}

// TestModelPersistence tests saveModel and loadModel functions
func TestModelPersistence(t *testing.T) {
	// Create temporary test directory
	tmpDir := t.TempDir()

	t.Run("SaveAndLoadModel", func(t *testing.T) {
		// Create a test perceptron
		perceptron := phase1.NewPerceptron(2, 0.1)

		// Train it a bit to have non-default weights
		inputs := [][]float64{{1, 0}, {0, 1}}
		targets := []float64{1, 0}
		_, err := perceptron.TrainDataset(inputs, targets, 10)
		if err != nil {
			t.Fatalf("Failed to train perceptron: %v", err)
		}

		// Save model with simple filename (no directory)
		modelPath := "test_model.json"
		err = saveModel(perceptron, modelPath)
		if err != nil {
			t.Errorf("Unexpected error saving model: %v", err)
			return
		}
		defer os.Remove(modelPath)

		// Load model
		loadedPerceptron, err := loadModel(modelPath)
		if err != nil {
			t.Errorf("Unexpected error loading model: %v", err)
			return
		}

		// Compare original and loaded models
		originalWeights := perceptron.GetWeights()
		loadedWeights := loadedPerceptron.GetWeights()

		if !reflect.DeepEqual(originalWeights, loadedWeights) {
			t.Errorf("Weights don't match: original %v, loaded %v", originalWeights, loadedWeights)
		}

		if perceptron.GetBias() != loadedPerceptron.GetBias() {
			t.Errorf("Bias doesn't match: original %v, loaded %v", perceptron.GetBias(), loadedPerceptron.GetBias())
		}
	})

	t.Run("SaveModelCreateDirectory", func(t *testing.T) {
		perceptron := phase1.NewPerceptron(2, 0.1)

		// Test saving to a nested directory that doesn't exist
		modelPath := filepath.Join(tmpDir, "subdir", "model.json")
		err := saveModel(perceptron, modelPath)
		if err != nil {
			t.Errorf("Unexpected error creating directory and saving model: %v", err)
		}
		defer os.Remove(modelPath)
	})

	t.Run("LoadNonExistentModel", func(t *testing.T) {
		_, err := loadModel("nonexistent_model.json")
		if err == nil {
			t.Errorf("Expected error for non-existent model file")
		}
	})

	t.Run("LoadInvalidPath", func(t *testing.T) {
		// Test directory traversal protection
		_, err := loadModel("../model.json")
		if err == nil {
			t.Errorf("Expected error for directory traversal attempt")
		}
	})

	t.Run("LoadAbsolutePath", func(t *testing.T) {
		// Test absolute path protection
		_, err := loadModel("/tmp/model.json")
		if err == nil {
			t.Errorf("Expected error for absolute path")
		}
	})

	t.Run("LoadInvalidJSON", func(t *testing.T) {
		// Create invalid JSON file
		invalidJSON := "invalid json content"
		modelPath := "invalid_model.json"
		err := os.WriteFile(modelPath, []byte(invalidJSON), 0600)
		if err != nil {
			t.Fatalf("Failed to create invalid JSON file: %v", err)
		}
		defer os.Remove(modelPath)

		_, err = loadModel(modelPath)
		if err == nil {
			t.Errorf("Expected error for invalid JSON")
		}
	})
}

// TestHelperFunctions tests various helper functions
func TestHelperFunctions(t *testing.T) {
	t.Run("GetDatasets", func(t *testing.T) {
		testCases := []struct {
			name        string
			datasetType string
			expectedLen int
		}{
			{"XOR", "xor", 1},
			{"AND", "and", 1},
			{"OR", "or", 1},
			{"All", "all", 5}, // From GetStandardDatasets()
			{"Linear", "linear", 1},
			{"NonLinear", "nonlinear", 1},
			{"Invalid", "invalid", 0},
		}

		for _, tc := range testCases {
			t.Run(tc.name, func(t *testing.T) {
				datasets := getDatasets(tc.datasetType)
				if len(datasets) != tc.expectedLen {
					t.Errorf("Expected %d datasets for %s, got %d", tc.expectedLen, tc.datasetType, len(datasets))
				}
			})
		}
	})

	t.Run("ParseHiddenLayers", func(t *testing.T) {
		testCases := []struct {
			name     string
			input    string
			expected []int
			hasError bool
		}{
			{
				name:     "SingleLayer",
				input:    "4",
				expected: []int{4},
				hasError: false,
			},
			{
				name:     "MultipleLayers",
				input:    "4,2,1",
				expected: []int{4, 2, 1},
				hasError: false,
			},
			{
				name:     "WithSpaces",
				input:    " 4 , 2 , 1 ",
				expected: []int{4, 2, 1},
				hasError: false,
			},
			{
				name:     "EmptyString",
				input:    "",
				expected: []int{4}, // Default
				hasError: false,
			},
			{
				name:     "InvalidNumber",
				input:    "4,abc,1",
				expected: nil,
				hasError: true,
			},
			{
				name:     "ZeroSize",
				input:    "4,0,1",
				expected: nil,
				hasError: true,
			},
			{
				name:     "NegativeSize",
				input:    "4,-2,1",
				expected: nil,
				hasError: true,
			},
		}

		for _, tc := range testCases {
			t.Run(tc.name, func(t *testing.T) {
				result, err := parseHiddenLayers(tc.input)

				if tc.hasError {
					if err == nil {
						t.Errorf("Expected error but got none")
					}
					return
				}

				if err != nil {
					t.Errorf("Unexpected error: %v", err)
					return
				}

				if !reflect.DeepEqual(result, tc.expected) {
					t.Errorf("Expected %v, got %v", tc.expected, result)
				}
			})
		}
	})
}

// TestPrintBenchmarkResults tests the benchmark results printing function
func TestPrintBenchmarkResults(t *testing.T) {
	t.Run("ValidMetrics", func(t *testing.T) {
		// Capture stdout for testing
		metrics := benchmark.PerformanceMetrics{
			ModelType:       "perceptron",
			DatasetName:     "xor",
			Accuracy:        0.75,
			TrainingTime:    1000000, // 1ms in nanoseconds
			InferenceTime:   500000,  // 0.5ms in nanoseconds
			MemoryUsage:     1024,    // 1KB
			ConvergenceRate: 100,
			FinalLoss:       0.1234,
		}

		// This test just ensures the function doesn't panic
		// In a more complete test, you would capture stdout and verify the output format
		printBenchmarkResults(metrics)
	})
}

// TestMainFunction tests the main function indirectly through command simulation
func TestMainFunction(t *testing.T) {
	// Save original os.Args
	originalArgs := os.Args
	defer func() { os.Args = originalArgs }()

	t.Run("HelpCommand", func(t *testing.T) {
		os.Args = []string{"bee", "help"}

		// Capture exit behavior by testing parseArgs instead of main directly
		config := parseArgs()
		if config.Command != "help" {
			t.Errorf("Expected command 'help', got %s", config.Command)
		}
	})

	t.Run("EmptyCommand", func(t *testing.T) {
		os.Args = []string{"bee", ""}

		config := parseArgs()
		if config.Command != "" {
			t.Errorf("Expected empty command, got %s", config.Command)
		}
	})

	t.Run("UnknownCommand", func(t *testing.T) {
		os.Args = []string{"bee", "unknown"}

		config := parseArgs()
		if config.Command != "unknown" {
			t.Errorf("Expected command 'unknown', got %s", config.Command)
		}
	})
}

// TestMainErrorHandlingPaths tests error handling in main function paths
func TestMainErrorHandlingPaths(t *testing.T) {
	// Save original os.Args
	originalArgs := os.Args
	defer func() { os.Args = originalArgs }()

	t.Run("TrainCommandMissingData", func(t *testing.T) {
		os.Args = []string{"bee", "train", "-model", "perceptron"}
		config := parseArgs()

		err := trainCommand(config)
		if err == nil {
			t.Errorf("Expected error for missing data path")
		}
	})

	t.Run("InferCommandMissingInput", func(t *testing.T) {
		os.Args = []string{"bee", "infer", "-model", "nonexistent.json"}
		config := parseArgs()

		err := inferCommand(config)
		if err == nil {
			t.Errorf("Expected error for missing input data")
		}
	})

	t.Run("TestCommandMissingData", func(t *testing.T) {
		os.Args = []string{"bee", "test", "-model-path", "nonexistent.json"}
		config := parseArgs()

		err := testCommand(config)
		if err == nil {
			t.Errorf("Expected error for missing test data")
		}
	})

	t.Run("BenchmarkCommandInvalidModel", func(t *testing.T) {
		os.Args = []string{"bee", "benchmark", "-model", "invalid"}
		config := parseArgs()

		err := benchmarkCommand(config)
		if err == nil {
			t.Errorf("Expected error for invalid model type")
		}
	})

	t.Run("CompareCommandInvalidDataset", func(t *testing.T) {
		os.Args = []string{"bee", "compare", "-dataset", "invalid"}
		config := parseArgs()

		err := compareCommand(config)
		if err == nil {
			t.Errorf("Expected error for invalid dataset")
		}
	})
}

// TestVerboseOutputPaths tests verbose output functionality
func TestVerboseOutputPaths(t *testing.T) {
	// Save original os.Args
	originalArgs := os.Args
	defer func() { os.Args = originalArgs }()

	t.Run("TrainVerboseMode", func(t *testing.T) {
		os.Args = []string{"bee", "train", "-data", "test.csv", "-verbose"}
		config := parseArgs()

		if !config.Verbose {
			t.Errorf("Expected verbose mode to be enabled")
		}
	})

	t.Run("InferVerboseMode", func(t *testing.T) {
		os.Args = []string{"bee", "infer", "-model", "model.json", "-input", "1,0", "-verbose"}
		config := parseArgs()

		if !config.Verbose {
			t.Errorf("Expected verbose mode to be enabled")
		}
	})

	t.Run("BenchmarkVerboseMode", func(t *testing.T) {
		os.Args = []string{"bee", "benchmark", "-verbose"}
		config := parseArgs()

		if !config.Verbose {
			t.Errorf("Expected verbose mode to be enabled")
		}
	})
}

// TestEdgeCases tests various edge cases and boundary conditions
func TestEdgeCases(t *testing.T) {

	t.Run("LoadCSVEmptyFile", func(t *testing.T) {
		emptyFile := "empty_edge.csv"
		err := os.WriteFile(emptyFile, []byte(""), 0600)
		if err != nil {
			t.Fatalf("Failed to create empty file: %v", err)
		}
		defer os.Remove(emptyFile)

		inputs, targets, err := loadCSVData(emptyFile)
		if err != nil {
			t.Logf("Empty file handling: %v", err)
		}
		if len(inputs) != 0 || len(targets) != 0 {
			t.Errorf("Expected empty results for empty file")
		}
	})

	t.Run("LoadCSVWithComments", func(t *testing.T) {
		commentFile := "comments_edge.csv"
		content := "# This is a comment\n1,0,1\n# Another comment\n0,1,1"
		err := os.WriteFile(commentFile, []byte(content), 0600)
		if err != nil {
			t.Fatalf("Failed to create comment file: %v", err)
		}
		defer os.Remove(commentFile)

		inputs, targets, err := loadCSVData(commentFile)
		if err != nil {
			// Comments might cause parsing errors, which is acceptable
			t.Logf("CSV with comments handling: %v", err)
		} else if len(inputs) != 2 || len(targets) != 2 {
			t.Errorf("Expected 2 data rows, got %d inputs and %d targets", len(inputs), len(targets))
		}
	})

	t.Run("ParseInputDataEdgeCases", func(t *testing.T) {
		testCases := []struct {
			name     string
			input    string
			hasError bool
		}{
			{"SingleValue", "1.0", false},
			{"LargeValues", "1000000.5,999999.2", false},
			{"ScientificNotation", "1.5e-3,2.1e+2", false},
			{"ZeroValues", "0,0,0", false},
		}

		for _, tc := range testCases {
			t.Run(tc.name, func(t *testing.T) {
				_, err := parseInputData(tc.input)
				if tc.hasError && err == nil {
					t.Errorf("Expected error for input: %s", tc.input)
				}
				if !tc.hasError && err != nil {
					t.Errorf("Unexpected error for input %s: %v", tc.input, err)
				}
			})
		}
	})
}

// TestCommandIntegration tests command functions with realistic scenarios
func TestCommandIntegration(t *testing.T) {
	t.Run("FullTrainInferTestWorkflow", func(t *testing.T) {
		// Create test data file
		csvContent := "0,0,0\n0,1,1\n1,0,1\n1,1,0"
		csvFile := "integration_test.csv"
		err := os.WriteFile(csvFile, []byte(csvContent), 0600)
		if err != nil {
			t.Fatalf("Failed to create test CSV: %v", err)
		}
		defer os.Remove(csvFile)

		modelFile := "integration_model.json"

		// Test training
		trainConfig := CLIConfig{
			Command:      "train",
			Model:        "perceptron",
			DataPath:     csvFile,
			ModelPath:    modelFile,
			LearningRate: 0.1,
			Epochs:       10,
			Verbose:      false,
		}

		err = trainCommand(trainConfig)
		if err != nil {
			t.Errorf("Training command failed: %v", err)
			return
		}
		defer os.Remove(modelFile)

		// Test inference
		inferConfig := CLIConfig{
			Command:   "infer",
			ModelPath: modelFile,
			InputData: "1,0",
			Verbose:   false,
		}

		err = inferCommand(inferConfig)
		if err != nil {
			t.Errorf("Inference command failed: %v", err)
		}

		// Test testing command
		testConfig := CLIConfig{
			Command:   "test",
			Model:     "perceptron",
			DataPath:  csvFile,
			ModelPath: modelFile,
			Verbose:   false,
		}

		err = testCommand(testConfig)
		if err != nil {
			t.Errorf("Test command failed: %v", err)
		}
	})

	t.Run("BenchmarkIntegration", func(t *testing.T) {
		config := CLIConfig{
			Command:    "benchmark",
			Model:      "perceptron",
			Dataset:    "xor",
			Iterations: 2,
			Verbose:    false,
		}

		err := benchmarkCommand(config)
		if err != nil {
			t.Errorf("Benchmark command failed: %v", err)
		}
	})

	t.Run("CompareIntegration", func(t *testing.T) {
		config := CLIConfig{
			Command:    "compare",
			Dataset:    "xor",
			Iterations: 2,
			MLPHidden:  "4",
			Verbose:    false,
		}

		err := compareCommand(config)
		if err != nil {
			t.Errorf("Compare command failed: %v", err)
		}
	})
}

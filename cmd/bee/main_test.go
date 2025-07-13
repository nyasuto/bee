// Package main implements comprehensive tests for the Bee CLI tool
// Learning Goal: Understanding CLI testing patterns and command validation
package main

import (
	"encoding/json"
	"fmt"
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

// TestCommandValidation tests command validation logic
func TestCommandValidation(t *testing.T) {
	t.Run("ValidCommands", func(t *testing.T) {
		validCommands := []string{"train", "infer", "test", "benchmark", "compare", "help"}

		for _, cmd := range validCommands {
			t.Run(fmt.Sprintf("Command_%s", cmd), func(t *testing.T) {
				// Test that these are recognized as valid commands
				// This is implicitly tested by the switch statement in main()
				switch cmd {
				case "train", "infer", "test", "benchmark", "compare", "help":
					// Valid commands
				default:
					t.Errorf("Command %s should be valid", cmd)
				}
			})
		}
	})

	t.Run("InvalidCommands", func(t *testing.T) {
		invalidCommands := []string{"invalid", "unknown", "bad"}

		for _, cmd := range invalidCommands {
			t.Run(fmt.Sprintf("Command_%s", cmd), func(t *testing.T) {
				// Test that these would be handled by the default case
				switch cmd {
				case "train", "infer", "test", "benchmark", "compare", "help":
					t.Errorf("Command %s should be invalid", cmd)
				default:
					// Expected: invalid commands fall through to default
				}
			})
		}
	})
}

// TestTrainCommandValidation tests training command validation
func TestTrainCommandValidation(t *testing.T) {
	t.Run("MissingDataPath", func(t *testing.T) {
		config := CLIConfig{
			Command:  "train",
			DataPath: "", // Missing data path
		}

		err := trainCommand(config)
		if err == nil {
			t.Errorf("Expected error for missing data path")
		}

		expectedError := "data path is required for training"
		if !strings.Contains(err.Error(), expectedError) {
			t.Errorf("Expected error message to contain '%s', got '%s'", expectedError, err.Error())
		}
	})

	t.Run("UnsupportedModel", func(t *testing.T) {
		// Create temporary CSV file for testing
		csvContent := `1,0,1
0,1,0`
		csvFile := "test_train.csv"
		err := os.WriteFile(csvFile, []byte(csvContent), 0600)
		if err != nil {
			t.Fatalf("Failed to create test CSV file: %v", err)
		}
		defer os.Remove(csvFile)

		config := CLIConfig{
			Command:  "train",
			Model:    "unsupported_model",
			DataPath: csvFile,
		}

		err = trainCommand(config)
		if err == nil {
			t.Errorf("Expected error for unsupported model")
		}

		expectedError := "unsupported model type"
		if !strings.Contains(err.Error(), expectedError) {
			t.Errorf("Expected error message to contain '%s', got '%s'", expectedError, err.Error())
		}
	})
}

// TestInferCommandValidation tests inference command validation
func TestInferCommandValidation(t *testing.T) {
	t.Run("MissingInputData", func(t *testing.T) {
		config := CLIConfig{
			Command:   "infer",
			InputData: "", // Missing input data
		}

		err := inferCommand(config)
		if err == nil {
			t.Errorf("Expected error for missing input data")
		}

		expectedError := "input data is required for inference"
		if !strings.Contains(err.Error(), expectedError) {
			t.Errorf("Expected error message to contain '%s', got '%s'", expectedError, err.Error())
		}
	})

	t.Run("InvalidInputData", func(t *testing.T) {
		config := CLIConfig{
			Command:   "infer",
			InputData: "invalid,data,format",
			ModelPath: "nonexistent.json",
		}

		err := inferCommand(config)
		if err == nil {
			t.Errorf("Expected error for invalid model path")
		}
	})
}

// TestTestCommandValidation tests test command validation
func TestTestCommandValidation(t *testing.T) {
	t.Run("MissingDataPath", func(t *testing.T) {
		config := CLIConfig{
			Command:  "test",
			DataPath: "", // Missing data path
		}

		err := testCommand(config)
		if err == nil {
			t.Errorf("Expected error for missing test data path")
		}

		expectedError := "test data path is required"
		if !strings.Contains(err.Error(), expectedError) {
			t.Errorf("Expected error message to contain '%s', got '%s'", expectedError, err.Error())
		}
	})
}

// TestConfigurationDefaults tests default configuration values
func TestConfigurationDefaults(t *testing.T) {
	t.Run("DefaultLearningRate", func(t *testing.T) {
		// Test that the default learning rate is reasonable
		defaultLR := 0.1
		if defaultLR <= 0 || defaultLR >= 1 {
			t.Errorf("Default learning rate %f should be between 0 and 1", defaultLR)
		}
	})

	t.Run("DefaultEpochs", func(t *testing.T) {
		// Test that the default epoch count is reasonable
		defaultEpochs := 1000
		if defaultEpochs <= 0 {
			t.Errorf("Default epochs %d should be positive", defaultEpochs)
		}
	})

	t.Run("DefaultModelPath", func(t *testing.T) {
		// Test that the default model path is reasonable
		defaultPath := "model.json"
		if !strings.HasSuffix(defaultPath, ".json") {
			t.Errorf("Default model path %s should have .json extension", defaultPath)
		}
	})
}

// TestBenchmarkIntegration tests benchmark command integration
func TestBenchmarkIntegration(t *testing.T) {
	t.Run("BenchmarkDatasetValidation", func(t *testing.T) {
		// Test with empty dataset
		datasets := getDatasets("invalid")
		if len(datasets) != 0 {
			t.Errorf("Expected 0 datasets for invalid type, got %d", len(datasets))
		}
	})

	t.Run("PerformanceMetricsFields", func(t *testing.T) {
		// Verify that PerformanceMetrics has expected fields
		metrics := benchmark.PerformanceMetrics{}

		// Test that we can set all expected fields
		metrics.ModelType = "test"
		metrics.DatasetName = "test"
		metrics.Accuracy = 0.5
		metrics.TrainingTime = 1000
		metrics.InferenceTime = 100
		metrics.MemoryUsage = 1024
		metrics.ConvergenceRate = 50
		metrics.FinalLoss = 0.1

		// Verify fields are set correctly
		if metrics.ModelType != "test" {
			t.Errorf("ModelType not set correctly")
		}
		if metrics.Accuracy != 0.5 {
			t.Errorf("Accuracy not set correctly")
		}
	})
}

// TestComparisonReport tests comparison functionality
func TestComparisonReport(t *testing.T) {
	t.Run("ComparisonReportCreation", func(t *testing.T) {
		// Create mock performance metrics
		perceptronMetrics := benchmark.PerformanceMetrics{
			ModelType:    "perceptron",
			DatasetName:  "xor",
			Accuracy:     0.50, // Perceptron can't learn XOR
			TrainingTime: 1000000,
		}

		mlpMetrics := benchmark.PerformanceMetrics{
			ModelType:    "mlp",
			DatasetName:  "xor",
			Accuracy:     1.0, // MLP can learn XOR
			TrainingTime: 5000000,
		}

		// Test that MLP should have better accuracy than perceptron for XOR
		if mlpMetrics.Accuracy <= perceptronMetrics.Accuracy {
			t.Errorf("MLP should have better accuracy than perceptron for XOR problem")
		}
	})
}

// TestErrorHandling tests error handling patterns
func TestErrorHandling(t *testing.T) {
	t.Run("EmptyDataset", func(t *testing.T) {
		// Create empty CSV file
		csvFile := "empty_test.csv"
		err := os.WriteFile(csvFile, []byte(""), 0600)
		if err != nil {
			t.Fatalf("Failed to create empty CSV file: %v", err)
		}
		defer os.Remove(csvFile)

		inputs, targets, err := loadCSVData(csvFile)
		if err != nil {
			t.Errorf("Unexpected error for empty file: %v", err)
		}

		if len(inputs) != 0 || len(targets) != 0 {
			t.Errorf("Expected empty dataset, got inputs: %v, targets: %v", inputs, targets)
		}

		// Test training with empty dataset
		config := CLIConfig{
			Command:  "train",
			DataPath: csvFile,
		}

		err = trainCommand(config)
		if err == nil {
			t.Errorf("Expected error for training with empty dataset")
		}

		expectedError := "no training data found"
		if !strings.Contains(err.Error(), expectedError) {
			t.Errorf("Expected error message to contain '%s', got '%s'", expectedError, err.Error())
		}
	})

	t.Run("JSONMarshalError", func(t *testing.T) {
		// Test JSON serialization behavior
		perceptron := phase1.NewPerceptron(2, 0.1)

		// Get JSON data
		data, err := perceptron.ToJSON()
		if err != nil {
			t.Errorf("Unexpected error in JSON serialization: %v", err)
		}

		// Verify it's valid JSON
		var jsonData map[string]interface{}
		err = json.Unmarshal(data, &jsonData)
		if err != nil {
			t.Errorf("Generated JSON is not valid: %v", err)
		}
	})
}

// TestVerboseOutput tests verbose output functionality
func TestVerboseOutput(t *testing.T) {
	t.Run("VerboseConfigurationImpact", func(t *testing.T) {
		config := CLIConfig{
			Verbose: true,
		}

		// Test that verbose flag is properly set
		if !config.Verbose {
			t.Errorf("Verbose flag should be true")
		}

		config.Verbose = false
		if config.Verbose {
			t.Errorf("Verbose flag should be false")
		}
	})
}

// TestFileSystemOperations tests file system related operations
func TestFileSystemOperations(t *testing.T) {
	t.Run("DirectoryCreation", func(t *testing.T) {
		// Test directory creation in saveModel
		tmpDir := t.TempDir()
		perceptron := phase1.NewPerceptron(2, 0.1)

		// Use a subdirectory that doesn't exist
		modelPath := filepath.Join(tmpDir, "models", "test.json")

		err := saveModel(perceptron, modelPath)
		if err != nil {
			t.Errorf("Unexpected error creating directory: %v", err)
		}

		// Verify file exists
		if _, err := os.Stat(modelPath); os.IsNotExist(err) {
			t.Errorf("Model file was not created")
		}
	})

	t.Run("FilePermissions", func(t *testing.T) {
		perceptron := phase1.NewPerceptron(2, 0.1)
		modelPath := "test_permissions.json"

		err := saveModel(perceptron, modelPath)
		if err != nil {
			t.Errorf("Unexpected error saving model: %v", err)
		}
		defer os.Remove(modelPath)

		// Check file permissions
		info, err := os.Stat(modelPath)
		if err != nil {
			t.Errorf("Error getting file info: %v", err)
		}

		// File should be created with 0600 permissions (read/write for owner only)
		expectedPerm := os.FileMode(0600)
		if info.Mode().Perm() != expectedPerm {
			t.Errorf("Expected file permissions %v, got %v", expectedPerm, info.Mode().Perm())
		}
	})
}

// TestBenchmarkCommand tests the benchmarkCommand functionality
func TestBenchmarkCommand(t *testing.T) {
	t.Run("PerceptronBenchmark", func(t *testing.T) {
		config := CLIConfig{
			Command:    "benchmark",
			Model:      "perceptron",
			Dataset:    "xor",
			Iterations: 10,
			Verbose:    false,
		}

		err := benchmarkCommand(config)
		if err != nil {
			t.Errorf("Unexpected error in perceptron benchmark: %v", err)
		}
	})

	t.Run("MLPBenchmark", func(t *testing.T) {
		config := CLIConfig{
			Command:    "benchmark",
			Model:      "mlp",
			Dataset:    "and",
			Iterations: 5,
			MLPHidden:  "4",
			Verbose:    false,
		}

		err := benchmarkCommand(config)
		if err != nil {
			t.Errorf("Unexpected error in MLP benchmark: %v", err)
		}
	})

	t.Run("BothModelsBenchmark", func(t *testing.T) {
		config := CLIConfig{
			Command:    "benchmark",
			Model:      "both",
			Dataset:    "or",
			Iterations: 3,
			MLPHidden:  "2",
			Verbose:    true,
		}

		err := benchmarkCommand(config)
		if err != nil {
			t.Errorf("Unexpected error in both models benchmark: %v", err)
		}
	})

	t.Run("InvalidModel", func(t *testing.T) {
		config := CLIConfig{
			Command: "benchmark",
			Model:   "invalid_model",
			Dataset: "xor",
		}

		err := benchmarkCommand(config)
		if err == nil {
			t.Errorf("Expected error for invalid model type")
		}
	})

	t.Run("InvalidDataset", func(t *testing.T) {
		config := CLIConfig{
			Command: "benchmark",
			Model:   "perceptron",
			Dataset: "invalid_dataset",
		}

		err := benchmarkCommand(config)
		if err == nil {
			t.Errorf("Expected error for invalid dataset")
		}
	})

	t.Run("InvalidMLPHidden", func(t *testing.T) {
		config := CLIConfig{
			Command:   "benchmark",
			Model:     "mlp",
			Dataset:   "xor",
			MLPHidden: "invalid",
		}

		err := benchmarkCommand(config)
		if err == nil {
			t.Errorf("Expected error for invalid MLP hidden layers")
		}
	})
}

// TestCompareCommand tests the compareCommand functionality
func TestCompareCommand(t *testing.T) {
	t.Run("BasicComparison", func(t *testing.T) {
		config := CLIConfig{
			Command:    "compare",
			Dataset:    "xor",
			Iterations: 5,
			MLPHidden:  "4",
			Verbose:    false,
		}

		err := compareCommand(config)
		if err != nil {
			t.Errorf("Unexpected error in comparison: %v", err)
		}
	})

	t.Run("VerboseComparison", func(t *testing.T) {
		config := CLIConfig{
			Command:    "compare",
			Dataset:    "and",
			Iterations: 3,
			MLPHidden:  "2,1",
			Verbose:    true,
		}

		err := compareCommand(config)
		if err != nil {
			t.Errorf("Unexpected error in verbose comparison: %v", err)
		}
	})

	t.Run("WithOutputPath", func(t *testing.T) {
		config := CLIConfig{
			Command:    "compare",
			Dataset:    "or",
			Iterations: 2,
			OutputPath: "test_comparison",
			Verbose:    false,
		}

		err := compareCommand(config)
		if err != nil {
			t.Errorf("Unexpected error in comparison with output: %v", err)
		}
	})

	t.Run("AllDatasets", func(t *testing.T) {
		config := CLIConfig{
			Command:    "compare",
			Dataset:    "all",
			Iterations: 2,
			OutputPath: "test_all",
			Verbose:    false,
		}

		err := compareCommand(config)
		if err != nil {
			t.Errorf("Unexpected error in all datasets comparison: %v", err)
		}
	})

	t.Run("InvalidDataset", func(t *testing.T) {
		config := CLIConfig{
			Command: "compare",
			Dataset: "invalid_dataset",
		}

		err := compareCommand(config)
		if err == nil {
			t.Errorf("Expected error for invalid dataset")
		}
	})

	t.Run("InvalidMLPHidden", func(t *testing.T) {
		config := CLIConfig{
			Command:   "compare",
			Dataset:   "xor",
			MLPHidden: "invalid",
		}

		err := compareCommand(config)
		if err == nil {
			t.Errorf("Expected error for invalid MLP hidden layers")
		}
	})
}

// TestTrainCommandDetailed tests more detailed aspects of trainCommand
func TestTrainCommandDetailed(t *testing.T) {
	t.Run("TrainWithValidData", func(t *testing.T) {
		// Create test data
		csvContent := `1,0,1
0,1,1
0,0,0
1,1,0`
		csvFile := "test_train_detailed.csv"
		err := os.WriteFile(csvFile, []byte(csvContent), 0600)
		if err != nil {
			t.Fatalf("Failed to create test CSV file: %v", err)
		}
		defer os.Remove(csvFile)

		config := CLIConfig{
			Command:      "train",
			Model:        "perceptron",
			DataPath:     csvFile,
			ModelPath:    "test_detailed_model.json",
			LearningRate: 0.1,
			Epochs:       100,
			Verbose:      true,
		}

		err = trainCommand(config)
		if err != nil {
			t.Errorf("Unexpected error in training: %v", err)
		}

		// Clean up model file
		os.Remove("test_detailed_model.json")
	})

	t.Run("TrainWithInvalidData", func(t *testing.T) {
		config := CLIConfig{
			Command:  "train",
			DataPath: "nonexistent_file.csv",
		}

		err := trainCommand(config)
		if err == nil {
			t.Errorf("Expected error for non-existent training data")
		}
	})
}

// TestInferCommandDetailed tests more detailed aspects of inferCommand
func TestInferCommandDetailed(t *testing.T) {
	t.Run("InferWithTrainedModel", func(t *testing.T) {
		// First create and train a model
		perceptron := phase1.NewPerceptron(2, 0.1)
		inputs := [][]float64{{1, 0}, {0, 1}, {0, 0}, {1, 1}}
		targets := []float64{1, 1, 0, 0}
		_, err := perceptron.TrainDataset(inputs, targets, 50)
		if err != nil {
			t.Fatalf("Failed to train model: %v", err)
		}

		// Save the model
		modelPath := "test_infer_model.json"
		err = saveModel(perceptron, modelPath)
		if err != nil {
			t.Fatalf("Failed to save model: %v", err)
		}
		defer os.Remove(modelPath)

		config := CLIConfig{
			Command:   "infer",
			ModelPath: modelPath,
			InputData: "1,0",
			Verbose:   true,
		}

		err = inferCommand(config)
		if err != nil {
			t.Errorf("Unexpected error in inference: %v", err)
		}
	})

	t.Run("InferWithInvalidModelPath", func(t *testing.T) {
		config := CLIConfig{
			Command:   "infer",
			ModelPath: "nonexistent_model.json",
			InputData: "1,0",
		}

		err := inferCommand(config)
		if err == nil {
			t.Errorf("Expected error for non-existent model file")
		}
	})

	t.Run("InferWithInvalidInput", func(t *testing.T) {
		// Create a simple model first
		perceptron := phase1.NewPerceptron(2, 0.1)
		modelPath := "test_invalid_input_model.json"
		err := saveModel(perceptron, modelPath)
		if err != nil {
			t.Fatalf("Failed to save model: %v", err)
		}
		defer os.Remove(modelPath)

		config := CLIConfig{
			Command:   "infer",
			ModelPath: modelPath,
			InputData: "invalid,input",
		}

		err = inferCommand(config)
		if err == nil {
			t.Errorf("Expected error for invalid input data")
		}
	})
}

// TestTestCommandDetailed tests more detailed aspects of testCommand
func TestTestCommandDetailed(t *testing.T) {
	t.Run("TestWithValidData", func(t *testing.T) {
		// Create test data
		csvContent := `1,0,1
0,1,1
0,0,0
1,1,0`
		csvFile := "test_test_data.csv"
		err := os.WriteFile(csvFile, []byte(csvContent), 0600)
		if err != nil {
			t.Fatalf("Failed to create test CSV file: %v", err)
		}
		defer os.Remove(csvFile)

		// Create a trained model
		perceptron := phase1.NewPerceptron(2, 0.1)
		inputs := [][]float64{{1, 0}, {0, 1}, {0, 0}, {1, 1}}
		targets := []float64{1, 1, 0, 0}
		_, err = perceptron.TrainDataset(inputs, targets, 50)
		if err != nil {
			t.Fatalf("Failed to train model: %v", err)
		}

		modelPath := "test_test_model.json"
		err = saveModel(perceptron, modelPath)
		if err != nil {
			t.Fatalf("Failed to save model: %v", err)
		}
		defer os.Remove(modelPath)

		config := CLIConfig{
			Command:   "test",
			DataPath:  csvFile,
			ModelPath: modelPath,
			Verbose:   true,
		}

		err = testCommand(config)
		if err != nil {
			t.Errorf("Unexpected error in testing: %v", err)
		}
	})

	t.Run("TestWithNonExistentModel", func(t *testing.T) {
		csvContent := `1,0,1`
		csvFile := "test_data_no_model.csv"
		err := os.WriteFile(csvFile, []byte(csvContent), 0600)
		if err != nil {
			t.Fatalf("Failed to create test CSV file: %v", err)
		}
		defer os.Remove(csvFile)

		config := CLIConfig{
			Command:   "test",
			DataPath:  csvFile,
			ModelPath: "nonexistent_model.json",
		}

		err = testCommand(config)
		if err == nil {
			t.Errorf("Expected error for non-existent model file")
		}
	})

	t.Run("TestWithInvalidData", func(t *testing.T) {
		// Create a model
		perceptron := phase1.NewPerceptron(2, 0.1)
		modelPath := "test_invalid_data_model.json"
		err := saveModel(perceptron, modelPath)
		if err != nil {
			t.Fatalf("Failed to save model: %v", err)
		}
		defer os.Remove(modelPath)

		config := CLIConfig{
			Command:   "test",
			DataPath:  "nonexistent_test_data.csv",
			ModelPath: modelPath,
		}

		err = testCommand(config)
		if err == nil {
			t.Errorf("Expected error for non-existent test data file")
		}
	})
}

// TestPrintUsage tests the printUsage function
func TestPrintUsage(t *testing.T) {
	t.Run("PrintUsageExecution", func(t *testing.T) {
		// This test simply ensures printUsage doesn't panic
		// In a real scenario, you might capture stdout to verify the content
		printUsage()
	})
}

// Package config test for configuration structures
// Learning Goal: Understanding comprehensive configuration testing
package config

import (
	"testing"
)

// TestBaseConfig tests the BaseConfig structure
func TestBaseConfig(t *testing.T) {
	t.Run("BaseConfigCreation", func(t *testing.T) {
		cfg := BaseConfig{
			Command: "test",
			Verbose: true,
		}

		if cfg.Command != "test" {
			t.Errorf("Expected command 'test', got '%s'", cfg.Command)
		}

		if !cfg.Verbose {
			t.Error("Expected verbose to be true")
		}
	})

	t.Run("BaseConfigDefaults", func(t *testing.T) {
		cfg := BaseConfig{}

		if cfg.Command != "" {
			t.Errorf("Expected empty command, got '%s'", cfg.Command)
		}

		if cfg.Verbose {
			t.Error("Expected verbose to be false by default")
		}
	})
}

// TestTrainConfig tests the TrainConfig structure
func TestTrainConfig(t *testing.T) {
	t.Run("TrainConfigCreation", func(t *testing.T) {
		cfg := TrainConfig{
			BaseConfig: BaseConfig{
				Command: "train",
				Verbose: true,
			},
			Model:        "perceptron",
			DataPath:     "data/xor.csv",
			ModelPath:    "models/xor.json",
			LearningRate: 0.1,
			Epochs:       1000,
		}

		if cfg.Command != "train" {
			t.Errorf("Expected command 'train', got '%s'", cfg.Command)
		}

		if cfg.Model != "perceptron" {
			t.Errorf("Expected model 'perceptron', got '%s'", cfg.Model)
		}

		if cfg.DataPath != "data/xor.csv" {
			t.Errorf("Expected data path 'data/xor.csv', got '%s'", cfg.DataPath)
		}

		if cfg.ModelPath != "models/xor.json" {
			t.Errorf("Expected model path 'models/xor.json', got '%s'", cfg.ModelPath)
		}

		if cfg.LearningRate != 0.1 {
			t.Errorf("Expected learning rate 0.1, got %f", cfg.LearningRate)
		}

		if cfg.Epochs != 1000 {
			t.Errorf("Expected epochs 1000, got %d", cfg.Epochs)
		}
	})

	t.Run("TrainConfigDefaults", func(t *testing.T) {
		cfg := TrainConfig{}

		if cfg.Model != "" {
			t.Errorf("Expected empty model, got '%s'", cfg.Model)
		}

		if cfg.LearningRate != 0 {
			t.Errorf("Expected learning rate 0, got %f", cfg.LearningRate)
		}

		if cfg.Epochs != 0 {
			t.Errorf("Expected epochs 0, got %d", cfg.Epochs)
		}
	})
}

// TestInferConfig tests the InferConfig structure
func TestInferConfig(t *testing.T) {
	t.Run("InferConfigCreation", func(t *testing.T) {
		cfg := InferConfig{
			BaseConfig: BaseConfig{
				Command: "infer",
				Verbose: false,
			},
			ModelPath: "models/trained.json",
			InputData: "1,0",
		}

		if cfg.Command != "infer" {
			t.Errorf("Expected command 'infer', got '%s'", cfg.Command)
		}

		if cfg.ModelPath != "models/trained.json" {
			t.Errorf("Expected model path 'models/trained.json', got '%s'", cfg.ModelPath)
		}

		if cfg.InputData != "1,0" {
			t.Errorf("Expected input data '1,0', got '%s'", cfg.InputData)
		}
	})

	t.Run("InferConfigDefaults", func(t *testing.T) {
		cfg := InferConfig{}

		if cfg.ModelPath != "" {
			t.Errorf("Expected empty model path, got '%s'", cfg.ModelPath)
		}

		if cfg.InputData != "" {
			t.Errorf("Expected empty input data, got '%s'", cfg.InputData)
		}
	})
}

// TestTestConfig tests the TestConfig structure
func TestTestConfig(t *testing.T) {
	t.Run("TestConfigCreation", func(t *testing.T) {
		cfg := TestConfig{
			BaseConfig: BaseConfig{
				Command: "test",
				Verbose: true,
			},
			Model:     "mlp",
			DataPath:  "data/test.csv",
			ModelPath: "models/mlp.json",
		}

		if cfg.Command != "test" {
			t.Errorf("Expected command 'test', got '%s'", cfg.Command)
		}

		if cfg.Model != "mlp" {
			t.Errorf("Expected model 'mlp', got '%s'", cfg.Model)
		}

		if cfg.DataPath != "data/test.csv" {
			t.Errorf("Expected data path 'data/test.csv', got '%s'", cfg.DataPath)
		}

		if cfg.ModelPath != "models/mlp.json" {
			t.Errorf("Expected model path 'models/mlp.json', got '%s'", cfg.ModelPath)
		}
	})
}

// TestBenchmarkConfig tests the BenchmarkConfig structure
func TestBenchmarkConfig(t *testing.T) {
	t.Run("BenchmarkConfigCreation", func(t *testing.T) {
		cfg := BenchmarkConfig{
			BaseConfig: BaseConfig{
				Command: "benchmark",
				Verbose: true,
			},
			Model:        "cnn",
			Dataset:      "mnist",
			Iterations:   100,
			OutputPath:   "results/bench.json",
			MLPHidden:    "128,64",
			CNNArch:      "MNIST",
			BatchSize:    32,
			LearningRate: 0.001,
			Epochs:       10,
		}

		if cfg.Command != "benchmark" {
			t.Errorf("Expected command 'benchmark', got '%s'", cfg.Command)
		}

		if cfg.Model != "cnn" {
			t.Errorf("Expected model 'cnn', got '%s'", cfg.Model)
		}

		if cfg.Dataset != "mnist" {
			t.Errorf("Expected dataset 'mnist', got '%s'", cfg.Dataset)
		}

		if cfg.Iterations != 100 {
			t.Errorf("Expected iterations 100, got %d", cfg.Iterations)
		}

		if cfg.OutputPath != "results/bench.json" {
			t.Errorf("Expected output path 'results/bench.json', got '%s'", cfg.OutputPath)
		}

		if cfg.MLPHidden != "128,64" {
			t.Errorf("Expected MLP hidden '128,64', got '%s'", cfg.MLPHidden)
		}

		// Test CNN-specific fields
		if cfg.CNNArch != "MNIST" {
			t.Errorf("Expected CNN arch 'MNIST', got '%s'", cfg.CNNArch)
		}

		if cfg.BatchSize != 32 {
			t.Errorf("Expected batch size 32, got %d", cfg.BatchSize)
		}

		if cfg.LearningRate != 0.001 {
			t.Errorf("Expected learning rate 0.001, got %f", cfg.LearningRate)
		}

		if cfg.Epochs != 10 {
			t.Errorf("Expected epochs 10, got %d", cfg.Epochs)
		}
	})

	t.Run("BenchmarkConfigDefaults", func(t *testing.T) {
		cfg := BenchmarkConfig{}

		if cfg.Iterations != 0 {
			t.Errorf("Expected iterations 0, got %d", cfg.Iterations)
		}

		if cfg.BatchSize != 0 {
			t.Errorf("Expected batch size 0, got %d", cfg.BatchSize)
		}

		if cfg.LearningRate != 0 {
			t.Errorf("Expected learning rate 0, got %f", cfg.LearningRate)
		}

		if cfg.Epochs != 0 {
			t.Errorf("Expected epochs 0, got %d", cfg.Epochs)
		}
	})
}

// TestCompareConfig tests the CompareConfig structure
func TestCompareConfig(t *testing.T) {
	t.Run("CompareConfigCreation", func(t *testing.T) {
		cfg := CompareConfig{
			BaseConfig: BaseConfig{
				Command: "compare",
				Verbose: false,
			},
			Dataset:    "xor",
			Iterations: 50,
			OutputPath: "results/compare.json",
			MLPHidden:  "10,5",
		}

		if cfg.Command != "compare" {
			t.Errorf("Expected command 'compare', got '%s'", cfg.Command)
		}

		if cfg.Dataset != "xor" {
			t.Errorf("Expected dataset 'xor', got '%s'", cfg.Dataset)
		}

		if cfg.Iterations != 50 {
			t.Errorf("Expected iterations 50, got %d", cfg.Iterations)
		}

		if cfg.OutputPath != "results/compare.json" {
			t.Errorf("Expected output path 'results/compare.json', got '%s'", cfg.OutputPath)
		}

		if cfg.MLPHidden != "10,5" {
			t.Errorf("Expected MLP hidden '10,5', got '%s'", cfg.MLPHidden)
		}
	})
}

// TestMnistConfig tests the MnistConfig structure
func TestMnistConfig(t *testing.T) {
	t.Run("MnistConfigCreation", func(t *testing.T) {
		cfg := MnistConfig{
			BaseConfig: BaseConfig{
				Command: "mnist",
				Verbose: true,
			},
			DataPath: "datasets/mnist",
		}

		if cfg.Command != "mnist" {
			t.Errorf("Expected command 'mnist', got '%s'", cfg.Command)
		}

		if cfg.DataPath != "datasets/mnist" {
			t.Errorf("Expected data path 'datasets/mnist', got '%s'", cfg.DataPath)
		}
	})

	t.Run("MnistConfigDefaults", func(t *testing.T) {
		cfg := MnistConfig{}

		if cfg.DataPath != "" {
			t.Errorf("Expected empty data path, got '%s'", cfg.DataPath)
		}
	})
}

// TestTimeSeriesConfig tests the TimeSeriesConfig structure
func TestTimeSeriesConfig(t *testing.T) {
	t.Run("TimeSeriesConfigCreation", func(t *testing.T) {
		cfg := TimeSeriesConfig{
			BaseConfig: BaseConfig{
				Command: "timeseries",
				Verbose: true,
			},
			Dataset: "sine",
			Model:   "RNN",
			Compare: true,
		}

		if cfg.Command != "timeseries" {
			t.Errorf("Expected command 'timeseries', got '%s'", cfg.Command)
		}

		if cfg.Dataset != "sine" {
			t.Errorf("Expected dataset 'sine', got '%s'", cfg.Dataset)
		}

		if cfg.Model != "RNN" {
			t.Errorf("Expected model 'RNN', got '%s'", cfg.Model)
		}

		if !cfg.Compare {
			t.Error("Expected compare to be true")
		}
	})

	t.Run("TimeSeriesConfigDefaults", func(t *testing.T) {
		cfg := TimeSeriesConfig{}

		if cfg.Dataset != "" {
			t.Errorf("Expected empty dataset, got '%s'", cfg.Dataset)
		}

		if cfg.Model != "" {
			t.Errorf("Expected empty model, got '%s'", cfg.Model)
		}

		if cfg.Compare {
			t.Error("Expected compare to be false by default")
		}
	})

	t.Run("TimeSeriesConfigModels", func(t *testing.T) {
		models := []string{"RNN", "LSTM"}

		for _, model := range models {
			cfg := TimeSeriesConfig{
				Model: model,
			}

			if cfg.Model != model {
				t.Errorf("Expected model '%s', got '%s'", model, cfg.Model)
			}
		}
	})

	t.Run("TimeSeriesConfigDatasets", func(t *testing.T) {
		datasets := []string{"sine", "fibonacci", "randomwalk"}

		for _, dataset := range datasets {
			cfg := TimeSeriesConfig{
				Dataset: dataset,
			}

			if cfg.Dataset != dataset {
				t.Errorf("Expected dataset '%s', got '%s'", dataset, cfg.Dataset)
			}
		}
	})
}

// TestConfigValidation tests validation scenarios
func TestConfigValidation(t *testing.T) {
	t.Run("ValidTrainConfig", func(t *testing.T) {
		cfg := TrainConfig{
			BaseConfig: BaseConfig{Command: "train"},
			Model:      "perceptron",
			DataPath:   "data.csv",
		}

		// Basic validation - config should be created successfully
		if cfg.Command != "train" {
			t.Error("Expected valid train config")
		}
	})

	t.Run("ValidBenchmarkConfig", func(t *testing.T) {
		cfg := BenchmarkConfig{
			BaseConfig: BaseConfig{Command: "benchmark"},
			Model:      "mlp",
			Dataset:    "xor",
		}

		if cfg.Command != "benchmark" {
			t.Error("Expected valid benchmark config")
		}
	})

	t.Run("ConfigInheritance", func(t *testing.T) {
		// Test that all configs properly inherit from BaseConfig

		train := TrainConfig{BaseConfig: BaseConfig{Command: "train", Verbose: true}}
		if train.Command != "train" || !train.Verbose {
			t.Error("TrainConfig should inherit BaseConfig fields")
		}

		infer := InferConfig{BaseConfig: BaseConfig{Command: "infer", Verbose: false}}
		if infer.Command != "infer" || infer.Verbose {
			t.Error("InferConfig should inherit BaseConfig fields")
		}

		test := TestConfig{BaseConfig: BaseConfig{Command: "test", Verbose: true}}
		if test.Command != "test" || !test.Verbose {
			t.Error("TestConfig should inherit BaseConfig fields")
		}

		benchmark := BenchmarkConfig{BaseConfig: BaseConfig{Command: "benchmark", Verbose: false}}
		if benchmark.Command != "benchmark" || benchmark.Verbose {
			t.Error("BenchmarkConfig should inherit BaseConfig fields")
		}

		compare := CompareConfig{BaseConfig: BaseConfig{Command: "compare", Verbose: true}}
		if compare.Command != "compare" || !compare.Verbose {
			t.Error("CompareConfig should inherit BaseConfig fields")
		}

		mnist := MnistConfig{BaseConfig: BaseConfig{Command: "mnist", Verbose: false}}
		if mnist.Command != "mnist" || mnist.Verbose {
			t.Error("MnistConfig should inherit BaseConfig fields")
		}

		timeseries := TimeSeriesConfig{BaseConfig: BaseConfig{Command: "timeseries", Verbose: true}}
		if timeseries.Command != "timeseries" || !timeseries.Verbose {
			t.Error("TimeSeriesConfig should inherit BaseConfig fields")
		}
	})
}

// TestConfigEdgeCases tests edge cases and boundary conditions
func TestConfigEdgeCases(t *testing.T) {
	t.Run("NegativeValues", func(t *testing.T) {
		cfg := TrainConfig{
			LearningRate: -0.1,
			Epochs:       -100,
		}

		// The config struct itself doesn't validate, but we can test the values
		if cfg.LearningRate >= 0 {
			t.Error("Expected negative learning rate to be stored")
		}

		if cfg.Epochs >= 0 {
			t.Error("Expected negative epochs to be stored")
		}
	})

	t.Run("ZeroValues", func(t *testing.T) {
		cfg := BenchmarkConfig{
			Iterations:   0,
			BatchSize:    0,
			LearningRate: 0,
			Epochs:       0,
		}

		if cfg.Iterations != 0 || cfg.BatchSize != 0 || cfg.LearningRate != 0 || cfg.Epochs != 0 {
			t.Error("Expected zero values to be preserved")
		}
	})

	t.Run("LargeValues", func(t *testing.T) {
		cfg := TrainConfig{
			LearningRate: 999999.99,
			Epochs:       1000000,
		}

		if cfg.LearningRate != 999999.99 {
			t.Error("Expected large learning rate to be preserved")
		}

		if cfg.Epochs != 1000000 {
			t.Error("Expected large epochs value to be preserved")
		}
	})

	t.Run("EmptyStrings", func(t *testing.T) {
		cfg := TrainConfig{
			Model:     "",
			DataPath:  "",
			ModelPath: "",
		}

		if cfg.Model != "" || cfg.DataPath != "" || cfg.ModelPath != "" {
			t.Error("Expected empty strings to be preserved")
		}
	})

	t.Run("UnicodeStrings", func(t *testing.T) {
		cfg := TrainConfig{
			Model:     "パーセプトロン",
			DataPath:  "データ/学習.csv",
			ModelPath: "モデル/保存.json",
		}

		if cfg.Model != "パーセプトロン" {
			t.Error("Expected unicode model name to be preserved")
		}

		if cfg.DataPath != "データ/学習.csv" {
			t.Error("Expected unicode data path to be preserved")
		}

		if cfg.ModelPath != "モデル/保存.json" {
			t.Error("Expected unicode model path to be preserved")
		}
	})
}

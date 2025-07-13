// Package cli provides tests for command line parsing
package cli

import (
	"reflect"
	"testing"

	"github.com/nyasuto/bee/cmd/bee/internal/cli/config"
)

func TestParser_Parse_TrainCommand(t *testing.T) {
	parser := NewParser([]string{"bee", "train", "-data", "test.csv", "-lr", "0.05", "-epochs", "500", "-verbose"})

	cfg, err := parser.Parse()
	if err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}

	trainCfg, ok := cfg.(*config.TrainConfig)
	if !ok {
		t.Fatalf("Expected TrainConfig, got %T", cfg)
	}

	if trainCfg.Command != "train" {
		t.Errorf("Expected command 'train', got %s", trainCfg.Command)
	}
	if trainCfg.DataPath != "test.csv" {
		t.Errorf("Expected data path 'test.csv', got %s", trainCfg.DataPath)
	}
	if trainCfg.LearningRate != 0.05 {
		t.Errorf("Expected learning rate 0.05, got %f", trainCfg.LearningRate)
	}
	if trainCfg.Epochs != 500 {
		t.Errorf("Expected epochs 500, got %d", trainCfg.Epochs)
	}
	if !trainCfg.Verbose {
		t.Errorf("Expected verbose to be true")
	}
}

func TestParser_Parse_InferCommand(t *testing.T) {
	parser := NewParser([]string{"bee", "infer", "-model", "model.json", "-input", "1,0"})

	cfg, err := parser.Parse()
	if err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}

	inferCfg, ok := cfg.(*config.InferConfig)
	if !ok {
		t.Fatalf("Expected InferConfig, got %T", cfg)
	}

	if inferCfg.Command != "infer" {
		t.Errorf("Expected command 'infer', got %s", inferCfg.Command)
	}
	if inferCfg.ModelPath != "model.json" {
		t.Errorf("Expected model path 'model.json', got %s", inferCfg.ModelPath)
	}
	if inferCfg.InputData != "1,0" {
		t.Errorf("Expected input data '1,0', got %s", inferCfg.InputData)
	}
}

func TestParser_Parse_HelpCommand(t *testing.T) {
	parser := NewParser([]string{"bee", "help"})

	cfg, err := parser.Parse()
	if err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}

	baseCfg, ok := cfg.(*config.BaseConfig)
	if !ok {
		t.Fatalf("Expected BaseConfig, got %T", cfg)
	}

	if baseCfg.Command != "help" {
		t.Errorf("Expected command 'help', got %s", baseCfg.Command)
	}
}

func TestParser_Parse_UnknownCommand(t *testing.T) {
	parser := NewParser([]string{"bee", "unknown"})

	_, err := parser.Parse()
	if err == nil {
		t.Errorf("Expected error for unknown command")
	}
}

func TestParser_Parse_NoCommand(t *testing.T) {
	parser := NewParser([]string{"bee"})

	_, err := parser.Parse()
	if err == nil {
		t.Errorf("Expected error for no command")
	}
}

func TestParser_ParseAllCommands(t *testing.T) {
	testCases := []struct {
		name         string
		args         []string
		expectedType reflect.Type
		expectError  bool
	}{
		{
			name:         "TrainCommand",
			args:         []string{"bee", "train", "-data", "test.csv"},
			expectedType: reflect.TypeOf(&config.TrainConfig{}),
			expectError:  false,
		},
		{
			name:         "InferCommand",
			args:         []string{"bee", "infer", "-input", "1,0"},
			expectedType: reflect.TypeOf(&config.InferConfig{}),
			expectError:  false,
		},
		{
			name:         "TestCommand",
			args:         []string{"bee", "test", "-data", "test.csv"},
			expectedType: reflect.TypeOf(&config.TestConfig{}),
			expectError:  false,
		},
		{
			name:         "BenchmarkCommand",
			args:         []string{"bee", "benchmark"},
			expectedType: reflect.TypeOf(&config.BenchmarkConfig{}),
			expectError:  false,
		},
		{
			name:         "CompareCommand",
			args:         []string{"bee", "compare"},
			expectedType: reflect.TypeOf(&config.CompareConfig{}),
			expectError:  false,
		},
		{
			name:         "MnistCommand",
			args:         []string{"bee", "mnist"},
			expectedType: reflect.TypeOf(&config.MnistConfig{}),
			expectError:  false,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			parser := NewParser(tc.args)
			cfg, err := parser.Parse()

			if tc.expectError {
				if err == nil {
					t.Errorf("Expected error but got none")
				}
				return
			}

			if err != nil {
				t.Errorf("Unexpected error: %v", err)
				return
			}

			if reflect.TypeOf(cfg) != tc.expectedType {
				t.Errorf("Expected type %v, got %v", tc.expectedType, reflect.TypeOf(cfg))
			}
		})
	}
}

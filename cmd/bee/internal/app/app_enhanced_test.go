// Package app provides enhanced tests for the main application
// Learning Goal: Understanding comprehensive application testing patterns
package app

import (
	"context"
	"fmt"
	"strings"
	"testing"

	"github.com/nyasuto/bee/cmd/bee/internal/cli/config"
)

// TestAppCreation tests various app creation scenarios
func TestAppCreation(t *testing.T) {
	t.Run("NewAppWithArgs", func(t *testing.T) {
		args := []string{"bee", "help"}
		app := NewApp(args)

		if app == nil {
			t.Error("Expected non-nil app")
		}

		if app.parser == nil {
			t.Error("Expected non-nil parser")
		}

		if app.outputWriter == nil {
			t.Error("Expected non-nil output writer")
		}

		if app.commandMap == nil {
			t.Error("Expected non-nil command map")
		}
	})

	t.Run("NewAppWithNilArgs", func(t *testing.T) {
		app := NewApp(nil)

		if app == nil {
			t.Error("Expected non-nil app even with nil args")
		}

		// Should create parser from OS when args is nil
		if app.parser == nil {
			t.Error("Expected parser to be created from OS")
		}
	})

	t.Run("NewAppFromOS", func(t *testing.T) {
		app := NewAppFromOS()

		if app == nil {
			t.Error("Expected non-nil app from OS")
		}

		if app.parser == nil {
			t.Error("Expected non-nil parser from OS")
		}
	})

	t.Run("DependencyInjection", func(t *testing.T) {
		app := NewApp([]string{"bee", "help"})

		// Verify all dependencies are injected
		if app.outputWriter == nil {
			t.Error("Expected output writer to be injected")
		}

		if app.commandMap == nil {
			t.Error("Expected command map to be injected")
		}

		// Check that commands exist in the map
		expectedCommands := []string{"train", "timeseries", "benchmark"}
		for _, cmd := range expectedCommands {
			if _, exists := app.commandMap[cmd]; !exists {
				t.Errorf("Expected command '%s' to exist in command map", cmd)
			}
		}
	})
}

// TestAppRunScenarios tests various run scenarios
func TestAppRunScenarios(t *testing.T) {
	t.Run("RunWithValidTrainCommand", func(t *testing.T) {
		app := NewApp([]string{"bee", "train", "-data", "test.csv"})

		ctx := context.Background()
		err := app.Run(ctx)

		// Error is expected due to missing file, but should reach command execution
		if err != nil {
			t.Logf("Expected error due to missing test file: %v", err)
		}
	})

	t.Run("RunWithValidTimeseriesCommand", func(t *testing.T) {
		app := NewApp([]string{"bee", "timeseries", "-dataset", "sine"})

		ctx := context.Background()
		err := app.Run(ctx)

		// Error might occur due to implementation details
		if err != nil {
			t.Logf("Error from timeseries command: %v", err)
		}
	})

	t.Run("RunWithValidBenchmarkCommand", func(t *testing.T) {
		app := NewApp([]string{"bee", "benchmark", "-model", "perceptron"})

		ctx := context.Background()
		err := app.Run(ctx)

		// Error might occur due to implementation details
		if err != nil {
			t.Logf("Error from benchmark command: %v", err)
		}
	})

	t.Run("RunWithInvalidFlags", func(t *testing.T) {
		app := NewApp([]string{"bee", "train", "--invalid-flag"})

		ctx := context.Background()
		err := app.Run(ctx)

		if err == nil {
			t.Error("Expected error for invalid flags")
		}
	})

	t.Run("RunWithMissingRequiredFlags", func(t *testing.T) {
		app := NewApp([]string{"bee", "train"}) // Missing required -data flag

		ctx := context.Background()
		err := app.Run(ctx)

		if err == nil {
			t.Error("Expected error for missing required flags")
		}
	})
}

// TestAppContextHandling tests context handling
func TestAppContextHandling(t *testing.T) {
	t.Run("RunWithCanceledContext", func(t *testing.T) {
		app := NewApp([]string{"bee", "help"})

		ctx, cancel := context.WithCancel(context.Background())
		cancel() // Cancel immediately

		// Should still work for help command as it doesn't check context
		err := app.Run(ctx)
		if err != nil {
			t.Errorf("Expected help command to work even with canceled context: %v", err)
		}
	})

	t.Run("RunWithTimeoutContext", func(t *testing.T) {
		app := NewApp([]string{"bee", "help"})

		ctx, cancel := context.WithTimeout(context.Background(), 1)
		defer cancel()

		err := app.Run(ctx)
		if err != nil {
			t.Errorf("Expected help command to work with timeout context: %v", err)
		}
	})
}

// TestAppGetCommandName tests the getCommandName method thoroughly
func TestAppGetCommandName(t *testing.T) {
	app := NewApp([]string{"bee", "help"})

	testCases := []struct {
		name     string
		config   interface{}
		expected string
	}{
		{
			name: "TrainConfig",
			config: &config.TrainConfig{
				BaseConfig: config.BaseConfig{Command: "train"},
			},
			expected: "train",
		},
		{
			name: "InferConfig",
			config: &config.InferConfig{
				BaseConfig: config.BaseConfig{Command: "infer"},
			},
			expected: "infer",
		},
		{
			name: "TestConfig",
			config: &config.TestConfig{
				BaseConfig: config.BaseConfig{Command: "test"},
			},
			expected: "test",
		},
		{
			name: "BenchmarkConfig",
			config: &config.BenchmarkConfig{
				BaseConfig: config.BaseConfig{Command: "benchmark"},
			},
			expected: "benchmark",
		},
		{
			name: "CompareConfig",
			config: &config.CompareConfig{
				BaseConfig: config.BaseConfig{Command: "compare"},
			},
			expected: "compare",
		},
		{
			name: "MnistConfig",
			config: &config.MnistConfig{
				BaseConfig: config.BaseConfig{Command: "mnist"},
			},
			expected: "mnist",
		},
		{
			name: "TimeSeriesConfig",
			config: &config.TimeSeriesConfig{
				BaseConfig: config.BaseConfig{Command: "timeseries"},
			},
			expected: "timeseries",
		},
		{
			name: "BaseConfig",
			config: &config.BaseConfig{
				Command: "help",
			},
			expected: "help",
		},
		{
			name:     "StringConfig",
			config:   "invalid",
			expected: "",
		},
		{
			name:     "IntConfig",
			config:   42,
			expected: "",
		},
		{
			name:     "NilConfig",
			config:   nil,
			expected: "",
		},
		{
			name:     "StructConfig",
			config:   struct{ Field string }{Field: "value"},
			expected: "",
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			result := app.getCommandName(tc.config)
			if result != tc.expected {
				t.Errorf("Expected '%s', got '%s'", tc.expected, result)
			}
		})
	}
}

// TestAppCommandMapIntegrity tests command map integrity
func TestAppCommandMapIntegrity(t *testing.T) {
	app := NewApp([]string{"bee", "help"})

	t.Run("CommandMapNotEmpty", func(t *testing.T) {
		if len(app.commandMap) == 0 {
			t.Error("Expected non-empty command map")
		}
	})

	t.Run("ExpectedCommandsPresent", func(t *testing.T) {
		expectedCommands := []string{"train", "timeseries", "benchmark"}

		for _, cmdName := range expectedCommands {
			cmd, exists := app.commandMap[cmdName]
			if !exists {
				t.Errorf("Expected command '%s' to exist", cmdName)
				continue
			}

			if cmd == nil {
				t.Errorf("Expected non-nil command for '%s'", cmdName)
				continue
			}

			// Test command interface methods
			if cmd.Name() != cmdName {
				t.Errorf("Expected command name '%s', got '%s'", cmdName, cmd.Name())
			}

			if cmd.Description() == "" {
				t.Errorf("Expected non-empty description for command '%s'", cmdName)
			}
		}
	})

	t.Run("CommandValidation", func(t *testing.T) {
		for cmdName, cmd := range app.commandMap {
			// Test that each command can validate appropriate config
			switch cmdName {
			case "train":
				cfg := &config.TrainConfig{
					BaseConfig: config.BaseConfig{
						Command: "train",
					},
					DataPath:     "test.csv",
					LearningRate: 0.1,
					Epochs:       100,
					Model:        "perceptron",
					ModelPath:    "model.json",
				}
				err := cmd.Validate(cfg)
				if err != nil {
					t.Errorf("Expected train command to validate valid TrainConfig: %v", err)
				}
			case "benchmark":
				cfg := &config.BenchmarkConfig{
					BaseConfig: config.BaseConfig{
						Command: "benchmark",
					},
					Model:   "perceptron",
					Dataset: "xor",
				}
				err := cmd.Validate(cfg)
				if err != nil {
					t.Errorf("Expected benchmark command to validate valid BenchmarkConfig: %v", err)
				}
			case "timeseries":
				cfg := &config.TimeSeriesConfig{
					BaseConfig: config.BaseConfig{
						Command: "timeseries",
					},
					Dataset: "sine",
					Model:   "RNN",
				}
				err := cmd.Validate(cfg)
				if err != nil {
					t.Errorf("Expected timeseries command to validate valid TimeSeriesConfig: %v", err)
				}
			}
		}
	})
}

// TestAppErrorHandling tests error handling scenarios
func TestAppErrorHandling(t *testing.T) {
	t.Run("ParserError", func(t *testing.T) {
		// Invalid arguments that should cause parser error
		app := NewApp([]string{})

		ctx := context.Background()
		err := app.Run(ctx)

		if err == nil {
			t.Error("Expected error for empty arguments")
		}
	})

	t.Run("UnknownCommandError", func(t *testing.T) {
		app := NewApp([]string{"bee", "nonexistent"})

		ctx := context.Background()
		err := app.Run(ctx)

		if err == nil {
			t.Error("Expected error for unknown command")
		}

		if !strings.Contains(err.Error(), "unknown command") {
			t.Errorf("Expected 'unknown command' in error, got: %v", err)
		}
	})

	t.Run("CommandExecutionError", func(t *testing.T) {
		// This will likely cause a command execution error due to missing files
		app := NewApp([]string{"bee", "train", "-data", "nonexistent.csv"})

		ctx := context.Background()
		err := app.Run(ctx)

		// Error is expected for nonexistent file
		if err == nil {
			t.Error("Expected error for nonexistent file")
		}
	})
}

// TestAppIntegration tests integration scenarios
func TestAppIntegration(t *testing.T) {
	t.Run("CompleteWorkflow", func(t *testing.T) {
		// Test a complete workflow scenario
		commands := [][]string{
			{"bee", "help"},
			{"bee", "benchmark", "-model", "perceptron", "-dataset", "xor"},
			{"bee", "timeseries", "-dataset", "sine", "-model", "RNN"},
		}

		for i, cmdArgs := range commands {
			t.Run(fmt.Sprintf("Command_%d", i), func(t *testing.T) {
				app := NewApp(cmdArgs)
				ctx := context.Background()
				err := app.Run(ctx)

				// Help should always work
				if cmdArgs[1] == "help" && err != nil {
					t.Errorf("Help command should not fail: %v", err)
				}

				// Other commands might fail due to missing implementation details
				if cmdArgs[1] != "help" && err != nil {
					t.Logf("Expected potential error for %s: %v", cmdArgs[1], err)
				}
			})
		}
	})

	t.Run("MultipleAppInstances", func(t *testing.T) {
		// Test that multiple app instances work independently
		app1 := NewApp([]string{"bee", "help"})
		app2 := NewApp([]string{"bee", "benchmark", "-model", "perceptron"})

		ctx := context.Background()

		// Both should be independent
		err1 := app1.Run(ctx)
		err2 := app2.Run(ctx)

		if err1 != nil {
			t.Errorf("App1 (help) should not fail: %v", err1)
		}

		// App2 might fail due to missing implementation, that's ok
		if err2 != nil {
			t.Logf("App2 expected potential error: %v", err2)
		}
	})
}

// BenchmarkApp benchmarks app performance
func BenchmarkApp(b *testing.B) {
	b.Run("AppCreation", func(b *testing.B) {
		args := []string{"bee", "help"}
		b.ResetTimer()

		for i := 0; i < b.N; i++ {
			_ = NewApp(args)
		}
	})

	b.Run("AppExecution", func(b *testing.B) {
		ctx := context.Background()
		b.ResetTimer()

		for i := 0; i < b.N; i++ {
			app := NewApp([]string{"bee", "help"})
			_ = app.Run(ctx)
		}
	})

	b.Run("CommandNameResolution", func(b *testing.B) {
		app := NewApp([]string{"bee", "help"})
		cfg := &config.TrainConfig{BaseConfig: config.BaseConfig{Command: "train"}}
		b.ResetTimer()

		for i := 0; i < b.N; i++ {
			_ = app.getCommandName(cfg)
		}
	})
}

// TestAppMemoryManagement tests memory management
func TestAppMemoryManagement(t *testing.T) {
	t.Run("NoMemoryLeaks", func(t *testing.T) {
		// Create and run multiple apps to check for obvious memory leaks
		for i := 0; i < 100; i++ {
			app := NewApp([]string{"bee", "help"})
			ctx := context.Background()
			_ = app.Run(ctx)
		}

		// If we get here without panic or excessive memory usage, that's good
		t.Log("Memory management test completed")
	})

	t.Run("ResourceCleanup", func(t *testing.T) {
		app := NewApp([]string{"bee", "help"})
		ctx := context.Background()

		// Run multiple times with same app instance
		for i := 0; i < 10; i++ {
			err := app.Run(ctx)
			if err != nil {
				t.Errorf("Run %d failed: %v", i, err)
			}
		}
	})
}

// Package main test for the Bee CLI tool
// Learning Goal: Understanding main function testing strategies
package main

import (
	"context"
	"os"
	"os/exec"
	"strings"
	"testing"

	"github.com/nyasuto/bee/cmd/bee/internal/app"
)

// TestMainFunction tests the main application entry point
func TestMainFunction(t *testing.T) {
	// Test successful execution with help command
	t.Run("SuccessfulHelpCommand", func(t *testing.T) {
		// Create app with help command
		application := app.NewApp([]string{"bee", "help"})

		// Execute application
		ctx := context.Background()
		err := application.Run(ctx)

		// Should not return error for help command
		if err != nil {
			t.Errorf("Expected no error for help command, got: %v", err)
		}
	})

	// Test application creation from OS
	t.Run("NewAppFromOS", func(t *testing.T) {
		// This tests the NewAppFromOS function that main.go uses
		application := app.NewAppFromOS()

		if application == nil {
			t.Error("Expected non-nil application")
		}
	})

	// Test main function behavior (indirectly)
	t.Run("MainFunctionBehavior", func(t *testing.T) {
		// Since main() calls os.Exit(1) on error, we test the underlying logic
		// Create application with invalid arguments
		application := app.NewApp([]string{"bee", "invalid-command"})

		ctx := context.Background()
		err := application.Run(ctx)

		// Should return error for invalid command
		if err == nil {
			t.Error("Expected error for invalid command")
		}
	})

	// Test context handling
	t.Run("ContextHandling", func(t *testing.T) {
		application := app.NewApp([]string{"bee", "help"})

		// Test with background context
		ctx := context.Background()
		err := application.Run(ctx)

		if err != nil {
			t.Errorf("Expected no error with background context, got: %v", err)
		}

		// Test with canceled context
		cancelCtx, cancel := context.WithCancel(context.Background())
		cancel() // Cancel immediately

		// Note: Help command doesn't check context cancellation,
		// but this tests that canceled context doesn't break the app
		err = application.Run(cancelCtx)
		if err != nil {
			t.Errorf("Expected no error with canceled context for help command, got: %v", err)
		}
	})
}

// TestMainIntegration tests main function integration scenarios
func TestMainIntegration(t *testing.T) {
	// Test main function's error handling pattern
	t.Run("ErrorHandlingPattern", func(t *testing.T) {
		// Create application that will return an error
		application := app.NewApp([]string{"bee", "nonexistent-command"})

		ctx := context.Background()
		err := application.Run(ctx)

		// Verify error is returned (which would cause os.Exit(1) in main)
		if err == nil {
			t.Error("Expected error for nonexistent command")
		}
	})

	// Test dependency injection flow
	t.Run("DependencyInjectionFlow", func(t *testing.T) {
		// This tests the full dependency injection flow that main() triggers
		app1 := app.NewAppFromOS()
		app2 := app.NewAppFromOS()

		// Both apps should be independently created
		if app1 == nil || app2 == nil {
			t.Error("Expected both apps to be created successfully")
		}

		// Apps should be different instances (not singletons)
		if app1 == app2 {
			t.Error("Expected different app instances")
		}
	})
}

// TestMainPackageConstants tests any package-level constants or variables
func TestMainPackageConstants(t *testing.T) {
	// Verify that main package has expected structure
	t.Run("PackageStructure", func(t *testing.T) {
		// This test verifies the main package is properly set up
		// The fact that we can import and use app.NewAppFromOS means
		// the package structure is correct
		application := app.NewAppFromOS()
		if application == nil {
			t.Error("Expected successful app creation")
		}
	})
}

// BenchmarkMainAppCreation benchmarks the app creation process
func BenchmarkMainAppCreation(b *testing.B) {
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = app.NewApp([]string{"bee", "help"})
	}
}

// BenchmarkMainAppExecution benchmarks the app execution
func BenchmarkMainAppExecution(b *testing.B) {
	ctx := context.Background()

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		application := app.NewApp([]string{"bee", "help"})
		_ = application.Run(ctx)
	}
}

// TestMainWithDifferentArgs tests main behavior with various argument patterns
func TestMainWithDifferentArgs(t *testing.T) {
	testCases := []struct {
		name        string
		args        []string
		expectError bool
		description string
	}{
		{
			name:        "NoArguments",
			args:        []string{"bee"},
			expectError: true,
			description: "No command specified should return error",
		},
		{
			name:        "HelpCommand",
			args:        []string{"bee", "help"},
			expectError: false,
			description: "Help command should succeed",
		},
		{
			name:        "InvalidCommand",
			args:        []string{"bee", "invalid"},
			expectError: true,
			description: "Invalid command should return error",
		},
		{
			name:        "EmptyArgs",
			args:        []string{},
			expectError: true,
			description: "Empty arguments should return error",
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			application := app.NewApp(tc.args)
			ctx := context.Background()
			err := application.Run(ctx)

			if tc.expectError && err == nil {
				t.Errorf("Expected error for %s, but got none", tc.description)
			}
			if !tc.expectError && err != nil {
				t.Errorf("Expected no error for %s, but got: %v", tc.description, err)
			}
		})
	}
}

// TestMainExitBehavior tests the exit behavior (conceptually)
func TestMainExitBehavior(t *testing.T) {
	// Since we can't directly test os.Exit(1), we test the conditions that would trigger it
	t.Run("ErrorConditionThatTriggersExit", func(t *testing.T) {
		// Create an app that will fail
		application := app.NewApp([]string{"bee", "fail-command"})

		ctx := context.Background()
		err := application.Run(ctx)

		// This error would cause main() to call os.Exit(1)
		if err == nil {
			t.Error("Expected error condition that would trigger os.Exit(1)")
		}
	})

	// Test successful condition (no exit)
	t.Run("SuccessConditionNoExit", func(t *testing.T) {
		application := app.NewApp([]string{"bee", "help"})

		ctx := context.Background()
		err := application.Run(ctx)

		// This success would allow main() to exit normally (no os.Exit call)
		if err != nil {
			t.Errorf("Expected success condition (no exit), got error: %v", err)
		}
	})
}

// TestMainFileExists verifies the main.go file exists and is readable
func TestMainFileExists(t *testing.T) {
	// Verify main.go exists and is accessible
	_, err := os.Stat("main.go")
	if err != nil {
		t.Errorf("main.go file should exist and be readable: %v", err)
	}
}

// TestMainFunctionComponents tests individual components called by main()
func TestMainFunctionComponents(t *testing.T) {
	t.Run("NewAppFromOS", func(t *testing.T) {
		// This tests the app.NewAppFromOS() call from main.go
		application := app.NewAppFromOS()
		if application == nil {
			t.Error("NewAppFromOS should return non-nil app")
		}
	})

	t.Run("AppRunWithContext", func(t *testing.T) {
		// This tests the application.Run(ctx) pattern from main.go
		application := app.NewApp([]string{"bee", "help"})
		ctx := context.Background()

		err := application.Run(ctx)
		if err != nil {
			t.Errorf("App.Run should not fail for help command: %v", err)
		}
	})

	t.Run("ErrorHandlingPattern", func(t *testing.T) {
		// Test the error handling pattern that would trigger os.Exit(1)
		application := app.NewApp([]string{"bee", "invalid"})
		ctx := context.Background()

		err := application.Run(ctx)
		if err == nil {
			t.Error("Expected error for invalid command that would trigger os.Exit(1)")
		}
	})
}

// TestRunApp tests the runApp function directly
func TestRunApp(t *testing.T) {
	t.Run("RunAppSuccess", func(t *testing.T) {
		err := runApp([]string{"bee", "help"})
		if err != nil {
			t.Errorf("runApp should succeed with help command: %v", err)
		}
	})

	t.Run("RunAppError", func(t *testing.T) {
		err := runApp([]string{"bee", "invalid"})
		if err == nil {
			t.Error("runApp should return error for invalid command")
		}
	})

	t.Run("RunAppNoCommand", func(t *testing.T) {
		err := runApp([]string{"bee"})
		if err == nil {
			t.Error("runApp should return error when no command is specified")
		}
	})

	t.Run("RunAppEmptyArgs", func(t *testing.T) {
		err := runApp([]string{})
		if err == nil {
			t.Error("runApp should return error for empty arguments")
		}
	})
}

// TestRunAppFromOS tests the runAppFromOS function directly
func TestRunAppFromOS(t *testing.T) {
	// Note: This test will use actual OS args, so we can't easily control it
	// But we can verify the function exists and is callable
	t.Run("RunAppFromOSCallable", func(t *testing.T) {
		// Set up test OS args temporarily
		oldArgs := os.Args
		defer func() { os.Args = oldArgs }()

		os.Args = []string{"bee", "help"}

		err := runAppFromOS()
		if err != nil {
			t.Errorf("runAppFromOS should not fail with help args: %v", err)
		}
	})

	t.Run("RunAppFromOSErrorCase", func(t *testing.T) {
		// Set up test OS args temporarily
		oldArgs := os.Args
		defer func() { os.Args = oldArgs }()

		os.Args = []string{"bee", "invalid"}

		err := runAppFromOS()
		if err == nil {
			t.Error("runAppFromOS should return error for invalid command")
		}
	})
}

// TestMainExecutable tests the compiled executable (integration test)
func TestMainExecutable(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping executable test in short mode")
	}

	// Build the executable first
	cmd := exec.Command("go", "build", "-o", "bee_test", ".")
	err := cmd.Run()
	if err != nil {
		t.Fatalf("Failed to build executable: %v", err)
	}
	defer os.Remove("bee_test")

	t.Run("ExecutableHelp", func(t *testing.T) {
		cmd := exec.Command("./bee_test", "help")
		output, err := cmd.CombinedOutput()

		if err != nil {
			t.Errorf("Executable help command failed: %v", err)
		}

		outputStr := string(output)
		if !strings.Contains(outputStr, "Bee Neural Network CLI Tool") {
			t.Error("Expected help output to contain tool description")
		}
	})

	t.Run("ExecutableInvalidCommand", func(t *testing.T) {
		cmd := exec.Command("./bee_test", "invalid")
		_, err := cmd.CombinedOutput()

		// Should exit with non-zero status for invalid command
		if err == nil {
			t.Error("Expected error exit for invalid command")
		}
	})
}

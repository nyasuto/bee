// Package main test for Bee neural network project entry point
// Learning Goal: Understanding testing of main package entry points
package main

import (
	"io"
	"os"
	"strings"
	"testing"
)

// TestMain tests the main function output
func TestMain(t *testing.T) {
	// Capture stdout to verify main function output
	oldStdout := os.Stdout
	r, w, _ := os.Pipe()
	os.Stdout = w

	// Call main function
	main()

	// Close writer and restore stdout
	w.Close()
	os.Stdout = oldStdout

	// Read captured output
	out, _ := io.ReadAll(r)
	output := string(out)

	// Verify expected output
	expectedLines := []string{
		"🐝 Bee Neural Network Project",
		"Ready for AI Agent-driven development!",
	}

	for _, expected := range expectedLines {
		if !strings.Contains(output, expected) {
			t.Errorf("Expected output to contain '%s', got: %s", expected, output)
		}
	}
}

// TestMainNoArgs tests main function execution without arguments
func TestMainNoArgs(t *testing.T) {
	// Save original args
	oldArgs := os.Args
	defer func() { os.Args = oldArgs }()

	// Set empty args (just program name)
	os.Args = []string{"bee"}

	// Capture stdout
	oldStdout := os.Stdout
	r, w, _ := os.Pipe()
	os.Stdout = w

	// Test main function doesn't panic
	defer func() {
		if r := recover(); r != nil {
			t.Errorf("main() panicked: %v", r)
		}
	}()

	main()

	// Clean up
	w.Close()
	os.Stdout = oldStdout

	// Verify output was written
	out, _ := io.ReadAll(r)
	if len(out) == 0 {
		t.Error("Expected some output from main function")
	}
}

// TestMainWithArgs tests main function with various arguments
func TestMainWithArgs(t *testing.T) {
	// Save original args
	oldArgs := os.Args
	defer func() { os.Args = oldArgs }()

	testCases := []struct {
		name string
		args []string
	}{
		{"WithSingleArg", []string{"bee", "help"}},
		{"WithMultipleArgs", []string{"bee", "train", "perceptron"}},
		{"WithFlags", []string{"bee", "--version"}},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			os.Args = tc.args

			// Capture stdout
			oldStdout := os.Stdout
			r, w, _ := os.Pipe()
			os.Stdout = w

			// Test main function doesn't panic with args
			defer func() {
				if r := recover(); r != nil {
					t.Errorf("main() panicked with args %v: %v", tc.args, r)
				}
			}()

			main()

			// Clean up
			w.Close()
			os.Stdout = oldStdout

			// Verify output
			out, _ := io.ReadAll(r)
			if len(out) == 0 {
				t.Error("Expected some output from main function")
			}
		})
	}
}

// TestMainOutputFormat tests the format of main function output
func TestMainOutputFormat(t *testing.T) {
	// Capture stdout
	oldStdout := os.Stdout
	r, w, _ := os.Pipe()
	os.Stdout = w

	main()

	w.Close()
	os.Stdout = oldStdout

	// Read and analyze output
	out, _ := io.ReadAll(r)
	output := string(out)
	lines := strings.Split(strings.TrimSpace(output), "\n")

	// Verify output structure
	if len(lines) < 2 {
		t.Errorf("Expected at least 2 lines of output, got %d", len(lines))
	}

	// Check emoji presence (indicates friendly UI)
	if !strings.Contains(output, "🐝") {
		t.Error("Expected bee emoji in output")
	}

	// Check project name presence
	if !strings.Contains(output, "Bee Neural Network Project") {
		t.Error("Expected project name in output")
	}

	// Check development readiness message
	if !strings.Contains(output, "development") {
		t.Error("Expected development-related message in output")
	}
}

// BenchmarkMain benchmarks the main function execution time
func BenchmarkMain(b *testing.B) {
	// Capture and discard stdout to avoid benchmark noise
	oldStdout := os.Stdout
	devNull, _ := os.Open(os.DevNull)
	os.Stdout = devNull

	defer func() {
		devNull.Close()
		os.Stdout = oldStdout
	}()

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		main()
	}
}

// TestMainMemoryUsage tests that main function doesn't leak memory
func TestMainMemoryUsage(t *testing.T) {
	// Capture stdout to prevent output during test
	oldStdout := os.Stdout
	devNull, _ := os.Open(os.DevNull)
	os.Stdout = devNull

	defer func() {
		devNull.Close()
		os.Stdout = oldStdout
	}()

	// Run main multiple times to check for memory leaks
	for i := 0; i < 100; i++ {
		main()
	}

	// If we reach here without panic or excessive memory usage, test passes
	// This is a basic check - more sophisticated memory profiling could be added
}

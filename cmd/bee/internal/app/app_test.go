// Package app provides tests for the main application
package app

import (
	"context"
	"strings"
	"testing"
)

func TestApp_Run_HelpCommand(t *testing.T) {
	// Test help command
	app := NewApp([]string{"bee", "help"})

	ctx := context.Background()
	err := app.Run(ctx)
	if err != nil {
		t.Errorf("Expected no error for help command, got: %v", err)
	}
}

func TestApp_Run_UnknownCommand(t *testing.T) {
	// Test unknown command
	app := NewApp([]string{"bee", "unknown"})

	ctx := context.Background()
	err := app.Run(ctx)
	if err == nil {
		t.Errorf("Expected error for unknown command")
	}

	if !strings.Contains(err.Error(), "unknown command") {
		t.Errorf("Expected 'unknown command' error, got: %v", err)
	}
}

func TestApp_Run_NoCommand(t *testing.T) {
	// Test no command specified
	app := NewApp([]string{"bee"})

	ctx := context.Background()
	err := app.Run(ctx)
	if err == nil {
		t.Errorf("Expected error for no command")
	}
}

func TestApp_getCommandName(t *testing.T) {
	app := NewApp([]string{"bee", "help"})

	testCases := []struct {
		name     string
		config   interface{}
		expected string
	}{
		{
			name: "TrainConfig",
			config: &struct {
				Command string
			}{Command: "train"},
			expected: "",
		},
		{
			name:     "InvalidConfig",
			config:   "invalid",
			expected: "",
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			result := app.getCommandName(tc.config)
			if result != tc.expected {
				t.Errorf("Expected %s, got %s", tc.expected, result)
			}
		})
	}
}
